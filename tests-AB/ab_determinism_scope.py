"""Prove that scoping the torch determinism globals changes no logit.

backends/hf.py used to pin the torch determinism globals at load and never put them
back, which changed them for every other model in a ComfyUI process. They are now set
around each forward pass and restored after. This checks that the swap cost nothing.

The control is the point of the script. If a hostile host state with the window removed
still produced identical logits, then the settings do not reach this forward at all and
the checks above it would prove nothing.

The sweep then flips one setting at a time, with the window removed, to find which ones
this model's forward pass actually reads. A setting that moves no logit is a setting the
window does not need to hold.

    uv run python tests-AB/ab_determinism_scope.py
    uv run python tests-AB/ab_determinism_scope.py --model Qwen/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np

from logit_classifier.backends import hf
from logit_classifier.classifier import Classifier
from logit_classifier.config import Config
from logit_classifier.schema import parse_request

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

# What the window holds during a forward pass.
PINNED = {"benchmark": False, "deterministic": True, "bf16": False,
          "matmul": "highest", "fp32": "ieee"}

# The opposite of every pinned value, so an escaped forward pass has the best chance
# of diverging. A host indifferent to determinism lands somewhere in here. The cuda
# matmul slot takes only none, ieee or tf32, so tf32 is as far from ieee as it goes.
HOSTILE = {"benchmark": True, "deterministic": False, "bf16": True,
           "matmul": "high", "fp32": "tf32"}

# set_float32_matmul_precision and the fp32_precision slots are one setting under two
# names, so the sweep flips them together rather than reporting a vacuous row.
SWEEP = {
    "cudnn.benchmark": ["benchmark"],
    "cudnn.deterministic": ["deterministic"],
    "allow_bf16_reduced_precision_reduction": ["bf16"],
    "float32 matmul precision": ["matmul", "fp32"],
}


def snapshot() -> dict:
    mkldnn = getattr(hf.torch.backends.mkldnn, "matmul", None)
    return {
        "benchmark": hf.torch.backends.cudnn.benchmark,
        "deterministic": hf.torch.backends.cudnn.deterministic,
        "bf16": hf.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "matmul": hf.torch.get_float32_matmul_precision(),
        "cuda_fp32": hf.torch.backends.cuda.matmul.fp32_precision,
        "mkldnn_fp32": getattr(mkldnn, "fp32_precision", None),
    }


def apply(state: dict) -> None:
    mkldnn = getattr(hf.torch.backends.mkldnn, "matmul", None)
    hf.torch.backends.cudnn.benchmark = state["benchmark"]
    hf.torch.backends.cudnn.deterministic = state["deterministic"]
    hf.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = state["bf16"]
    hf.torch.set_float32_matmul_precision(state["matmul"])
    hf.torch.backends.cuda.matmul.fp32_precision = state["fp32"]
    if mkldnn is not None:
        mkldnn.fp32_precision = state["fp32"]


def restore(state: dict) -> None:
    """Put back a snapshot, which carries the two fp32 slots separately."""
    mkldnn = getattr(hf.torch.backends.mkldnn, "matmul", None)
    hf.torch.backends.cudnn.benchmark = state["benchmark"]
    hf.torch.backends.cudnn.deterministic = state["deterministic"]
    hf.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = state["bf16"]
    hf.torch.set_float32_matmul_precision(state["matmul"])
    hf.torch.backends.cuda.matmul.fp32_precision = state["cuda_fp32"]
    if mkldnn is not None and state["mkldnn_fp32"] is not None:
        mkldnn.fp32_precision = state["mkldnn_fp32"]


@contextlib.contextmanager
def no_window():
    """Remove the window so the process globals reach the forward pass directly."""
    saved = hf._determinism
    hf._determinism = contextlib.nullcontext
    try:
        yield
    finally:
        hf._determinism = saved


def recorder(classifier: Classifier) -> list:
    """Capture every logit row the backend returns, without changing the real path."""
    captured: list = []
    original = classifier.backend.score

    def wrapped(*args, **kwargs):
        rows = original(*args, **kwargs)
        captured.extend(np.asarray(row.z, dtype=np.float64) for row in rows)
        return rows

    classifier.backend.score = wrapped  # type: ignore[method-assign]
    return captured


def run(classifier: Classifier, captured: list, requests: list) -> np.ndarray:
    captured.clear()
    for request in requests:
        classifier.classify(request)
    return np.concatenate([row.ravel() for row in captured])


def identical(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and bool(np.array_equal(left, right))


def gap(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=Config().model_id)
    args = parser.parse_args()

    requests = [
        parse_request(json.loads((FIXTURES / name).read_text(encoding="utf-8")))
        for name in ("quickstart_request.json", "many_options_request.json")
    ]

    # A prior that learns between runs would change the answer on its own, and the
    # question here is only whether the forward pass changed.
    config = Config(model_id=args.model, use_prior_debias=False, permutations=1)
    print(f"loading {config.model_id}", flush=True)
    classifier = Classifier(config)
    captured = recorder(classifier)

    host = snapshot()
    print(f"host state before any run : {host}")

    apply(HOSTILE)
    hostile_run = run(classifier, captured, requests)
    left_behind = snapshot()
    print(f"rows captured per run     : {len(captured)}")

    repeat_run = run(classifier, captured, requests)

    apply(PINNED)
    pinned_run = run(classifier, captured, requests)

    with no_window():
        apply(HOSTILE)
        control_run = run(classifier, captured, requests)

    expected = dict(HOSTILE)
    checks = [
        ("window restores every slot it touched",
         all(left_behind[k] == expected[k] for k in ("benchmark", "deterministic", "bf16")),
         f"left behind {left_behind}"),
        ("same request returns the same logits", identical(hostile_run, repeat_run),
         f"max abs gap {gap(hostile_run, repeat_run):.3e}"),
        ("no forward pass escapes the window", identical(hostile_run, pinned_run),
         f"max abs gap {gap(hostile_run, pinned_run):.3e}"),
        ("CONTROL, the settings do reach this forward",
         not identical(hostile_run, control_run),
         f"max abs gap {gap(hostile_run, control_run):.3e}"),
    ]

    print()
    failed = 0
    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL':4}  {name:44} {detail}")
        failed += not passed

    # Which settings this model's forward pass actually reads. Each row flips one
    # setting away from pinned, with the window removed, and reports the logit shift.
    print()
    print("  sweep, one setting flipped at a time, window removed:")
    with no_window():
        apply(PINNED)
        base = run(classifier, captured, requests)
        for name, keys in SWEEP.items():
            apply({**PINNED, **{key: HOSTILE[key] for key in keys}})
            shift = gap(base, run(classifier, captured, requests))
            reads = "reads it" if shift > 0 else "no effect"
            print(f"    {name:40} max abs gap {shift:.3e}  {reads}")

    restore(host)

    print()
    if failed:
        print(f"{failed} of {len(checks)} checks failed")
        return 1
    print(f"all {len(checks)} checks passed on {config.model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
