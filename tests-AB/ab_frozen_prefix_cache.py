"""Whether scoring every chunk against one frozen prefix cache keeps the logits and lowers the peak.

A frozen prefix cache joins the broadcast prefix to each chunk one layer at a time and
stores no copy per chunk. Every branch's raw logit row must come back bitwise equal,
and the VRAM peak is what the change is for. Run the measurement once on the commit
before the change and once after, on one idle GPU, then compare the two files.

    uv run python tests-AB/ab_frozen_prefix_cache.py --out before.json
    uv run python tests-AB/ab_frozen_prefix_cache.py --out after.json
    uv run python tests-AB/ab_frozen_prefix_cache.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# ab_env goes first so this checkout's src outranks an editable install, which the run on
# the commit before the change depends on.
import ab_env
import numpy as np
import torch
from ab_branch_packing import SHAPES, capture_logits
from benchmark import make_request, make_state
from PIL import Image

from logit_classifier.classifier import Classifier
from logit_classifier.config import Config
from logit_classifier.schema import SystemOneRequest, parse_request

LONG_STATE_TOKENS = 8192
LONG_STATE_QUESTIONS = 8
IMAGE = ab_env.INPUTS / "top-L.png"
IMAGE_REQUEST = {"state": "", "questions": {"subject": {
    "type": "choice", "instructions": "What is the main subject of this image",
    "criteria": dict.fromkeys(["a person", "an animal", "a building", "a landscape", "an object"])}}}
GIB = 1024**3


def request_set(classifier: Classifier) -> dict[str, tuple[SystemOneRequest, Any]]:
    """Every measured request in a fixed order, each with its decoded image or None."""
    requests: dict[str, tuple[SystemOneRequest, Any]] = {
        name: (parse_request(body), None) for name, body in SHAPES.items()
    }
    long_state = make_state(classifier.backend, LONG_STATE_TOKENS)
    requests[f"{LONG_STATE_TOKENS} token state, {LONG_STATE_QUESTIONS} questions"] = (
        make_request(long_state, LONG_STATE_QUESTIONS), None)
    requests[f"image state, {IMAGE.name}"] = (
        parse_request(IMAGE_REQUEST), Image.open(IMAGE).convert("RGB"))
    return requests


def measure(out: Path) -> int:
    classifier = Classifier(Config(use_prior_debias=False, permutations=1))
    captured = capture_logits(classifier)
    results: dict[str, dict[str, Any]] = {}
    print(f"model {classifier.config.model_id}, device {torch.cuda.get_device_name(0)}\n")

    for name, (request, picture) in request_set(classifier).items():
        captured.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        classifier.classify(request, image=picture)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        results[name] = {"peak_bytes": peak, "rows": [row.tolist() for row in captured]}
        print(f"  {name[:52]:52}  {len(captured):4d} rows   peak {peak / GIB:6.2f} GiB")

    # json writes a float64 as its shortest round trip repr, so the rows reload bit for bit.
    out.write_text(json.dumps({"model": classifier.config.model_id, "requests": results}),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def bitwise_equal(old_rows: list[list[float]], new_rows: list[list[float]]) -> bool:
    if len(old_rows) != len(new_rows):
        return False
    return all(np.asarray(old, dtype=np.float64).tobytes() == np.asarray(new, dtype=np.float64).tobytes()
               for old, new in zip(old_rows, new_rows, strict=True))


def compare(before: Path, after: Path) -> int:
    old = json.loads(before.read_text(encoding="utf-8"))
    new = json.loads(after.read_text(encoding="utf-8"))
    identical = 0

    if old["model"] != new["model"]:
        print(f"different models: {old['model']} against {new['model']}")
        return 1
    if list(old["requests"]) != list(new["requests"]):
        print("the two files measured different request sets")
        return 1

    header = f"{'request':52} {'before GiB':>10} {'after GiB':>9}  logits"
    print(header)
    print("-" * len(header))
    for name, earlier in old["requests"].items():
        later = new["requests"][name]
        same = bitwise_equal(earlier["rows"], later["rows"])
        identical += same
        print(f"{name[:52]:52} {earlier['peak_bytes'] / GIB:10.2f} {later['peak_bytes'] / GIB:9.2f}  "
              f"{'bitwise identical' if same else 'DIFFERENT'}")
    print(f"\n{identical} of {len(old['requests'])} requests came back bitwise identical")
    return 0 if identical == len(old["requests"]) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path, help="measure every request and write the rows and peaks here")
    mode.add_argument("--compare", type=Path, nargs=2, metavar=("BEFORE", "AFTER"),
                      help="report the bitwise result and both peaks for two written files")
    args = parser.parse_args()

    if args.compare:
        return compare(*args.compare)
    return measure(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
