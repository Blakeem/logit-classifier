"""Which attention kernel the HF backend should pin: cuDNN on grouped heads, or memory efficient on expanded ones.

cuDNN builds an execution plan for every new sequence length, and served traffic rarely
repeats a length. Both arms run on one loaded model, switched between measurements.

    uv run python tests-AB/ab_attention_kernel.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np
import torch
from ab_env import INPUTS, split_by_label
from ab_temperature import collect_with
from ab_temperature import measure as measure_banking
from benchmark import make_request, make_state
from PIL import Image
from torch.nn.attention import SDPBackend

from logit_classifier.backends.hf import ATTENTION_IMPLEMENTATION, PREFERRED_ATTENTION
from logit_classifier.classifier import Classifier
from logit_classifier.config import Config, fitted_temperature
from logit_classifier.schema import parse_request

ARMS = {
    "cudnn, grouped heads (0.4.0)": ("sdpa", (SDPBackend.CUDNN_ATTENTION,)),
    "efficient, expanded heads": (ATTENTION_IMPLEMENTATION, PREFERRED_ATTENTION),
}
# Distinct state lengths, so every request is one the kernel has not seen.
NEW_LENGTHS = [96 + 71 * step for step in range(40)]
LONG_STATE_TOKENS = 8192
IMAGE = INPUTS / "top-L.png"


def use_arm(classifier: Classifier, arm: str) -> None:
    implementation, kernels = ARMS[arm]
    classifier.backend.model.set_attn_implementation(implementation)
    classifier.backend.attention_backends = kernels


def percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * share
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def timed_ms(classifier: Classifier, request, image=None) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    classifier.classify(request, image=image)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000


def spread(values: list[float]) -> str:
    return (f"p50 {percentile(values, .5):7.1f}  p95 {percentile(values, .95):7.1f}  "
            f"max {max(values):7.1f} ms")


def new_length_latency(classifier: Classifier) -> None:
    print(f"A. One 4 option question per request, {len(NEW_LENGTHS)} state lengths "
          f"from {NEW_LENGTHS[0]} to {NEW_LENGTHS[-1]} tokens")
    requests = [make_request(make_state(classifier.backend, n), 1) for n in NEW_LENGTHS]
    for arm in ARMS:
        use_arm(classifier, arm)
        first = [timed_ms(classifier, request) for request in requests]
        repeat = [timed_ms(classifier, request) for request in requests]
        print(f"  {arm:30}  first seen  {spread(first)}")
        print(f"  {'':30}  repeated    {spread(repeat)}")


def long_state(classifier: Classifier) -> None:
    print(f"\nB. {LONG_STATE_TOKENS} token state, 8 questions")
    request = make_request(make_state(classifier.backend, LONG_STATE_TOKENS), 8)
    for arm in ARMS:
        use_arm(classifier, arm)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        first = timed_ms(classifier, request)
        warm = statistics.median(timed_ms(classifier, request) for _ in range(3))
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  {arm:30}  first seen {first:7.0f} ms   warm {warm:7.0f} ms   peak {peak:5.2f} GB")


def banking(classifier: Classifier) -> None:
    temperature = fitted_temperature(classifier.backend.canonical_model_id)
    print(f"\nC. Banking77, 924 row sample, held out half at T={temperature}")
    collected = {}
    for arm in ARMS:
        use_arm(classifier, arm)
        collected[arm] = collect_with(classifier, 12, 0)
        _, test = split_by_label(collected[arm]["records"])
        held = measure_banking(test, collected[arm]["layout"], temperature)
        print(f"  {arm:30}  acc {held['acc']:.4f}   ECE {held['ece']:.4f}   NLL {held['nll']:.4f}")

    old, new = (collected[arm]["records"] for arm in ARMS)
    shifts = [float(np.max(np.abs(np.array(a, dtype=float) - np.array(b, dtype=float))))
              for x, y in zip(old, new, strict=True)
              for a, b in zip(x["logits"], y["logits"], strict=True)]
    flips = sum(np.argmax(np.concatenate(x["logits"])) != np.argmax(np.concatenate(y["logits"]))
                for x, y in zip(old, new, strict=True))
    print(f"  logit shift between arms: max {max(shifts):.4f}, median {statistics.median(shifts):.4f}; "
          f"top letter changed on {flips} of {len(old)} rows")

    arm = list(ARMS)[1]
    use_arm(classifier, arm)
    again = collect_with(classifier, 12, 0)["records"]
    identical = sum(x["logits"] == y["logits"] for x, y in zip(collected[arm]["records"], again, strict=True))
    print(f"\nD. Same request, same logits, {arm}: {identical} of {len(again)} rows bitwise equal")


def image(classifier: Classifier) -> None:
    print(f"\nE. Image state, {Path(IMAGE).name}, through the vision tower")
    picture = Image.open(IMAGE).convert("RGB")
    request = parse_request({"state": "", "questions": {"subject": {
        "type": "choice", "instructions": "What is the main subject of this image",
        "criteria": dict.fromkeys(["a person", "an animal", "a building", "a landscape", "an object"])}}})
    answers = {}
    for arm in ARMS:
        use_arm(classifier, arm)
        first = timed_ms(classifier, request, picture)
        response, _ = classifier.classify(request, image=picture)
        answers[arm] = response.answers["subject"].probabilities
        print(f"  {arm:30}  first seen {first:6.0f} ms   {answers[arm]}")
    old, new = answers.values()
    print(f"  max probability shift {max(abs(old[k] - new[k]) for k in old):.6f}")


def main() -> None:
    classifier = Classifier(Config(use_prior_debias=False, permutations=1))
    print(f"model {classifier.config.model_id}, device {torch.cuda.get_device_name(0)}\n")
    new_length_latency(classifier)
    long_state(classifier)
    image(classifier)
    banking(classifier)


if __name__ == "__main__":
    main()
