"""Run the Banking77 intent set, which has a published Jev and Laya score.

Banking77 has 77 labels, so every request goes through the split path. TypeSafe reports
0.870 for Jev on this set and 0.425 for Laya, which makes it the one directly comparable
number available.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ab_env import (
    FIXTURES,
    expected_calibration_error,
    stratified,
)

from logit_classifier.classifier import Classifier
from logit_classifier.config import Config
from logit_classifier.schema import parse_request

BINS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-label", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration", default="banking77_prior.json")
    parser.add_argument("--no-prior", action="store_true")
    parser.add_argument("--permutations", type=int, default=1)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    data = json.loads((FIXTURES / "banking77_test.json").read_text(encoding="utf-8"))
    names: list[str] = data["names"]
    indices = stratified(data["label"], args.per_label, args.seed)
    print(f"Banking77: {len(names)} labels, evaluating {len(indices)} of {len(data['text'])} test rows")

    config = Config(
        calibration_path=Path(args.calibration),
        use_prior_debias=not args.no_prior,
        permutations=args.permutations,
        **({"model_id": args.model} if args.model else {}),
    )
    classifier = Classifier(config)
    print(f"prior debias: {config.use_prior_debias}  temperature: {classifier.temperature}  "
          f"letterings: {config.permutations}")
    print(f"existing prior buckets: {classifier.priors.stats()}\n")

    criteria = {name.replace("_", " "): None for name in names}
    readable = [name.replace("_", " ") for name in names]

    correct: list[bool] = []
    confidences: list[float] = []
    latencies: list[float] = []
    masses: list[float] = []

    for position, index in enumerate(indices, 1):
        request = parse_request(
            {"state": data["text"][index],
             "questions": {"intent": {
                 "type": "choice",
                 "instructions": "Which banking support intent does this message express",
                 "criteria": criteria}}}
        )
        started = time.perf_counter()
        response, diagnostics = classifier.classify(request)
        latencies.append((time.perf_counter() - started) * 1000)
        answer = response.answers["intent"]
        correct.append(answer.choice == readable[data["label"][index]])
        confidences.append(answer.confidence)
        masses.append(min(diagnostics.candidate_mass["intent"]))
        if position % 100 == 0:
            print(f"  {position:>4}/{len(indices)}  running accuracy {sum(correct) / len(correct):.4f}")

    accuracy = sum(correct) / len(correct)
    print("\n" + "=" * 62)
    print(f"accuracy            {accuracy:.4f}   ({sum(correct)} of {len(correct)})")
    print(f"mean confidence     {statistics.mean(confidences):.4f}")
    print(f"ECE                 {expected_calibration_error(correct, confidences):.4f}")
    print(f"min label mass      {min(masses):.4f}   mean {statistics.mean(masses):.4f}")
    print(f"median latency      {statistics.median(latencies):.1f} ms")
    print("=" * 62)
    print("published Jev       0.870")
    print("published Laya      0.425")


if __name__ == "__main__":
    main()
