"""Measure accuracy and calibration on the labelled eval set.

Raw label logits are collected once per score method, then temperatures are
swept offline, so a sweep costs no extra forward passes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ab_env import FIXTURES

from logit_classifier.backends.hf import HFBackend
from logit_classifier.config import ABSTAIN, ANSWER_PREFILL, Config
from logit_classifier.prompt import SYSTEM_PROMPT, branch_content, build_branches, prefix_content
from logit_classifier.schema import parse_request
from logit_classifier.scoring import (
    choice_confidence,
    combine_escape,
    expected_score,
    normalise_levels,
    restricted_softmax,
    score_confidence,
)

BINS = 10


@dataclass
class Scored:
    item_id: str
    kind: str
    logits: list[np.ndarray]
    names: list[str]
    expect: object


def collect(backend: HFBackend, items: list[dict], score_method: str) -> list[Scored]:
    collected: list[Scored] = []

    for item in items:
        request = parse_request(
            {"state": item["state"], "questions": {"q": item["q"]}}
        )
        branches = build_branches(request.questions, score_method, ABSTAIN)
        prefix_text = backend.render(SYSTEM_PROMPT, prefix_content(request.state), ANSWER_PREFILL,
                                    open_ended=True)
        prefix_ids = backend.encode(prefix_text)
        suffixes = [
            backend.encode(backend.render(SYSTEM_PROMPT, branch_content(request.state, b),
                                        ANSWER_PREFILL)[len(prefix_text):])
            for b in branches
        ]
        scored = backend.score(prefix_ids, suffixes, [b.label_count for b in branches])
        question = request.questions["q"]
        names = list(question.criteria) if question.type == "choice" else []
        collected.append(
            Scored(item["id"], f"{question.type}:{branches[0].kind}",
                   [s.z for s in scored], names, item["expect"])
        )
    return collected


def predict(entry: Scored, temperature: float) -> tuple[bool, float, float | None]:
    """Return (correct, confidence, absolute score error)."""
    kind = entry.kind

    if kind.startswith("choice"):
        probabilities = restricted_softmax(entry.logits[0], temperature=temperature)
        if ABSTAIN:
            # The escape label sits last and is reported beside the distribution.
            probabilities = combine_escape([probabilities])[0]
        return entry.names[int(probabilities.argmax())] == entry.expect, choice_confidence(probabilities), None

    if kind == "score:score_joint":
        probabilities = restricted_softmax(entry.logits[0], temperature=temperature)
    elif kind == "score:score_level":
        yes = np.array([restricted_softmax(z, temperature=temperature)[0] for z in entry.logits])
        probabilities = normalise_levels(yes)
    else:
        probabilities = restricted_softmax(entry.logits[0], temperature=temperature)
        truth = bool(entry.expect)
        positive = float(probabilities[0])
        return (positive > 0.5) == truth, max(positive, 1.0 - positive), None

    index = int(probabilities.argmax())
    return index == entry.expect, score_confidence(probabilities), abs(expected_score(probabilities) - entry.expect)


def expected_calibration_error(correct: list[bool], confidences: list[float]) -> float:
    total = len(correct)
    error = 0.0

    for edge in range(BINS):
        low, high = edge / BINS, (edge + 1) / BINS
        members = [i for i, c in enumerate(confidences) if (low < c <= high or (edge == 0 and c == 0))]
        if not members:
            continue
        accuracy = sum(correct[i] for i in members) / len(members)
        mean_confidence = sum(confidences[i] for i in members) / len(members)
        error += (len(members) / total) * abs(accuracy - mean_confidence)
    return error


def report(collected: list[Scored], temperature: float) -> dict:
    groups: dict[str, list[tuple[bool, float, float | None]]] = {}

    for entry in collected:
        family = entry.kind.split(":")[0]
        groups.setdefault(family, []).append(predict(entry, temperature))
    summary = {}
    all_correct, all_confidence = [], []
    for family, rows in groups.items():
        correct = [r[0] for r in rows]
        confidences = [r[1] for r in rows]
        errors = [r[2] for r in rows if r[2] is not None]
        summary[family] = {
            "n": len(rows),
            "accuracy": sum(correct) / len(correct),
            "mean_confidence": sum(confidences) / len(confidences),
            "ece": expected_calibration_error(correct, confidences),
            "score_mae": (sum(errors) / len(errors)) if errors else None,
        }
        all_correct += correct
        all_confidence += confidences
    summary["overall"] = {
        "n": len(all_correct),
        "accuracy": sum(all_correct) / len(all_correct),
        "mean_confidence": sum(all_confidence) / len(all_confidence),
        "ece": expected_calibration_error(all_correct, all_confidence),
        "score_mae": None,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default=str(FIXTURES / "eval_set.json"))
    parser.add_argument("--temperatures", default="0.75,1.0,1.25,1.5,2.5,4.0,6.0")
    parser.add_argument("--methods", default="joint,independent")
    args = parser.parse_args()

    items = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    temperatures = [float(t) for t in args.temperatures.split(",")]
    backend = HFBackend(Config())

    for method in args.methods.split(","):
        collected = collect(backend, items, method)
        print(f"\n=== score_method={method} ===")
        header = f"{'T':>6} | {'family':10} | {'n':>3} | {'acc':>6} | {'meanconf':>8} | {'ECE':>6} | {'scoreMAE':>8}"
        print(header)
        print("-" * len(header))
        for temperature in temperatures:
            summary = report(collected, temperature)
            for family in ["choice", "score", "noul", "overall"]:
                if family not in summary:
                    continue
                row = summary[family]
                mae = f"{row['score_mae']:.3f}" if row["score_mae"] is not None else "     -"
                print(f"{temperature:>6.1f} | {family:10} | {row['n']:>3} | {row['accuracy']:>6.3f} | "
                      f"{row['mean_confidence']:>8.3f} | {row['ece']:>6.3f} | {mae:>8}")
            print()


if __name__ == "__main__":
    main()
