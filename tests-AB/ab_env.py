"""Shared paths and measures for the A/B harnesses.

Every script here answers one question with a number. The pytest suite lives in tests/
and asserts behaviour instead.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CACHE = Path(__file__).resolve().parent / "cache"
INPUTS = Path(__file__).resolve().parent / "inputs"
BINS = 10

sys.path.insert(0, str(ROOT / "src"))


def banking77() -> dict:
    """The 3,080 row test split, with labels rewritten as readable option names."""
    data = json.loads((FIXTURES / "banking77_test.json").read_text(encoding="utf-8"))
    data["names"] = [name.replace("_", " ") for name in data["names"]]
    return data


def stratified(labels: list[int], per_label: int, seed: int) -> list[int]:
    """Row indices with an equal count of every label, so accuracy is not label weighted."""
    buckets: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        buckets.setdefault(label, []).append(index)
    chooser = random.Random(seed)
    chosen: list[int] = []

    for label in sorted(buckets):
        pool = buckets[label]
        chosen.extend(chooser.sample(pool, min(per_label, len(pool))))
    chosen.sort()
    return chosen


def split_by_label(records: list[dict], key: str = "gold") -> tuple[list[dict], list[dict]]:
    """Halve a stratified sample per label, so a fitted value is never tested on its own rows."""
    buckets: dict[str, list[dict]] = {}
    for record in records:
        buckets.setdefault(record[key], []).append(record)
    fit = [r for rows in buckets.values() for r in rows[: len(rows) // 2]]
    test = [r for rows in buckets.values() for r in rows[len(rows) // 2 :]]
    return fit, test


def expected_calibration_error(correct: list[bool], confidences: list[float]) -> float:
    total = len(correct)
    error = 0.0

    for edge in range(BINS):
        low, high = edge / BINS, (edge + 1) / BINS
        members = [i for i, c in enumerate(confidences) if low < c <= high or (edge == 0 and c == 0)]
        if not members:
            continue
        accuracy = sum(correct[i] for i in members) / len(members)
        mean_confidence = sum(confidences[i] for i in members) / len(members)
        error += (len(members) / total) * abs(accuracy - mean_confidence)
    return error


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a random positive outranks a random negative, ties counted as half."""
    if not len(positive) or not len(negative):
        return float("nan")
    values = np.concatenate([positive, negative])
    order = values.argsort()
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)

    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    count = len(positive)
    return (ranks[:count].sum() - count * (count + 1) / 2) / (count * len(negative))
