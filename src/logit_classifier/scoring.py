"""Turning label logits into calibrated probabilities."""

from __future__ import annotations

import numpy as np


def restricted_softmax(
    z: np.ndarray, log_prior: np.ndarray | None = None, temperature: float = 1.0
) -> np.ndarray:
    """Normalise over the option labels only, in float64.

    Subtracting a log-prior is the shared operation behind batch calibration,
    contextual calibration and PriDe. Temperature then widens the distribution,
    which a restricted softmax over an aligned model is otherwise too sharp for.
    """
    adjusted = z.astype(np.float64)

    if log_prior is not None:
        adjusted = adjusted - log_prior
    if temperature and temperature != 1.0:
        adjusted = adjusted / temperature
    adjusted = adjusted - adjusted.max()
    weights: np.ndarray = np.exp(adjusted)
    normalised: np.ndarray = weights / weights.sum()
    return normalised


def choice_confidence(probabilities: np.ndarray) -> float:
    """Peak probability rescaled so uniform maps to 0 and a point mass to 1.

    Ported with score_confidence from TypeSafe's own
    system_one_adapter._utils.confidence_metrics, so a System One client reading
    our answers computes the same number it would from theirs.
    """
    count = len(probabilities)

    if count < 2:
        return 1.0
    peak = float(probabilities.max())
    return float(np.clip((count * peak - 1.0) / (count - 1.0), 0.0, 1.0))


def score_confidence(probabilities: np.ndarray) -> float:
    """Concentration of an ordered distribution around its modal level.

    Levels are ordinal, so mass one level from the mode counts less than mass
    four levels away. The choice formula cannot express that, and it reports a
    tight two level split as low confidence.
    """
    count = len(probabilities)

    if count < 2:
        return 1.0
    indices = np.arange(count, dtype=np.float64)
    spread = float(probabilities @ np.abs(indices - int(np.argmax(probabilities))))
    uniform_spread = float(np.abs(indices - (count - 1) / 2.0).mean())
    return float(max(0.0, 1.0 - spread / uniform_spread))


def normalise_levels(yes_probabilities: np.ndarray) -> np.ndarray:
    """Turn independent per-level yes probabilities into one distribution."""
    total = float(yes_probabilities.sum())

    if total <= 0.0:
        return np.full(len(yes_probabilities), 1.0 / len(yes_probabilities))
    return yes_probabilities / total


def expected_score(level_probabilities: np.ndarray) -> float:
    """Probability-weighted mean of the zero-based level indices."""
    return float(level_probabilities @ np.arange(len(level_probabilities), dtype=np.float64))


def combine_escape(group_probabilities: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Merge groups whose last label is "none of these" into one distribution.

    A group's weight is the mass it did not put on its escape label, so the
    groups compete without ever being summarised for the model. That removes the
    separate group branch, and with it the label bias that branch carried.

    The second return is the abstain score, which is how hard the least declining
    group declined. Every group has to decline for the question to be unanswerable,
    so the weakest decline is the binding one. It is reported beside the distribution
    rather than inside it.
    """
    weights = np.array([1.0 - float(p[-1]) for p in group_probabilities], dtype=np.float64)
    parts: list[np.ndarray] = []

    if weights.sum() <= 0:
        weights = np.ones(len(group_probabilities), dtype=np.float64)
    weights = weights / weights.sum()

    for weight, probabilities in zip(weights, group_probabilities, strict=True):
        members = probabilities[:-1]
        total = members.sum()
        share = members / total if total > 0 else np.full(len(members), 1.0 / len(members))
        parts.append(weight * share)
    combined = np.concatenate(parts)
    total = combined.sum()
    declined = float(min(float(p[-1]) for p in group_probabilities))
    return (combined / total if total > 0 else combined), declined
