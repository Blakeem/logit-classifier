"""Batch calibration: a running per-label prior learned from served traffic.

The label prior is token bias, the model's standing preference for the token
"A" over "B". It is estimated as a running mean of the uncalibrated label
distribution and subtracted in log space before the softmax. No labelled data
and no extra forward passes are required.
"""

from __future__ import annotations

import contextlib
import json
import math
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np

from .config import PRIOR_MIN_OBSERVATIONS, PRIOR_SMOOTHING


def _stored_mean(key: str, vector: object) -> list[float] | None:
    """Return the mean of one `kind:width` bucket, or None when the stored value is unusable."""
    kind, separator, width = key.partition(":")

    if not kind or separator != ":" or not (width.isascii() and width.isdigit()):
        return None
    if not isinstance(vector, list) or len(vector) != int(width):
        return None
    if not all(isinstance(value, int | float) and not isinstance(value, bool) for value in vector):
        return None
    values = [float(value) for value in vector]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        return None
    return values


class PriorStore:
    def __init__(
        self, path: Path | None, fingerprint: str, min_observations: int = PRIOR_MIN_OBSERVATIONS
    ) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.min_observations = min_observations
        self._lock = threading.Lock()
        self._means: dict[str, list[float]] = {}
        self._counts: dict[str, int] = {}
        self._load()

    @staticmethod
    def _bucket(kind: str, label_count: int) -> str:
        return f"{kind}:{label_count}"

    def _load(self) -> None:
        means: dict[str, list[float]] = {}
        counts: dict[str, int] = {}

        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        # A prior measured under a different model or prompt describes a different
        # distribution, so a mismatch starts over rather than blending the two.
        if payload.get("fingerprint") != self.fingerprint:
            return
        stored_means = payload.get("means")
        stored_counts = payload.get("counts")
        if not isinstance(stored_means, dict) or not isinstance(stored_counts, dict):
            return

        # The path is user-visible state, so a hand-edited bucket is a reachable input.
        # One that does not match its own key starts over the way a mismatch does, rather
        # than reaching observe and raising on every later request.
        for key, vector in stored_means.items():
            mean = _stored_mean(key, vector) if isinstance(key, str) else None
            count = stored_counts.get(key)
            if mean is None or not isinstance(count, int) or isinstance(count, bool) or count < 0:
                continue
            means[key] = mean
            counts[key] = count

        self._means = means
        self._counts = counts

    def save(self) -> None:
        # A path with no final component, `.` or a drive root, has no staged sibling to
        # name, and with_name raises ValueError, which the OSError suppression misses.
        if self.path is None or not self.path.name:
            return
        payload = {"fingerprint": self.fingerprint, "means": self._means, "counts": self._counts}

        # Two processes can hold a store over one path, and a truncating write hands a
        # reader a partial file that loads as an empty prior. A rename is atomic.
        with self._lock:
            staged = self.path.with_name(f"{self.path.name}.{uuid4().hex}.tmp")
            try:
                staged.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                staged.replace(self.path)
            except OSError:
                # A rename that keeps failing would otherwise leave one staged file per
                # request, since the uuid name is never reused.
                with contextlib.suppress(OSError):
                    staged.unlink(missing_ok=True)

    def observe(self, kind: str, probabilities: np.ndarray) -> None:
        """Fold one uncalibrated distribution into the running mean."""
        bucket = self._bucket(kind, len(probabilities))

        with self._lock:
            count = self._counts.get(bucket, 0)
            mean = np.asarray(self._means.get(bucket, [0.0] * len(probabilities)), dtype=np.float64)
            self._means[bucket] = ((count * mean + probabilities) / (count + 1)).tolist()
            self._counts[bucket] = count + 1

    def log_prior(self, kind: str, label_count: int) -> np.ndarray | None:
        """Return the log-prior to subtract, or None while the estimate is thin."""
        bucket = self._bucket(kind, label_count)

        with self._lock:
            count = self._counts.get(bucket, 0)
            mean = self._means.get(bucket)
        if mean is None or count < self.min_observations or len(mean) != label_count:
            return None
        prior = np.asarray(mean, dtype=np.float64)
        if not np.all(prior >= 0) or prior.sum() <= 0:
            return None
        prior = prior / prior.sum()
        prior = (1.0 - PRIOR_SMOOTHING) * prior + PRIOR_SMOOTHING / label_count
        # Centring keeps the subtraction from shifting overall scale, which only
        # the temperature should control.
        log_prior = np.log(prior)
        return log_prior - log_prior.mean()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)
