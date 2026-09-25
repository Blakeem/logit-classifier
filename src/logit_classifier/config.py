"""Runtime configuration for the classifier service."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ._version import __version__
from .errors import ConfigError

# Jev caps a Choice at 255 options and a Score at 10 levels. Matching those caps
# keeps a Jev-shaped request valid here without translation.
JEV_MAX_OPTIONS = 255
JEV_MAX_SCORE_LEVELS = 10
JEV_MAX_QUESTIONS = 256

# The tokenizer merges a letter into a preceding space, so the prefill must end
# on a non-space. Verified against Qwen: "Answer: (" ends on the token "Ġ(" and
# "Ġ(A" is absent from the vocabulary, leaving each letter its own token.
ANSWER_PREFILL = "Answer: ("

# Temperature belongs to the model, not to the method. Each of these was fitted on the
# held out half of a 924 row Banking77 sample and chosen on calibration error. Accuracy
# also climbs with temperature above 52 options, but that is the split path artefact
# rather than a gain, so the fit does not follow it.
FITTED_TEMPERATURES = {
    "Qwen/Qwen3-VL-4B-Instruct": 1.25,
    "Qwen/Qwen3-4B-Instruct-2507": 6.0,
}

# What an unfitted model gets. Published fitted temperatures for this readout cluster
# near 1 in domain and near 3 out of domain, so the midpoint is the least wrong guess.
DEFAULT_TEMPERATURE = 2.5


def fitted_temperature(model_id: str) -> float:
    return FITTED_TEMPERATURES.get(model_id, DEFAULT_TEMPERATURE)


def canonical_model_id(model_id: str) -> str:
    """Map a local directory holding shipped weights onto its Hugging Face repo id.

    A local path would otherwise miss FITTED_TEMPERATURES and fall back to the default.
    """
    if model_id in FITTED_TEMPERATURES:
        return model_id
    folder = model_id.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    for known in FITTED_TEMPERATURES:
        if known.rsplit("/", 1)[-1] == folder:
            return known
    return model_id


# Offering an escape label costs nothing and scores how badly the model wants to decline.
ABSTAIN = True

# The two branch layouts _score_branches implements. Any other spelling would reach
# the independent path without a word.
SCORE_METHODS = frozenset({"joint", "independent"})

# Below this many observations the running prior is noise, so it stays unused.
PRIOR_MIN_OBSERVATIONS = 32

# A label the model never picks drives its mean toward zero, and subtracting the log of a
# near-zero prior would hand that label an unbounded boost. Blending toward uniform bounds
# the correction. Measured on Banking77, an unsmoothed prior reached a 40 million to one
# spread across 40 labels.
PRIOR_SMOOTHING = 0.1


def _reject_score_method(where: str, value: str) -> None:
    if value not in SCORE_METHODS:
        raise ConfigError(
            f"{where}={value!r} is not a score method. "
            f"Accepted values are {', '.join(sorted(SCORE_METHODS))}."
        )


def _reject_temperature(where: str, value: float | None) -> None:
    if value is not None and not (math.isfinite(value) and value > 0):
        raise ConfigError(
            f"{where}={value!r} is not a temperature. "
            "Accepted values are None or a finite number above 0."
        )


def _reject_below_one(where: str, value: int) -> None:
    if value < 1:
        raise ConfigError(
            f"{where}={value!r} is below 1. Accepted values are whole numbers of 1 or more."
        )


def _env_switch(name: str, default: bool) -> bool:
    raw = os.environ.get(name)

    if raw is None:
        return default
    if raw not in ("0", "1"):
        raise ConfigError(f"{name}={raw!r} is not a switch. Accepted values are 0 and 1.")
    return raw == "1"


def _env_number[N: (int, float)](name: str, parse: Callable[[str], N]) -> N | None:
    raw = os.environ.get(name)

    if not raw:
        return None
    try:
        return parse(raw)
    except ValueError as error:
        raise ConfigError(
            f"{name}={raw!r} is not a number. Accepted values are {parse.__name__} literals."
        ) from error


@dataclass(frozen=True)
class Config:
    # Reads images as well as text, and scored 0.626 against 0.554 for the text only
    # Qwen3-4B-Instruct-2507 on the same held out Banking77 rows at the same latency.
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"
    # None leaves the download location to HF_HOME, so installing this package puts
    # nothing in whatever directory the caller happened to start in. A path keeps the
    # weights beside the project instead, which is what the examples do.
    models_dir: Path | None = None

    served_model_id: str = f"logit-classifier-{__version__}"
    model_aliases: tuple[str, ...] = ("logit-latest", "logit-preview", "jev-latest", "jev-preview")

    # None asks for the backend model's fitted value, resolved by Classifier.
    temperature: float | None = None
    use_prior_debias: bool = True
    # None keeps the running prior in memory for the life of the process, so importing
    # this library writes no file the host never asked for. from_env turns it back on.
    calibration_path: Path | None = None

    # "joint" scores all levels in one branch, "independent" judges each alone.
    score_method: str = "joint"
    # Relettering varies which group each option lands in once a question splits above 52 options.
    # On Banking77's 77 options, four letterings moved accuracy from 0.554 to 0.693. At 10 options
    # in one branch, accuracy did not move. Off by default because it multiplies the branch count.
    permutations: int = 1
    # An escape label absorbs the mass the model would otherwise spread over wrong
    # options, so offering one raised accuracy from 0.881 to 0.887 as well as scoring
    # how badly the model wants to decline.
    abstain: bool = ABSTAIN

    # False scores each branch in its own forward pass, which makes a question's
    # logits independent of the sibling questions batched with it.
    batch_branches: bool = True
    max_batch_rows: int = 32

    def __post_init__(self) -> None:
        # A ComfyUI pack builds Config directly, so from_env cannot be the only check.
        _reject_score_method("Config.score_method", self.score_method)
        _reject_temperature("Config.temperature", self.temperature)
        _reject_below_one("Config.permutations", self.permutations)
        _reject_below_one("Config.max_batch_rows", self.max_batch_rows)

    @classmethod
    def from_env(cls) -> Config:
        models_dir = os.environ.get("LOGIT_MODELS_DIR")
        score_method = os.environ.get("LOGIT_SCORE_METHOD", cls.score_method)
        permutations = _env_number("LOGIT_PERMUTATIONS", int)
        temperature = _env_number("LOGIT_TEMPERATURE", float)

        # Checked here as well as in __post_init__ so the message names the variable set.
        _reject_score_method("LOGIT_SCORE_METHOD", score_method)
        _reject_temperature("LOGIT_TEMPERATURE", temperature)
        if permutations is not None:
            _reject_below_one("LOGIT_PERMUTATIONS", permutations)
        return cls(
            model_id=os.environ.get("LOGIT_MODEL_ID", cls.model_id),
            models_dir=Path(models_dir) if models_dir else None,
            device=os.environ.get("LOGIT_DEVICE", cls.device),
            temperature=temperature,
            use_prior_debias=_env_switch("LOGIT_PRIOR_DEBIAS", cls.use_prior_debias),
            batch_branches=_env_switch("LOGIT_BATCH_BRANCHES", cls.batch_branches),
            score_method=score_method,
            permutations=cls.permutations if permutations is None else permutations,
            abstain=_env_switch("LOGIT_ABSTAIN", cls.abstain),
            calibration_path=Path(os.environ.get("LOGIT_CALIBRATION_PATH", "calibration.json")),
        )
