"""Local zero-shot classifier that reads its answer off a logit row.

One state and a set of declared questions go in. One calibrated probability per
declared option comes out. No token is generated.

Importing this package pulls in numpy alone. The transformers backend arrives
with the `[hf]` extra and the HTTP service with `[service]`.
"""

from __future__ import annotations

from ._version import __version__
from .backends.base import (
    Backend,
    BackendContractError,
    BranchLogits,
    UnsupportedModelError,
    VisionUnsupportedError,
    verify_backend,
)
from .classifier import Classifier, Diagnostics, load_model
from .config import ANSWER_PREFILL, Config
from .deps import MissingDependencyError
from .errors import ConfigError, LogitClassifierError
from .labels import (
    LABEL_ALPHABET,
    MAX_LABELS_PER_BRANCH,
    LabelBoundaryError,
    verify_label_ids,
)
from .schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
    Question,
    SchemaError,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    parse_questions,
    parse_request,
)
from .vision import ImageError

# The ComfyUI socket type a node pack declares for a loaded classifier. It lives
# here so the library and the packs cannot drift apart on the spelling.
COMFY_SOCKET_TYPE = "LOGIT_CLASSIFIER"

__all__ = [
    "ANSWER_PREFILL",
    "COMFY_SOCKET_TYPE",
    "LABEL_ALPHABET",
    "MAX_LABELS_PER_BRANCH",
    "Answer",
    "Backend",
    "BackendContractError",
    "BranchLogits",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "Classifier",
    "Config",
    "ConfigError",
    "Diagnostics",
    "ImageError",
    "LabelBoundaryError",
    "LogitClassifierError",
    "MissingDependencyError",
    "NoulAnswer",
    "NoulCriteria",
    "NoulQuestion",
    "Question",
    "SchemaError",
    "ScoreAnswer",
    "ScoreQuestion",
    "SystemOneRequest",
    "SystemOneResponse",
    "UnsupportedModelError",
    "Usage",
    "VisionUnsupportedError",
    "__version__",
    "load_model",
    "parse_questions",
    "parse_request",
    "verify_backend",
    "verify_label_ids",
]
