"""Request and response models mirroring the TypeSafe System One wire format.

Field names, nesting and answer shapes follow the published Jev contract so a
Jev request body is accepted unchanged. Deviations are noted where they occur.

Plain dataclasses rather than pydantic, so a ComfyUI node pack importing this
package never pulls a pydantic version into ComfyUI's own venv.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .config import JEV_MAX_OPTIONS, JEV_MAX_QUESTIONS, JEV_MAX_SCORE_LEVELS
from .errors import LogitClassifierError

# Jev accepts a string, object or array anywhere prose is expected.
JSONContent = str | dict[str, Any] | list[Any]

DEFAULT_MODEL = "logit-latest"

_REQUEST_FIELDS = frozenset({"state", "model", "questions"})
_QUESTION_FIELDS = frozenset({"type", "instructions", "criteria"})
_NOUL_CRITERIA_FIELDS = frozenset({"true", "false"})
_QUESTION_TYPES = ("choice", "score", "noul")
# Keeps render_content's recursive json.dumps(indent=2) far below the interpreter recursion limit.
MAX_CONTENT_DEPTH = 64


class SchemaError(LogitClassifierError, ValueError):
    """A body does not match the System One contract.

    `field` carries the dotted path to the offending value, which is what the
    service reports back so a caller can find it without guessing.
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.field = field or None


def render_content(value: JSONContent | None) -> str:
    """Flatten a JSONContent value into the text the model sees."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True, kw_only=True)
class NoulCriteria:
    """Wording for the two sides of a noul. `true` and `false` are reserved words."""

    true_: JSONContent | None = None
    false_: JSONContent | None = None


@dataclass(frozen=True, kw_only=True)
class ChoiceQuestion:
    """Pick one of the named options."""

    type: Literal["choice"] = "choice"
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None]


@dataclass(frozen=True, kw_only=True)
class ScoreQuestion:
    """Place the state on an ordered scale, lowest level first."""

    type: Literal["score"] = "score"
    instructions: JSONContent | None = None
    criteria: list[JSONContent]


@dataclass(frozen=True, kw_only=True)
class NoulQuestion:
    """How true one statement is of the state."""

    type: Literal["noul"] = "noul"
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


Question = ChoiceQuestion | ScoreQuestion | NoulQuestion


@dataclass(frozen=True, kw_only=True)
class SystemOneRequest:
    """One state, and every question asked about it."""

    state: JSONContent
    model: str = DEFAULT_MODEL
    questions: dict[str, Question]


# Answer field order matches the published response examples, and asdict follows
# the order the fields are declared in.
@dataclass(frozen=True, kw_only=True)
class ChoiceAnswer:
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]
    # How strongly the model wants none of the offered options. It sits outside
    # probabilities, which still covers the caller's own options and sums to 1.
    abstain: float | None = None


@dataclass(frozen=True, kw_only=True)
class ScoreAnswer:
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, JSONContent]
    probabilities: dict[str, float]


@dataclass(frozen=True, kw_only=True)
class NoulAnswer:
    type: Literal["noul"] = "noul"
    noul: float


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


@dataclass(frozen=True, kw_only=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, kw_only=True)
class SystemOneResponse:
    model: str
    answers: dict[str, Answer]
    usage: Usage

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class ErrorBody:
    """The error shape this service returns.

    Jev documents its status codes but publishes no error body, so this shape is ours.
    """

    error: str
    message: str
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _path(*parts: str) -> str:
    return ".".join(part for part in parts if part)


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"expected an object, got {type(value).__name__}", where)
    for key in value:
        if not isinstance(key, str):
            raise SchemaError(f"object keys must be strings, got {type(key).__name__}", where)
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise SchemaError(f"unexpected field {unknown[0]!r}", _path(where, unknown[0]))


def _content(value: Any, where: str) -> JSONContent:
    pending: list[tuple[Any, int]] = [(value, 1)]

    if isinstance(value, str):
        return value
    if not isinstance(value, dict | list):
        raise SchemaError(f"expected a string, object or array, got {type(value).__name__}", where)
    while pending:
        node, depth = pending.pop()
        if depth > MAX_CONTENT_DEPTH:
            raise SchemaError(f"content nests deeper than {MAX_CONTENT_DEPTH} levels", where)
        children = node.values() if isinstance(node, dict) else node
        pending.extend((child, depth + 1) for child in children if isinstance(child, dict | list))
    return value


def _optional_content(value: Any, where: str) -> JSONContent | None:
    return None if value is None else _content(value, where)


def _choice_criteria(value: Any, where: str) -> dict[str, JSONContent | None]:
    mapping = _require_mapping(value, where)

    if not 2 <= len(mapping) <= JEV_MAX_OPTIONS:
        raise SchemaError(
            f"a choice needs 2 to {JEV_MAX_OPTIONS} options, got {len(mapping)}", where
        )
    return {name: _optional_content(body, _path(where, name)) for name, body in mapping.items()}


def _score_criteria(value: Any, where: str) -> list[JSONContent]:
    if not isinstance(value, list):
        raise SchemaError(f"expected an array, got {type(value).__name__}", where)
    if not 2 <= len(value) <= JEV_MAX_SCORE_LEVELS:
        raise SchemaError(
            f"a score needs 2 to {JEV_MAX_SCORE_LEVELS} levels, got {len(value)}", where
        )
    return [_content(level, _path(where, str(index))) for index, level in enumerate(value)]


def _noul_criteria(value: Any, where: str) -> NoulCriteria:
    mapping = _require_mapping(value, where)

    _reject_unknown(mapping, _NOUL_CRITERIA_FIELDS, where)
    return NoulCriteria(
        true_=_optional_content(mapping.get("true"), _path(where, "true")),
        false_=_optional_content(mapping.get("false"), _path(where, "false")),
    )


def _parse_question(value: Any, where: str) -> Question:
    mapping = _require_mapping(value, where)
    kind = mapping.get("type")
    instructions = _optional_content(mapping.get("instructions"), _path(where, "instructions"))
    criteria = mapping.get("criteria")
    criteria_at = _path(where, "criteria")

    if kind not in _QUESTION_TYPES:
        raise SchemaError(
            f"type must be one of {', '.join(_QUESTION_TYPES)}, got {kind!r}",
            _path(where, "type"),
        )
    _reject_unknown(mapping, _QUESTION_FIELDS, where)

    if kind == "choice":
        if criteria is None:
            raise SchemaError("a choice needs criteria", criteria_at)
        return ChoiceQuestion(
            instructions=instructions, criteria=_choice_criteria(criteria, criteria_at)
        )
    if kind == "score":
        if criteria is None:
            raise SchemaError("a score needs criteria", criteria_at)
        return ScoreQuestion(
            instructions=instructions, criteria=_score_criteria(criteria, criteria_at)
        )
    return NoulQuestion(
        instructions=instructions,
        criteria=None if criteria is None else _noul_criteria(criteria, criteria_at),
    )


def parse_questions(value: Any) -> dict[str, Question]:
    """Validate a question map on its own, for a caller holding no full request."""
    mapping = _require_mapping(value, "questions")

    if not 1 <= len(mapping) <= JEV_MAX_QUESTIONS:
        raise SchemaError(
            f"1 to {JEV_MAX_QUESTIONS} questions are allowed, got {len(mapping)}", "questions"
        )
    return {qid: _parse_question(body, _path("questions", qid)) for qid, body in mapping.items()}


def parse_request(body: Any) -> SystemOneRequest:
    """Validate a decoded JSON body and return the request it describes.

    Every failure raises SchemaError carrying the dotted path to the value that
    failed, so the first rejection names its own field rather than the body.
    """
    mapping = _require_mapping(body, "")

    _reject_unknown(mapping, _REQUEST_FIELDS, "")
    if "state" not in mapping:
        raise SchemaError("state is required", "state")
    if "questions" not in mapping:
        raise SchemaError("questions is required", "questions")

    model = mapping.get("model", DEFAULT_MODEL)
    if not isinstance(model, str):
        raise SchemaError(f"expected a string, got {type(model).__name__}", "model")

    return SystemOneRequest(
        state=_content(mapping["state"], "state"),
        model=model,
        questions=parse_questions(mapping["questions"]),
    )
