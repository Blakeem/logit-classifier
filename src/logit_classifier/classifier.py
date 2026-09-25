"""Pipeline coordinator: request in, Jev-shaped answers out."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .backends.base import Backend, BackendContractError, BranchLogits
from .calibrate import PriorStore
from .config import ANSWER_PREFILL, Config, fitted_temperature
from .deps import MissingDependencyError
from .prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    Branch,
    branch_content,
    build_branches,
    prefix_content,
)
from .schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    SchemaError,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)
from .scoring import (
    choice_confidence,
    combine_escape,
    expected_score,
    normalise_levels,
    restricted_softmax,
    score_confidence,
)
from .vision import IMAGE_ONLY_STATE, extract_image, image_key

# The label prior is a property of the token and how many labels compete, so
# branches sharing a shape share a bucket. A score's letter names a fixed rung, so a
# score never shares a bucket with a choice of the same width.
_PRIOR_KIND = {
    "choice": "choice",
    "member": "choice",
    "score_level": "binary",
    "score_joint": "score",
    "noul": "binary",
}


@dataclass
class Diagnostics:
    """Per-branch signals that Jev does not return, surfaced on request."""

    candidate_mass: dict[str, list[float]] = field(default_factory=dict)
    prior_applied: dict[str, bool] = field(default_factory=dict)
    branch_counts: dict[str, int] = field(default_factory=dict)


def load_model(model_id: str | None = None, config: Config | None = None) -> Backend:
    """Load a model through transformers and return it as a Backend.

    `model_id` names a Hugging Face repo or a local directory, and overrides the one on
    `config`. Where the weights land is `config.models_dir`. A ComfyUI node skips this
    entirely and passes its own Backend over a model the workflow already loaded.

    The import sits inside the call so the core install needs no torch.
    """
    config = config if config is not None else Config()
    if model_id is not None:
        config = replace(config, model_id=model_id)

    try:
        from .backends.hf import HFBackend
    except ImportError as error:
        raise MissingDependencyError(
            "no backend was passed and the local transformers backend is not installed. "
            'Install it with: pip install "logit-classifier[hf]"'
        ) from error
    return HFBackend(config)


def _beside_host_image(state: object) -> object:
    """Return the state a host-supplied image is rendered beside.

    An empty state renders as extract_image renders a state that held only an image, so
    a host and the service build the same prompt for the same content.
    """
    carried = image_key(state)

    if carried is not None:
        raise SchemaError(
            f"state carries an image under {carried!r} while image= was also given, "
            f"so the request holds two images and one prefix reads one",
            field=f"state.{carried}",
        )
    if state in ("", {}):
        return IMAGE_ONLY_STATE
    return state


def _model_identity(backend: Backend) -> str:
    """Return the id the fitted temperature and the prior fingerprint key on."""
    return getattr(backend, "canonical_model_id", None) or backend.model_id


class Classifier:
    """Turns one System One request into one calibrated answer per question.

    A caller that already holds a loaded model, such as a ComfyUI node over a
    resident text encoder, passes its own backend and no weights are loaded here.

    One instance belongs to one backend. The fitted temperature and the prior bucket
    are both derived from the backend's model at construction, so swapping the
    backend afterwards leaves both belonging to the previous model.
    """

    def __init__(self, config: Config, backend: Backend | None = None) -> None:
        self.config = config
        self.backend = backend if backend is not None else load_model(config=config)
        # Keyed on the model the backend actually loaded, not on a Config field a
        # caller passing its own backend never had reason to set.
        self.temperature = (
            config.temperature if config.temperature is not None
            else fitted_temperature(_model_identity(self.backend))
        )
        self.priors = PriorStore(config.calibration_path, self._fingerprint())

    def _fingerprint(self) -> str:
        """Identify the prompt shape a stored prior was measured under."""
        parts = [_model_identity(self.backend), ANSWER_PREFILL, SYSTEM_PROMPT, self.config.score_method,
                 str(PROMPT_VERSION), str(self.config.abstain)]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    def persist_prior(self) -> None:
        """Write the running prior, unless the caller turned the prior off.

        The service calls this again at shutdown, so the gate lives here rather than
        at each call site.
        """
        if self.config.use_prior_debias:
            self.priors.save()

    def _calibrated(self, branch: Branch, logits: BranchLogits) -> tuple[np.ndarray, bool]:
        """Raw distribution feeds the prior estimate, calibrated one is returned."""
        kind = _PRIOR_KIND[branch.kind]
        # The escape label means "none of these" wherever it sits, so its mass is content
        # rather than letter bias, and only the lettered options get a prior.
        lettered = branch.label_count - 1 if branch.kind == "member" else branch.label_count
        letters_prior: np.ndarray | None = None
        log_prior: np.ndarray | None = None

        # One switch turns learning, application and persistence off together, so a host
        # that asked for no debias gets the same numbers on every call.
        if self.config.use_prior_debias:
            self.priors.observe(kind, restricted_softmax(logits.z[:lettered]))
            letters_prior = self.priors.log_prior(kind, lettered)
        if letters_prior is not None:
            log_prior = np.zeros(branch.label_count, dtype=np.float64)
            log_prior[:lettered] = letters_prior
        calibrated = restricted_softmax(logits.z, log_prior, self.temperature)
        return calibrated, log_prior is not None

    def _letterings(self, questions: dict[str, Question]) -> list[dict[str, Question]]:
        """One question map per lettering, the first in the order the caller gave.

        Only choice options are reordered. A score lists its levels lowest first, so
        that order carries meaning, and a noul has two fixed sides. Each lettering
        draws from a fixed seed, so the same request still answers the same way.
        """
        rounds = [questions]

        for seed in range(1, max(1, self.config.permutations)):
            reordered: dict[str, Question] = {}
            for qid, question in questions.items():
                if not isinstance(question, ChoiceQuestion):
                    continue
                names = list(question.criteria)
                random.Random(seed).shuffle(names)
                criteria = {name: question.criteria[name] for name in names}
                reordered[qid] = replace(question, criteria=criteria)
            if reordered:
                rounds.append(reordered)
        return rounds

    def _suffix_ids(self, state: object, branches: list[Branch], has_image: bool,
                    prefix_text: str) -> list[list[int]]:
        """Each branch's tokens after the shared prefix, which it must begin with."""
        ids: list[list[int]] = []

        for branch in branches:
            rendered = self.backend.render(
                SYSTEM_PROMPT, branch_content(state, branch, has_image), ANSWER_PREFILL
            )
            if not rendered.startswith(prefix_text):
                raise BackendContractError(
                    f"{type(self.backend).__name__}.render did not begin the closed render "
                    f"of question {branch.question_id!r} with the open-ended render of the "
                    "same state, so the shared prefix cannot be split off by offset"
                )
            ids.append(self.backend.encode(rendered[len(prefix_text):]))
        return ids

    def classify(
        self, request: SystemOneRequest, *, allow_image_paths: bool = True, image: Any = None
    ) -> tuple[SystemOneResponse, Diagnostics]:
        """Answer every question in the request.

        `image` is an image the host already decoded, which the state, being JSON, cannot
        carry. Its type is whatever the backend's encode_prefix accepts: a PIL image for
        HFBackend, a ComfyUI IMAGE tensor for ComfyClipBackend.
        """
        rounds = self._letterings(request.questions)
        diagnostics = Diagnostics()
        probabilities: list[np.ndarray] = []
        backend_name = type(self.backend).__name__
        drafts: list[dict[str, Answer]] = []
        start = 0

        if image is not None:
            state = _beside_host_image(request.state)
        else:
            state, image = extract_image(request.state, allow_paths=allow_image_paths)
        seen = image is not None
        prefix_text = self.backend.render(
            SYSTEM_PROMPT, prefix_content(state, seen), ANSWER_PREFILL, open_ended=True
        )
        prefix_ids, vision = self.backend.encode_prefix(prefix_text, image)
        round_branches = [build_branches(q, self.config.score_method, self.config.abstain)
                          for q in rounds]
        branches: list[Branch] = [b for lettering in round_branches for b in lettering]
        suffix_ids = self._suffix_ids(state, branches, seen, prefix_text)
        scored = self.backend.score(prefix_ids, suffix_ids,
                                    [b.label_count for b in branches], vision)

        if len(scored) != len(branches):
            raise BackendContractError(
                f"{backend_name}.score returned {len(scored)} rows for "
                f"{len(branches)} branches"
            )
        for branch, logits in zip(branches, scored, strict=True):
            if len(logits.z) != branch.label_count:
                raise BackendContractError(
                    f"{backend_name}.score returned a {len(logits.z)}-wide row for "
                    f"question {branch.question_id}, which needs {branch.label_count}"
                )
            calibrated, applied = self._calibrated(branch, logits)
            probabilities.append(calibrated)
            diagnostics.candidate_mass.setdefault(branch.question_id, []).append(
                round(logits.candidate_mass, 6)
            )
            diagnostics.prior_applied[branch.question_id] = applied
            diagnostics.branch_counts[branch.question_id] = (
                diagnostics.branch_counts.get(branch.question_id, 0) + 1
            )
        self.persist_prior()

        for questions, lettering in zip(rounds, round_branches, strict=True):
            window = slice(start, start + len(lettering))
            start += len(lettering)
            drafts.append(self._round_answers(questions, lettering, probabilities[window]))
        answers = self._merge(request.questions, drafts)

        usage = Usage(
            input_tokens=len(prefix_ids) + sum(len(s) for s in suffix_ids),
            output_tokens=len(branches),
        )
        response = SystemOneResponse(
            model=self.config.served_model_id, answers=answers, usage=usage
        )
        return response, diagnostics

    def _round_answers(self, questions: dict[str, Question], branches: list[Branch],
                       probabilities: list[np.ndarray]) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}

        for qid, question in questions.items():
            span = [i for i, b in enumerate(branches) if b.question_id == qid]
            answers[qid] = self._assemble(question, [branches[i] for i in span],
                                          [probabilities[i] for i in span])
        return answers

    def _merge(self, questions: dict[str, Question],
               drafts: list[dict[str, Answer]]) -> dict[str, Answer]:
        """Average each option's probability across the letterings that scored it."""
        answers: dict[str, Answer] = {}

        for qid in questions:
            parts = [draft[qid] for draft in drafts if qid in draft]
            if len(parts) == 1:
                answers[qid] = parts[0]
                continue
            answers[qid] = self._average_choice([p for p in parts if isinstance(p, ChoiceAnswer)])
        return answers

    def _average_choice(self, parts: list[ChoiceAnswer]) -> ChoiceAnswer:
        names = list(parts[0].probabilities)
        stacked = np.array([[part.probabilities[name] for name in names] for part in parts])
        mean = stacked.mean(axis=0)
        mean = mean / mean.sum()
        declined = [part.abstain for part in parts if part.abstain is not None]
        return ChoiceAnswer(
            choice=names[int(np.argmax(mean))],
            confidence=round(choice_confidence(mean), 6),
            probabilities={name: round(float(p), 6) for name, p in zip(names, mean, strict=True)},
            abstain=round(float(np.mean(declined)), 6) if declined else None,
        )

    def _assemble(self, question: Question, branches: list[Branch],
                  probabilities: list[np.ndarray]) -> Answer:
        if isinstance(question, ChoiceQuestion):
            return self._choice_answer(question, branches, probabilities)
        if isinstance(question, ScoreQuestion):
            return self._score_answer(question, branches, probabilities)
        if isinstance(question, NoulQuestion):
            return NoulAnswer(noul=round(float(probabilities[0][0]), 6))
        raise TypeError(f"unsupported question type {type(question)!r}")

    def _choice_answer(self, question: ChoiceQuestion, branches: list[Branch],
                       probabilities: list[np.ndarray]) -> ChoiceAnswer:
        names = list(question.criteria)
        declined: float | None = None

        if branches[0].kind == "choice":
            flat = probabilities[0]
        else:
            flat, declined = combine_escape(probabilities)
        mapping = {name: round(float(p), 6) for name, p in zip(names, flat, strict=True)}
        winner = names[int(np.argmax(flat))]
        return ChoiceAnswer(
            choice=winner,
            confidence=round(choice_confidence(flat), 6),
            probabilities=mapping,
            abstain=None if declined is None else round(declined, 6),
        )

    def _score_answer(self, question: ScoreQuestion, branches: list[Branch],
                      probabilities: list[np.ndarray]) -> ScoreAnswer:
        if branches[0].kind == "score_joint":
            levels = probabilities[0]
        else:
            # Each level was judged alone, so index 0 of each branch is its yes.
            levels = normalise_levels(np.array([float(p[0]) for p in probabilities]))
        return ScoreAnswer(
            score=round(expected_score(levels), 6),
            confidence=round(score_confidence(levels), 6),
            legend={str(i): level for i, level in enumerate(question.criteria)},
            probabilities={str(i): round(float(p), 6) for i, p in enumerate(levels)},
        )
