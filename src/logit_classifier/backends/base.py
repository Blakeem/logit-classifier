"""The readout port, the one boundary between the arithmetic and a host's forward pass.

Everything above this port is stdlib and numpy. Everything below it is one host's
way of turning token ids into a logit row: transformers here, a ComfyUI CLIP next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..config import ANSWER_PREFILL
from ..errors import LogitClassifierError
from ..labels import MAX_LABELS_PER_BRANCH, verify_label_ids

# The probe text only has to be stable. Routing the label proof through render
# rather than a hand-built template measured identical ids on both shipped models.
_PROBE_SYSTEM = "probe system"
_PROBE_STATE = "probe state"
# The closed render must carry a branch suffix, since _suffix_ids splits the real
# closed render on the open-ended one and a shared body would not exercise that.
_PROBE_SUFFIX = "\nQuestion: (A) probe option"


class VisionUnsupportedError(LogitClassifierError, ValueError):
    """An image was given to a model that carries no vision tower."""


class BackendContractError(LogitClassifierError, RuntimeError):
    """A backend returned rows the port does not allow."""


@dataclass(frozen=True)
class BranchLogits:
    """Raw label logits for one branch, before any calibration."""

    #: One logit per label this branch asked for, ordered as LABEL_ALPHABET[:count]
    #: and exactly that wide, never the full vocabulary row.
    z: np.ndarray
    # Share of full-vocabulary probability mass sitting on the option labels. A
    # low value means the restricted softmax is normalising noise.
    candidate_mass: float


@runtime_checkable
class Backend(Protocol):
    """What a host must provide for the classifier to read a distribution off it.

    Runtime checkable, so a node pack can assert its own duck-typed object
    satisfies the port before it reaches the classifier.

    A backend may also declare `canonical_model_id: str`, the identity the fitted
    temperature and the calibration fingerprint key on. Without it, `model_id` is that
    identity.
    """

    #: The key FITTED_TEMPERATURES is looked up by when no canonical_model_id is declared.
    #: Crossing the two shipped models took calibration error from 0.088 to 0.565, so a
    #: wrong string is silently miscalibrated.
    model_id: str
    #: One token id per label in LABEL_ALPHABET, proven at load against the rendered prefill.
    label_ids: list[int]
    #: Whether this host's model carries a vision tower.
    sees_images: bool

    def render(self, system: str, user: str, prefill: str, *, open_ended: bool = False) -> str:
        """Wrap the message bodies in the host's chat template and append the prefill.

        open_ended returns only the span up to the end of the user body, which is
        exactly what every branch of one request shares. The closed render of a state
        must begin with the open_ended render of that same state character for
        character, and encode must split at that same point, so that
        encode(prefix) + encode(suffix) equals encode(prefix + suffix) there.
        """
        ...

    def encode(self, text: str) -> list[int]:
        """Token ids for a fragment, with no special tokens added."""
        ...

    def encode_prefix(self, text: str, image: Any = None) -> tuple[list[int], dict[str, Any]]:
        """Prefix token ids, plus whatever tensors its forward pass needs for the image."""
        ...

    def score(
        self,
        prefix_ids: list[int],
        suffix_ids: list[list[int]],
        label_counts: list[int],
        vision: dict[str, Any] | None = None,
    ) -> list[BranchLogits]:
        """Read the label logits at each branch's final position, sharing one prefix.

        The returned list carries one entry per suffix, in the order given. Entry i's
        z is exactly label_counts[i] wide, ordered to match LABEL_ALPHABET[:count],
        since the caller maps those positions straight onto the branch's options.
        """
        ...


def verify_backend(backend: Backend) -> list[int]:
    """Prove a backend meets the port at load, and return its label ids.

    A backend that gets any of this wrong returns wrong probabilities rather than
    raising, so it is worth one call before a node goes live. The render check
    mirrors what _suffix_ids does per branch on every request.
    """
    if not isinstance(getattr(backend, "model_id", None), str):
        raise BackendContractError(
            f"{type(backend).__name__} declares no model_id, so the fitted temperature "
            f"cannot be looked up and the answer would be silently miscalibrated"
        )

    open_ended = backend.render(_PROBE_SYSTEM, _PROBE_STATE, ANSWER_PREFILL, open_ended=True)
    closed = backend.render(_PROBE_SYSTEM, _PROBE_STATE + _PROBE_SUFFIX, ANSWER_PREFILL)

    if not closed.startswith(open_ended):
        raise BackendContractError(
            f"{type(backend).__name__}.render did not begin its closed render with the "
            f"open-ended render of the same state, so every branch would be encoded at "
            f"the wrong offset"
        )
    return verify_label_ids(backend, closed, MAX_LABELS_PER_BRANCH)
