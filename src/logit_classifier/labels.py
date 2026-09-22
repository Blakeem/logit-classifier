"""Option label alphabet, boundary verification and the >52-option split."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Protocol

from .errors import LogitClassifierError

# Bare single letters are one token each in the Qwen vocabulary when the prefill
# ends on "Ġ(", which gives 52 usable labels. Verified per tokenizer at startup.
LABEL_ALPHABET = string.ascii_uppercase + string.ascii_lowercase
MAX_LABELS_PER_BRANCH = len(LABEL_ALPHABET)


class LabelBoundaryError(LogitClassifierError, RuntimeError):
    """A label does not occupy exactly one token after the prefill."""


class Encoder(Protocol):
    """The one method the boundary proof needs, so it binds to no host."""

    def encode(self, text: str) -> list[int]: ...


def verify_label_ids(tokenizer: Encoder, prefill_text: str, count: int) -> list[int]:
    """Return one token id per label, proving each adds exactly one token.

    The check must run against the rendered prefill, not the label alone. A
    prefill ending in a space silently merges the space into the letter, which
    an isolated encode of "A" cannot detect.
    """
    base_ids: list[int] = []
    label_ids: list[int] = []

    if count > MAX_LABELS_PER_BRANCH:
        raise LabelBoundaryError(f"{count} labels exceeds the {MAX_LABELS_PER_BRANCH} available")

    base_ids = tokenizer.encode(prefill_text)
    for label in LABEL_ALPHABET[:count]:
        full = tokenizer.encode(prefill_text + label)
        if len(full) != len(base_ids) + 1 or full[: len(base_ids)] != base_ids:
            raise LabelBoundaryError(
                f"label {label!r} does not add exactly one token after {prefill_text!r}"
            )
        label_ids.append(full[-1])

    if len(set(label_ids)) != len(label_ids):
        raise LabelBoundaryError("labels collide onto the same token id")
    return label_ids


@dataclass(frozen=True)
class BranchPlan:
    """How one choice question's options map onto scored branches.

    A flat plan scores every option in a single branch. A split plan scores each
    option group in its own branch with an escape label, and combine_escape weighs the
    groups by the mass each keeps off that label.
    """

    groups: tuple[tuple[int, ...], ...]
    split: bool


def plan_branches(option_count: int, *, abstain: bool = False) -> BranchPlan:
    """Split options into groups only when they exceed one branch's labels.

    A split group holds one label back for the escape label that lets it say the
    answer is not among its members. abstain means a flat plan carries that label
    as well, so it has room for one option fewer.
    """
    group_count = 0
    group_size = 0
    capacity = MAX_LABELS_PER_BRANCH - 1
    groups: list[tuple[int, ...]] = []

    if option_count + (1 if abstain else 0) <= MAX_LABELS_PER_BRANCH:
        return BranchPlan(groups=(tuple(range(option_count)),), split=False)

    group_count = -(-option_count // capacity)
    if group_count > capacity:
        raise LabelBoundaryError(
            f"{option_count} options needs {group_count} groups, over the "
            f"{capacity} label limit"
        )
    group_size = -(-option_count // group_count)
    for start in range(0, option_count, group_size):
        groups.append(tuple(range(start, min(start + group_size, option_count))))
    return BranchPlan(groups=tuple(groups), split=True)
