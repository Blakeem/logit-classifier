"""Prompt assembly: one shared prefix, one suffix per scored branch.

Everything here is text. Applying the host's chat template around it is the
backend's job, since the HF tokenizer and a ComfyUI CLIP render it differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .labels import LABEL_ALPHABET, BranchPlan, plan_branches
from .schema import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    render_content,
)

# Bumped whenever a change to branch layout or prior bucketing makes an older learned
# prior invalid.
PROMPT_VERSION = 2

SYSTEM_PROMPT = (
    "You are a classification engine. Read the context, then answer the single "
    "question about it.\n"
    "Reply with exactly one option label from the list. Labels are case-sensitive."
)

# The processor replaces this single token with one per patch, so the marker costs
# nothing to write and expands to the right length on its own.
IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"

# A split group carries this extra label so it can decline, which is what lets
# groups be weighed against each other without a separate group branch.
ESCAPE_LABEL = "none of these"
ESCAPE_DESCRIPTION = "the answer is not in this list"

BranchKind = Literal["choice", "member", "score_level", "score_joint", "noul"]


@dataclass(frozen=True)
class Branch:
    """One prompt whose final token position carries a distribution."""

    question_id: str
    kind: BranchKind
    label_count: int
    # Which option or level indices this branch's labels map back to.
    targets: tuple[int, ...]
    suffix_text: str


def _option_lines(entries: list[tuple[str, str]]) -> str:
    lines = []
    for index, (name, description) in enumerate(entries):
        label = LABEL_ALPHABET[index]
        lines.append(f"({label}) {name}: {description}" if description else f"({label}) {name}")
    return "\n".join(lines)


def _question_block(prompt_line: str, entries: list[tuple[str, str]]) -> str:
    return f"{prompt_line}\nOptions:\n{_option_lines(entries)}"


def _choice_branches(qid: str, question: ChoiceQuestion, plan: BranchPlan,
                     abstain: bool = False) -> list[Branch]:
    names: list[str] = list(question.criteria)
    prompt_line = f"Question: {render_content(question.instructions)}".rstrip()
    branches: list[Branch] = []

    if not plan.split:
        entries = [(n, render_content(question.criteria[n])) for n in names]
        if not abstain:
            return [Branch(qid, "choice", len(names), tuple(range(len(names))),
                           _question_block(prompt_line, entries))]
        # One group carrying the escape label, so the split path reads it the same way.
        entries.append((ESCAPE_LABEL, ESCAPE_DESCRIPTION))
        return [Branch(qid, "member", len(entries), tuple(range(len(names))),
                       _question_block(prompt_line, entries))]

    for group in plan.groups:
        entries = [(names[i], render_content(question.criteria[names[i]])) for i in group]
        entries.append((ESCAPE_LABEL, ESCAPE_DESCRIPTION))
        branches.append(Branch(qid, "member", len(entries), tuple(group),
                               _question_block(prompt_line, entries)))
    return branches


def _score_branches(qid: str, question: ScoreQuestion, method: str) -> list[Branch]:
    """Expand a score question into branches.

    "independent" follows Jev's stated semantics, judging each level on its own.
    "joint" offers the levels as one labelled list, which forces the model to
    discriminate between adjacent levels instead of accepting several at once.
    Levels carry letters rather than indices either way, so no level number
    reaches the model.
    """
    instructions = render_content(question.instructions)
    branches: list[Branch] = []

    if method == "joint":
        entries = [(render_content(level), "") for level in question.criteria]
        prompt_line = (
            f"Question: {instructions}\n" if instructions else ""
        ) + "Which description best matches the context?"
        return [Branch(qid, "score_joint", len(question.criteria),
                       tuple(range(len(question.criteria))),
                       _question_block(prompt_line, entries))]

    for index, level in enumerate(question.criteria):
        prompt_line = (
            f"Question: {instructions}\n" if instructions else ""
        ) + f"Statement: the context matches this description: {render_content(level)}"
        branches.append(Branch(qid, "score_level", 2, (index,),
                               _question_block(prompt_line, [("yes", ""), ("no", "")])))
    return branches


def _noul_branches(qid: str, question: NoulQuestion) -> list[Branch]:
    prompt_line = f"Statement: {render_content(question.instructions)}".rstrip()
    yes_text = render_content(question.criteria.true_) if question.criteria else ""
    no_text = render_content(question.criteria.false_) if question.criteria else ""
    return [Branch(qid, "noul", 2, (0,),
                   _question_block(prompt_line, [("yes", yes_text), ("no", no_text)]))]


def build_branches(questions: dict[str, Question], score_method: str = "joint",
                   abstain: bool = False) -> list[Branch]:
    """Expand every question into the branches that must be scored."""
    branches: list[Branch] = []

    for qid, question in questions.items():
        if isinstance(question, ChoiceQuestion):
            branches.extend(_choice_branches(
                qid, question, plan_branches(len(question.criteria), abstain=abstain), abstain
            ))
        elif isinstance(question, ScoreQuestion):
            branches.extend(_score_branches(qid, question, score_method))
        else:
            branches.extend(_noul_branches(qid, question))
    return branches


def prefix_content(state: Any, has_image: bool = False) -> str:
    """Build the user-message body every branch shares, image marker included."""
    marker = f"{IMAGE_MARKER}\n" if has_image else ""
    return f"Context:\n{marker}{render_content(state)}\n\n"


def branch_content(state: Any, branch: Branch, has_image: bool = False) -> str:
    """One branch's full user-message body, opening with the shared prefix."""
    return prefix_content(state, has_image) + branch.suffix_text
