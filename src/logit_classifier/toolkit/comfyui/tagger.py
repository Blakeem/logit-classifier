"""The tagging stages both ComfyUI packs run: the prompt listing, propose, merge, the thing check, verify and clean.

Each pack passes its own TagSettings, so the stages live here once.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ...classifier import Classifier
from ...schema import ChoiceAnswer, ChoiceQuestion, JSONContent, Question, SystemOneRequest
from ..tags import (
    MergedCandidates,
    drop_subsets,
    in_prompt,
    merge_candidates,
    parse_candidates,
    passed_by_origin,
    prompt_word_forms,
    tag_stop,
)
from .generate import generate_text
from .scopes import shared_vision_encode

_PROMPT = "{PROMPT}"
_TAG = "{TAG}"
_THING_CHECKS = ("all", "prompt")
_SUBSETS_RULES = ("words", "head-noun")
_ORDERS = ("score", "proposed")


@dataclass(frozen=True, kw_only=True)
class TagSettings:
    """Every wording, budget, cap and threshold of the tagging stages.

    The defaults are the current recommended values, and a later measurement may change them. A caller that needs
    fixed output passes every value it relies on.
    """

    # The fastest of eight wordings on 28 images, with the fewest rejected tags (the Logit Tagger's
    # tests-AB/ab_propose_prompts.py).
    propose_instruction: str = (
        "Tag this image. List every distinct visible thing in it: objects, people, animals, "
        "clothing, materials, and the setting. Use short lowercase noun phrases separated by "
        "commas. List each thing once. Do not list moods, styles, or ideas. Output only the tags."
    )
    # A reply of only the propose wording's category nouns left 4 of 174 images without a tag (the Logit Tagger's
    # tests-AB/ab_tagger_compare.py). This wording names no category.
    echo_fallback_instruction: str | None = (
        "List the things you can see in this image as short lowercase noun phrases separated by commas. "
        "Output only the list."
    )
    # In the Logit Tagger's tests-AB/ab_propose_budget.py both images that reached 128 tokens were repeating tags,
    # and 192 added no tag.
    propose_max_tokens: int = 128
    # On fp8, 40 over 25 added 0.85 correct and 0.15 wrong or vague tags per render, and 0.87 and 0.10 per tile
    # (the Logit Tagger's tests-AB/ab_tagger_score.py --replay, ContextAnchoredTileRefine tests-AB/ab_tags_bench.py).
    propose_cap: int = 25
    # The model's 25 and a long prompt's 30 to 40 tags, in one packed verify pass beside a 1 MP picture
    # (ContextAnchoredTileRefine tests-AB/tags-bench-log.md).
    merge_cap: int = 64
    # The propose wording's own nouns came back as tags and passed verify (ContextAnchoredTileRefine
    # tests-AB/ab_tile_tags.py).
    category_nouns: frozenset[str] = frozenset({"objects", "people", "animals", "clothing", "materials", "setting"})

    # A 500 word prompt listed 28 to 45 tags in 70 to 256 tokens (ContextAnchoredTileRefine tags-bench-log.md). Past
    # the prompt's own things the model pads the list with words the prompt lacks, so a streak of them ends it.
    prompt_tags_instruction: str = (
        "This is a prompt for an image:\n{PROMPT}\n\nList the physical things the prompt names that "
        "could be pointed at in the image: objects, people, animals, plants, clothing, materials, "
        "buildings and parts of the scene. Use short lowercase noun phrases of one to four words, "
        "separated by commas. Keep each thing's own descriptive words, such as its color. Leave out "
        "places and settings, actions, sizes, moods, the image's style, medium, quality, camera and "
        "lighting, artist names and position words. List each thing once. Output only the list."
    )
    prompt_tags_max_tokens: int = 256
    ungrounded_streak: int = 2

    # The question read 0.000 to 0.027 on 13 prompts in 10 other languages and 1.000 on 10 English ones (the Logit
    # Tagger's tests-AB/ab_prompt_language.py). 512 tokens hold about 380 English words.
    language_question: str | None = "Is this text written in English?"
    english_threshold: float = 0.5
    translate_instruction: str = "Translate this image prompt into English. Output only the translation.\n\n{PROMPT}"
    translate_max_tokens: int = 512

    # No verify threshold separates scene, lighting, color and quality words from things. On fp8 this wording read
    # AUC 0.839 and five others 0.546 to 0.846 (the Logit Tagger's tests-AB/ab_thing_wordings.py).
    thing_question: str = 'What does the image tag "{TAG}" name'
    thing_criteria: tuple[tuple[str, str], ...] = (
        ("thing", "something that could be pointed at in a picture: an object, a person, a group of people "
                  "or animals, an animal, a plant, a body part, a garment, a material, a building, a "
                  "landscape feature such as hills, cliffs, sky, the moon or stars, a light source such as "
                  "lamps, signs or lit windows, or a substance such as water, smoke, fire, clouds or rain"),
        ("other", "nothing to point at: a kind of place or a whole scene, a time of day, a color alone, a "
                  "shape or form, lighting in general, a camera or render effect, a style, a mood, a "
                  "quality or an idea"),
    )
    # Mid scores mix things with non-things ("robots" 0.85 on fp8), so only a confident "other" drops a tag.
    thing_threshold: float = 0.9
    # Judging only the prompt's tags kept "watermark" and "pixel art", and added 0.28 vague tags per render and 0.81
    # per tile on fp8 (ab_tagger_score.py --replay, ab_tags_bench.py).
    thing_check: str = "all"

    # The crop wording "This crop visibly contains {f}" reached AUC 0.963 in this repo's tests-AB/ab_multi_label.py.
    # "Is there {TAG} in this image?" removed 34 correct tile tags per 4 wrong ones (ab_tags_bench.py).
    verify_statement: str = "This image visibly contains {TAG}"
    # On fp8, 0.999 over 0.9 removed a wrong or vague tag per 3 to 4 correct ones on renders, photos and tiles, and
    # 0.9999 cost about 50 correct tile tags per further one (ab_tagger_score.py --replay, ab_tags_bench.py).
    verify_threshold: float = 0.999
    # A near name passes 0.9, and 0.9999 cut wrong tags per tile from 0.92 to 0.29 (ContextAnchoredTileRefine
    # tests-AB/tags-bench-log.md).
    prompt_only_threshold: float = 0.9999

    # The "head-noun" rule keeps "cat" beside "cat ear", and also "stone" beside "stone walls" and "castle" beside
    # "stone castle tower". On fp8 tiles it added 0.03 wrong tags per tile (ab_tags_bench.py).
    subsets_rule: str = "words"
    order: str = "score"

    def __post_init__(self) -> None:
        wordings = [
            ("prompt_tags_instruction", self.prompt_tags_instruction, _PROMPT),
            ("translate_instruction", self.translate_instruction, _PROMPT),
            ("thing_question", self.thing_question, _TAG),
        ]
        thresholds = {
            "english_threshold": self.english_threshold,
            "thing_threshold": self.thing_threshold,
            "verify_threshold": self.verify_threshold,
            "prompt_only_threshold": self.prompt_only_threshold,
        }
        budgets = {
            "propose_max_tokens": self.propose_max_tokens,
            "propose_cap": self.propose_cap,
            "merge_cap": self.merge_cap,
            "prompt_tags_max_tokens": self.prompt_tags_max_tokens,
            "translate_max_tokens": self.translate_max_tokens,
        }
        choices = {
            "thing_check": (self.thing_check, _THING_CHECKS),
            "subsets_rule": (self.subsets_rule, _SUBSETS_RULES),
            "order": (self.order, _ORDERS),
        }
        criteria_keys = {key for key, _text in self.thing_criteria}

        # An empty verify statement turns verify off, so it needs no placeholder.
        if self.verify_statement:
            wordings.append(("verify_statement", self.verify_statement, _TAG))
        for name, wording, placeholder in wordings:
            if placeholder not in wording:
                raise ValueError(f"TagSettings got a {name} without {placeholder}. Add {placeholder} where the "
                                 f"stage inserts its text.")
        for name, threshold in thresholds.items():
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"TagSettings got {name}={threshold!r}. Use a value above 0 and at most 1.")
        for name, budget in budgets.items():
            if budget < 1:
                raise ValueError(f"TagSettings got {name}={budget!r}. Use 1 or more.")
        if self.ungrounded_streak < 0:
            raise ValueError(f"TagSettings got ungrounded_streak={self.ungrounded_streak!r}. Use 0 or more, where 0 "
                             f"turns the streak stop off.")
        if self.merge_cap < self.propose_cap:
            raise ValueError(f"TagSettings got merge_cap={self.merge_cap!r} below propose_cap={self.propose_cap!r}. "
                             f"Use a merge_cap of at least propose_cap, so the merge keeps every proposed tag.")
        for name, (value, allowed) in choices.items():
            if value not in allowed:
                raise ValueError(f"TagSettings got {name}={value!r}. Use {' or '.join(map(repr, allowed))}.")
        if not {"thing", "other"} <= criteria_keys:
            raise ValueError(f"TagSettings got thing_criteria with the keys {sorted(criteria_keys)!r}. Give a "
                             f'"thing" and an "other" criterion.')


@dataclass(frozen=True, kw_only=True)
class PromptTags:
    """One prompt's listing, for the pack to merge into every picture's candidates.

    text is the English text the listing read. listed holds the tags in_prompt keeps, p_other each one's thing check
    score, and tags the listed tags the thing check keeps.
    """

    text: str
    reply: str
    listed: tuple[str, ...]
    p_other: tuple[float, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class TagTrace:
    """One picture's pass through every stage.

    echo_reply holds the replaced reply when the echo fallback ran. dropped holds (name, reason) pairs from the merge
    and the thing check. scores holds None per candidate with verify off. verified is before drop_subsets, and kept
    is in the settings' order.
    """

    reply: str
    echo_reply: str | None
    proposed: tuple[str, ...]
    candidates: tuple[str, ...]
    origins: tuple[str, ...]
    dropped: tuple[tuple[str, str], ...]
    scores: tuple[float | None, ...]
    verified: tuple[str, ...]
    kept: tuple[str, ...]
    propose_ms: float
    verify_ms: float


def fit_picture(image: Any, max_pixels: int) -> Any:
    """Return the first three channels of a [B, H, W, C] image, area-downscaled to at most max_pixels, never upscaled.

    Core resizes a vision image only above 12.8 MP, so a 4k input would cost about 12k vision tokens.
    """
    import torch

    height, width = int(image.shape[1]), int(image.shape[2])
    rgb = image[..., :3]
    scale = 0.0
    fitted_height = 0
    fitted_width = 0

    if max_pixels < 1:
        raise ValueError(f"fit_picture got max_pixels={max_pixels!r}. Use 1 or more.")
    if height * width <= max_pixels:
        return rgb
    scale = math.sqrt(max_pixels / (height * width))
    # Floored sides, each held to the cap, so the product stays within it at any aspect ratio.
    fitted_height = min(max_pixels, max(1, math.floor(height * scale)))
    fitted_width = max(1, min(math.floor(width * scale), max_pixels // fitted_height))
    return torch.nn.functional.interpolate(
        rgb.movedim(-1, 1), size=(fitted_height, fitted_width), mode="area"
    ).movedim(1, -1)


def thing_scores(
    classifier: Classifier, tags: Sequence[str], settings: TagSettings, known: dict[str, float]
) -> tuple[float, ...]:
    """Return the thing check's p("other") per tag, asking every tag missing from known in one text-only request.

    known carries the answers across a run, so each tag is asked once per known.
    """
    unknown = tuple(dict.fromkeys(tag for tag in tags if tag not in known))
    criteria: dict[str, JSONContent | None] = dict(settings.thing_criteria)
    questions: dict[str, Question] = {}

    if unknown:
        # A literal replace, so braces in a tag are never read as format fields.
        questions = {f"t{index}": ChoiceQuestion(instructions=settings.thing_question.replace(_TAG, tag),
                                                 criteria=criteria)
                     for index, tag in enumerate(unknown)}
        response, _diagnostics = classifier.classify(SystemOneRequest(state="", questions=questions))
        for tag, qid in zip(unknown, questions, strict=True):
            answer = response.answers[qid]
            if not isinstance(answer, ChoiceAnswer):
                raise TypeError(f"thing question {qid} got a {answer.type} answer, not a choice")
            known[tag] = answer.probabilities["other"]
    return tuple(known[tag] for tag in tags)


def english_text(clip: Any, classifier: Classifier, prompt: str, settings: TagSettings) -> str:
    """Return the prompt when the language gate is off or reads it as English, otherwise its English translation.

    A blank translation gives the prompt back.
    """
    translation = ""

    if settings.language_question is None:
        return prompt
    if classifier.nouls([settings.language_question], state=prompt)[0] >= settings.english_threshold:
        return prompt
    # A literal replace, so braces in the prompt are never read as format fields.
    translation = generate_text(clip, settings.translate_instruction.replace(_PROMPT, prompt),
                                max_tokens=settings.translate_max_tokens, whole_tags=False).strip()
    return translation or prompt


def prompt_tags(
    clip: Any, classifier: Classifier, prompt: str, settings: TagSettings, known: dict[str, float]
) -> PromptTags:
    """List the physical things the prompt names that its English text grounds and the thing check keeps.

    The listing reads the prompt's English text, since in_prompt and the streak stop match the text's own words.
    """
    text = ""
    word_forms: frozenset[str] = frozenset()
    reply = ""
    listed: tuple[str, ...] = ()
    p_other: tuple[float, ...] = ()

    if not prompt.strip():
        return PromptTags(text="", reply="", listed=(), p_other=(), tags=())
    text = english_text(clip, classifier, prompt, settings)
    word_forms = prompt_word_forms(text)
    reply = generate_text(clip, settings.prompt_tags_instruction.replace(_PROMPT, text),
                          max_tokens=settings.prompt_tags_max_tokens,
                          should_stop=tag_stop(max_tags=None, ungrounded_streak=settings.ungrounded_streak,
                                               word_forms=word_forms))
    listed = tuple(tag for tag in parse_candidates(reply) if in_prompt(tag, word_forms))
    p_other = thing_scores(classifier, listed, settings, known)
    return PromptTags(text=text, reply=reply, listed=listed, p_other=p_other,
                      tags=tuple(tag for tag, p in zip(listed, p_other, strict=True) if p < settings.thing_threshold))


def _propose(clip: Any, picture: Any, instruction: str, settings: TagSettings) -> tuple[str, tuple[str, ...]]:
    reply = generate_text(clip, instruction, image=picture, max_tokens=settings.propose_max_tokens,
                          should_stop=tag_stop(max_tags=settings.propose_cap, ungrounded_streak=0))

    return reply, tuple(parse_candidates(reply, max_candidates=settings.propose_cap))


def _things_only(classifier: Classifier, merged: MergedCandidates, settings: TagSettings,
                 known: dict[str, float]) -> MergedCandidates:
    p_other = thing_scores(classifier, merged.candidates, settings, known)
    is_thing = [p < settings.thing_threshold for p in p_other]
    not_things = tuple((tag, "not a thing") for tag, thing in zip(merged.candidates, is_thing, strict=True)
                       if not thing)

    return MergedCandidates(tuple(itertools.compress(merged.candidates, is_thing)),
                            tuple(itertools.compress(merged.origins, is_thing)), merged.dropped + not_things)


def tag_picture(
    clip: Any,
    classifier: Classifier,
    picture: Any,
    *,
    prompt_tags: Sequence[str] = (),
    settings: TagSettings,
    known: dict[str, float],
    on_stage: Callable[[str], None] | None = None,
) -> TagTrace:
    """Propose, merge with prompt_tags, check, verify and clean the tags of a picture the caller already fitted.

    Propose and verify run in one shared_vision_encode scope, so the vision tower encodes the picture once. on_stage
    hears "propose", then "verify".
    """
    announce = on_stage if on_stage is not None else (lambda _stage: None)
    thresholds = {"model": settings.verify_threshold, "both": settings.verify_threshold,
                  "prompt": settings.prompt_only_threshold}
    reply = ""
    echo_reply: str | None = None
    proposed: tuple[str, ...] = ()
    merged = MergedCandidates((), (), ())
    verify_scores: tuple[float, ...] = ()
    scores: tuple[float | None, ...] = ()
    verified: tuple[str, ...] = ()
    kept: tuple[str, ...] = ()
    score_of: dict[str, float] = {}
    started = 0.0
    propose_ms = 0.0
    verify_ms = 0.0

    with shared_vision_encode(clip):
        announce("propose")
        started = time.perf_counter()
        reply, proposed = _propose(clip, picture, settings.propose_instruction, settings)
        if (settings.echo_fallback_instruction is not None
                and not merge_candidates(proposed, (), cap=settings.merge_cap,
                                         drop=settings.category_nouns).candidates):
            echo_reply = reply
            reply, proposed = _propose(clip, picture, settings.echo_fallback_instruction, settings)
        propose_ms = (time.perf_counter() - started) * 1000.0
        merged = merge_candidates(proposed, prompt_tags, cap=settings.merge_cap, drop=settings.category_nouns)
        if settings.thing_check == "all":
            merged = _things_only(classifier, merged, settings, known)
        announce("verify")
        started = time.perf_counter()
        if settings.verify_statement:
            verify_scores = tuple(classifier.nouls(
                [settings.verify_statement.replace(_TAG, tag) for tag in merged.candidates], image=picture))
            verified = passed_by_origin(merged.candidates, merged.origins, verify_scores, thresholds=thresholds)
            scores = verify_scores
        else:
            verified = merged.candidates
            scores = (None,) * len(merged.candidates)
        verify_ms = (time.perf_counter() - started) * 1000.0

    kept = tuple(drop_subsets(list(verified), rule=settings.subsets_rule))
    if settings.order == "score" and settings.verify_statement:
        score_of = dict(zip(merged.candidates, verify_scores, strict=True))
        # sorted is stable, so tied tags keep candidate order.
        kept = tuple(sorted(kept, key=score_of.__getitem__, reverse=True))
    return TagTrace(reply=reply, echo_reply=echo_reply, proposed=proposed, candidates=merged.candidates,
                    origins=merged.origins, dropped=merged.dropped, scores=scores, verified=verified, kept=kept,
                    propose_ms=propose_ms, verify_ms=verify_ms)
