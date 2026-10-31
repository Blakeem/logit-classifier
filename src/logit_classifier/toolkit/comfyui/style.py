"""The style line of an entire image, and the medium and style words a prompt names."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ...classifier import Classifier
from ..captions import clean_caption
from ..tags import in_prompt, parse_candidates, prompt_word_forms, tag_stop
from .describe import describe_image
from .generate import generate_text
from .tagger import _PROMPT, _TAG, TagSettings, _criterion_scores, resample_image, to_english

# ContextAnchoredTileRefine's MAX_CAPTION_TOKENS, the range its settings.toml gives every caption budget.
_MAX_TOKENS_CEILING = 4096


@dataclass(frozen=True, kw_only=True)
class StyleSettings:
    """The wordings, budgets, pixel count and threshold of the style line and the prompt style listing.

    The style line's defaults are ContextAnchoredTileRefine's shipped values. The prompt listing's defaults won on 152
    prompts that two blind judges graded, in ComfyUI-LogitTagger's tests-AB/ab_style.py run 2026-10-17-prompt-superset.
    """

    # ContextAnchoredTileRefine settings.toml global_style_instruction.
    image_instruction: str = (
        "Concise prose containing only the style of media used and the general style within that type of media."
    )
    # ContextAnchoredTileRefine settings.toml global_style_max_tokens, which also covers the reasoning turn.
    image_max_tokens: int = 768
    # ContextAnchoredTileRefine settings.toml [vision] caption_megapixels = 0.786432, as caption_budget_pixels
    # rounds it.
    image_pixels: int = 786432

    # The tagger's prompt listing with a none answer, since without one the model copies the things out of a prompt
    # that names no medium.
    prompt_instruction: str = (
        "This is a prompt for an image:\n{PROMPT}\n\nList the words of the prompt that name how the image is "
        "made: its medium, such as a photograph, a painting, a drawing or a 3D render, and the style within that "
        "medium, such as a technique, an art movement or an artist. Copy each phrase from the prompt. Use short "
        "lowercase phrases separated by commas. Leave out the things, people, places, actions, lighting and mood. "
        "List each phrase once. If the prompt names no medium and no style, output none. Output only the list."
    )
    prompt_max_tokens: int = 128
    # TagSettings' streak for its prompt listing, since past the prompt's own words the model pads the list.
    ungrounded_streak: int = 2
    # Naming both choices in the question kept media the thing check's wording scored as things, such as
    # "comic book panel".
    style_question: str = 'In an image prompt, does the phrase "{TAG}" say how the picture is made or what it shows'
    style_criteria: tuple[tuple[str, str], ...] = (
        ("style", "how a picture is made or styled: a medium such as a photograph, a painting, a drawing, a print "
                  "or a 3D render, a technique such as watercolor, oil paint, impasto or pixel art, an art "
                  "movement, an artist's style, or a film or camera look"),
        ("other", "what a picture shows: an object, a person, an animal, a place, an action, a time of day, a "
                  "color, lighting or a mood"),
    )
    # 0.2 to 0.3 kept the same wrong phrases, and 0.5 dropped "comic book panel" at p("style") 0.45.
    style_threshold: float = 0.2

    def __post_init__(self) -> None:
        budgets = {"image_max_tokens": self.image_max_tokens, "prompt_max_tokens": self.prompt_max_tokens}
        wordings = (("prompt_instruction", self.prompt_instruction, _PROMPT),
                    ("style_question", self.style_question, _TAG))

        if not self.image_instruction.strip():
            raise ValueError(f"StyleSettings got image_instruction={self.image_instruction!r}. Give the instruction "
                             f"the style line asks about the image.")
        for name, budget in budgets.items():
            if not 1 <= budget <= _MAX_TOKENS_CEILING:
                raise ValueError(f"StyleSettings got {name}={budget!r}. Use 1 to {_MAX_TOKENS_CEILING}.")
        if self.image_pixels < 1:
            raise ValueError(f"StyleSettings got image_pixels={self.image_pixels!r}. Use 1 or more.")
        for name, wording, placeholder in wordings:
            if placeholder not in wording:
                raise ValueError(f"StyleSettings got {name}={wording!r}. Add {placeholder} where the stage inserts "
                                 f"its text.")
        if self.ungrounded_streak < 0:
            raise ValueError(f"StyleSettings got ungrounded_streak={self.ungrounded_streak!r}. Use 0 or more, where "
                             f"0 turns the streak stop off.")
        if len(self.style_criteria) < 2 or self.style_criteria[0][0] != "style":
            raise ValueError(f"StyleSettings got style_criteria={self.style_criteria!r}. Give two or more criteria, "
                             f'the first named "style".')
        if not 0.0 <= self.style_threshold <= 1.0:
            raise ValueError(f"StyleSettings got style_threshold={self.style_threshold!r}. Use a value from 0 to 1.")


@dataclass(frozen=True, kw_only=True)
class PromptStyleTrace:
    """One prompt's pass through prompt_style.

    text is the English text the listing read. listed holds the phrases in_prompt keeps, p_style each one's p("style"),
    and style the listed phrases at or above the style threshold, in listed order.
    """

    text: str
    reply: str
    listed: tuple[str, ...]
    p_style: tuple[float, ...]
    style: tuple[str, ...]


def style_line(clip: Any, image: Any, *, settings: StyleSettings, node: str) -> str:
    """Return the style line of a [1, H, W, C] image.

    The line names the medium of the entire image and the style within that medium. ContextAnchoredTileRefine places
    one such line on top of every tile's text, so that all tiles follow one style.
    """
    batch = int(image.shape[0])
    reply = ""

    if batch != 1:
        raise ValueError(f"{node}: style_line got a batch of {batch} images. Pass one [1, H, W, C] image per call, "
                         f"since core's tokenizer attaches one image per request.")
    reply = describe_image(clip, resample_image(image, settings.image_pixels), settings.image_instruction,
                           max_tokens=settings.image_max_tokens, node=node)
    return clean_caption(reply)


def style_scores(
    classifier: Classifier, tags: Sequence[str], settings: StyleSettings, style_cache: dict[str, float]
) -> tuple[float, ...]:
    """Return p("style") per tag, asking every tag missing from style_cache in one text-only request.

    style_cache carries the answers across a run, so each tag is asked once per cache.
    """
    return _criterion_scores(classifier, tags, question=settings.style_question, criteria=settings.style_criteria,
                             criterion="style", cache=style_cache)


def prompt_style(
    clip: Any,
    classifier: Classifier,
    prompt: str,
    *,
    settings: StyleSettings,
    tag_settings: TagSettings,
    style_cache: dict[str, float],
) -> PromptStyleTrace:
    """List the medium and style phrases the prompt names that its English text grounds and the style check keeps.

    Every listed phrase grounds in the English text's words, so the stage never names a style the prompt lacks.
    tag_settings supplies the language gate and the translation.
    """
    text = ""
    word_forms: frozenset[str] = frozenset()
    reply = ""
    listed: tuple[str, ...] = ()
    p_style: tuple[float, ...] = ()

    if not prompt.strip():
        return PromptStyleTrace(text="", reply="", listed=(), p_style=(), style=())
    text = to_english(clip, classifier, prompt, tag_settings)
    word_forms = prompt_word_forms(text)
    # A literal replace, so braces in the text are never read as format fields.
    reply = generate_text(clip, settings.prompt_instruction.replace(_PROMPT, text),
                          max_tokens=settings.prompt_max_tokens,
                          should_stop=tag_stop(max_tags=None, ungrounded_streak=settings.ungrounded_streak,
                                               word_forms=word_forms))
    listed = tuple(phrase for phrase in parse_candidates(reply) if in_prompt(phrase, word_forms))
    p_style = style_scores(classifier, listed, settings, style_cache)
    return PromptStyleTrace(text=text, reply=reply, listed=listed, p_style=p_style,
                            style=tuple(phrase for phrase, p in zip(listed, p_style, strict=True)
                                        if p >= settings.style_threshold))
