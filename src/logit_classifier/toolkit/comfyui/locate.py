"""The position of each tag on six strips of an image.

The verify statement is scored on the top, center and bottom thirds at full width and the left, center and right thirds
at full height. A tag no strip holds is dropped, and an axis where exactly one strip holds the tag names its position,
such as top-left.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from ...classifier import Classifier
from ..tags import PRESENCE_STATEMENT, presence_statement
from .tagger import resample_image

# ContextAnchoredTileRefine's STRIP_MEGAPIXELS = 0.25. Strips at 0.5 MP gave 303 judged position terms against 338,
# with 3 wrong in each (ContextAnchoredTileRefine tests-AB/ab_tags_fp8_arms.py arm fp8-strip05,
# ContextAnchoredTileRefine tests-AB/tags-bench-log.md section 12).
STRIP_PIXELS = 250_000

# ContextAnchoredTileRefine settings.toml tile_tags_position_threshold.
POSITION_THRESHOLD = 0.9

ROW_WORDS = ("top", "center", "bottom")
COLUMN_WORDS = ("left", "center", "right")

# Each strip spans a third of one side, so a side below 3 pixels leaves some strip with no pixels.
_MIN_SIDE = 3


class StripBox(NamedTuple):
    """One strip in image pixels, from x0 and y0 up to but not including x1 and y1."""

    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True, kw_only=True)
class TagPlaces:
    """One image's pass through locate_tags.

    strips holds per tag its six scores in strip_boxes order, and terms one position term per tag, "" for none.
    unplaced holds the tags no strip holds, and placed pairs every other tag with its term, in input order.
    """

    strips: tuple[tuple[float, ...], ...]
    terms: tuple[str, ...]
    unplaced: tuple[str, ...]
    placed: tuple[tuple[str, str], ...]


def strip_boxes(height: int, width: int) -> tuple[StripBox, ...]:
    """Return the top, center and bottom thirds at full width, then the left, center and right thirds at full height."""
    rows = [height * index // 3 for index in range(4)]
    columns = [width * index // 3 for index in range(4)]

    return (*(StripBox(0, rows[index], width, rows[index + 1]) for index in range(3)),
            *(StripBox(columns[index], 0, columns[index + 1], height) for index in range(3)))


def axis_word(probabilities: Sequence[float], words: Sequence[str], threshold: float) -> str | None:
    """Return the word of the one strip on this axis that holds the tag, or None when no strip or more than one does.

    Two adjacent strips named the center or the side by a coin flip, and a term covering part of a spread tag points
    the diffusion model at the wrong place.
    """
    present = [index for index, p in enumerate(probabilities) if p >= threshold]

    if len(present) != 1:
        return None
    return words[present[0]]


def position_term(row: str | None, column: str | None) -> str:
    """Return "<row>-<column>", or one word when an axis names nothing or the column is the center.

    With neither axis naming a word the term is "".
    """
    if row is None or column is None:
        return row or column or ""
    if column == "center":
        return row
    return f"{row}-{column}"


def strip_scores(
    classifier: Classifier, image: Any, tags: Sequence[str], *, statement: str, pixels: int
) -> tuple[tuple[float, ...], ...]:
    """Return per tag the statement's score on each strip of a [1, H, W, C] image, in strip_boxes order.

    Each strip is resampled to pixels total pixels and asked about every tag in one nouls call. The strips fill
    the statement through presence_statement, so they read the statement verify read.
    """
    statements = [presence_statement(tag, wording=statement) for tag in tags]
    per_strip: list[list[float]] = []

    for box in strip_boxes(int(image.shape[1]), int(image.shape[2])):
        strip = resample_image(image[:, box.y0:box.y1, box.x0:box.x1, :], pixels)
        per_strip.append(classifier.nouls(statements, image=strip))
    return tuple(tuple(scores[index] for scores in per_strip) for index in range(len(tags)))


def strip_term(strips: Sequence[float], threshold: float) -> str:
    """Return the position term of one tag's six strip scores."""
    return position_term(axis_word(strips[:3], ROW_WORDS, threshold), axis_word(strips[3:], COLUMN_WORDS, threshold))


def on_no_strip(strips: Sequence[float], threshold: float) -> bool:
    """Return whether no strip holds the tag.

    The entire image's score can pass a tag that no part of the image shows. On the owner's storm sky tile in
    ContextAnchoredTileRefine every such tag named something the tile lacks.
    """
    return max(strips) < threshold


def locate_tags(
    classifier: Classifier,
    image: Any,
    tags: Sequence[str],
    *,
    statement: str = PRESENCE_STATEMENT,
    threshold: float = POSITION_THRESHOLD,
    pixels: int = STRIP_PIXELS,
) -> TagPlaces:
    """Score every tag on the six strips of a [1, H, W, C] image, drop the tags no strip holds and place the rest.

    A strip holds a tag when the tag's score there reaches threshold. Any number of tags costs six nouls calls, and no
    tags cost none.
    """
    batch, height, width = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    strips: tuple[tuple[float, ...], ...] = ()
    terms: tuple[str, ...] = ()
    unplaced: tuple[str, ...] = ()
    placed: tuple[tuple[str, str], ...] = ()

    if batch != 1:
        raise ValueError(f"locate_tags got a batch of {batch} images. Pass one [1, H, W, C] image per call, since "
                         f"core's tokenizer attaches one image per request.")
    if height < _MIN_SIDE or width < _MIN_SIDE:
        raise ValueError(f"locate_tags got an image of height {height} and width {width}. Pass an image at least "
                         f"{_MIN_SIDE} pixels on each side, since each strip spans a third of one side.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"locate_tags got threshold={threshold!r}. Use a value from 0 to 1.")
    if not tags:
        return TagPlaces(strips=strips, terms=terms, unplaced=unplaced, placed=placed)

    strips = strip_scores(classifier, image, tags, statement=statement, pixels=pixels)
    terms = tuple(strip_term(scores, threshold) for scores in strips)
    unplaced = tuple(tag for tag, scores in zip(tags, strips, strict=True) if on_no_strip(scores, threshold))
    placed = tuple((tag, term) for tag, term, scores in zip(tags, terms, strips, strict=True)
                   if not on_no_strip(scores, threshold))
    return TagPlaces(strips=strips, terms=terms, unplaced=unplaced, placed=placed)


def render_placed(placed: Sequence[tuple[str, str]]) -> str:
    """Return "<tag> <term>" per pair, or the tag alone without a term, joined by ", "."""
    return ", ".join(f"{tag} {term}" if term else tag for tag, term in placed)
