"""The text side of tagging: parse a proposal, split a prompt, and clean the items.

Stdlib only, since the ComfyUI tagging packs import it beside their own torch.
"""

from __future__ import annotations

import re

# Bounds the packed verify pass. In the Logit Tagger's tests-AB/ab_tagger.py a 42 candidate
# image fit one pass beside a 1 MP image.
MAX_CANDIDATES = 48

# A tag longer than this is a sentence or a run-on, not a noun phrase.
MAX_TAG_WORDS = 6
MAX_TAG_CHARS = 60

# A stop on a block of up to 4 tags repeated back to back ended every greedy loop the Logit
# Tagger's tests-AB/ab_propose_loops.py found and changed no kept tag.
MAX_REPEAT_BLOCK = 4

# The `|$` also removes an unclosed block, the shape a decode cut off mid thought leaves.
_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
_SEPARATORS = re.compile(r"[,;\n]")
# A numbered bullet needs whitespace after it, so a tag like "2.5 liter bottle" survives.
_BULLET = re.compile(r"^(?:[-*\u2022]\s*|\d+[.)](?:\s+|$))")
_QUOTES = "\"'`\u201c\u201d\u2018\u2019"

# A period or colon between two digits is part of a number or a ratio, such as 2.5 or 16:9.
_PROMPT_DIVIDERS = re.compile(r"[,;!?\n\r()\[\]{}|/\"<>]|(?<!\d)[.:]|[.:](?!\d)")
_FRAGMENT_EDGES = " '`*-_"

_LEADING_FILLER = frozenset({"a", "an", "the", "and", "with", "of"})
_WORD_BREAKS = re.compile(r"[\s-]+")


def clean_item(text: str) -> str:
    """Return one list item without its bullet, quotes and trailing periods, lowercased."""
    item = _BULLET.sub("", text.strip())

    item = item.strip(" \t" + _QUOTES).rstrip(". \t" + _QUOTES)
    return " ".join(item.lower().split())


def parse_candidates(text: str, *, max_candidates: int = MAX_CANDIDATES, max_words: int = MAX_TAG_WORDS,
                     max_chars: int = MAX_TAG_CHARS) -> list[str]:
    """Turn proposal text into unique candidate tags, in the order they were written."""
    body = _THINK_BLOCK.sub("", text).strip()
    lines = body.split("\n")
    candidates: list[str] = []
    seen: set[str] = set()

    # The stray role line an empty think block provoked from Qwen3-VL in the Logit Tagger's probe.
    if lines and lines[0].strip() == "assistant":
        body = "\n".join(lines[1:])
    for item in _SEPARATORS.split(body):
        tag = clean_item(item)
        if not tag or len(tag) > max_chars or len(tag.split()) > max_words or tag in seen:
            continue
        seen.add(tag)
        candidates.append(tag)
        if len(candidates) == max_candidates:
            break
    return candidates


def complete_tags(text: str) -> list[str]:
    """Return the tags a separator follows, since the text after the last one may be mid tag."""
    parts = _SEPARATORS.split(text)[:-1]

    return [tag for tag in (clean_item(part) for part in parts) if tag]


def repeated_block(tags: list[str], max_block: int = MAX_REPEAT_BLOCK) -> int:
    """Return the size of the block of tags the list just wrote twice back to back, or 0."""
    for size in range(1, max_block + 1):
        if len(tags) >= 2 * size and tags[-size:] == tags[-2 * size:-size]:
            return size
    return 0


def split_prompt(prompt: str) -> list[str]:
    """Split a prompt into its unique lowercase fragments, in the order they were written."""
    fragments: list[str] = []
    seen: set[str] = set()

    for part in _PROMPT_DIVIDERS.split(prompt):
        fragment = " ".join(part.split()).strip(_FRAGMENT_EDGES).lower()
        if not any(char.isalpha() for char in fragment) or fragment in seen:
            continue
        seen.add(fragment)
        fragments.append(fragment)
    return fragments


def normalize_item(text: str) -> str:
    """Return the cleaned item without its leading articles and joiners, keeping at least one word."""
    words = clean_item(text).split()

    while len(words) > 1 and words[0] in _LEADING_FILLER:
        words = words[1:]
    return " ".join(words)


def _word_set(item: str) -> frozenset[str]:
    return frozenset(word for word in _WORD_BREAKS.split(item) if word)


def drop_subsets(items: list[str]) -> list[str]:
    """Drop every item whose words another item holds, keeping the first of equal word sets.

    Words split on whitespace and hyphens, so "sun-dried tomato" holds "tomato".
    """
    word_sets = [_word_set(item) for item in items]
    kept: list[str] = []

    for index, (item, words) in enumerate(zip(items, word_sets, strict=True)):
        if any(words < other for other in word_sets) or words in word_sets[:index]:
            continue
        kept.append(item)
    return kept
