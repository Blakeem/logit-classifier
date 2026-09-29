"""The general text tools of tagging: parse a proposal, split a prompt, clean and merge the items, and ask about one.

Stdlib only, since the ComfyUI tagging packs import it beside their own torch.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

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
# Two or more single letters joined by periods, such as u.s.a. or e.g., whose periods divide
# nothing. The final period is optional so that "u.s.a" matches as well, and a letter after
# it, as in "w.b.yeats", makes the run no initialism.
_INITIALISM = r"(?<![\w.])[^\W\d_](?:\.[^\W\d_])+(?:\.(?!\w)|(?![\w.]))"
_INITIALISMS = re.compile(_INITIALISM)
# One scan finds both, so a period inside an initialism is never read as a divider. A divider
# is never a letter, so the two alternatives cannot start at the same character.
_PROMPT_PARTS = re.compile(f"(?P<initialism>{_INITIALISM})|{_PROMPT_DIVIDERS.pattern}")
_FRAGMENT_EDGES = " '`*-_"

_LEADING_FILLER = frozenset({"a", "an", "the", "and", "with", "of"})
_WORD_BREAKS = re.compile(r"[\s-]+")

# A word that opens a modifier after the head noun, as in "wheels of cheese" or "sign reading open".
HEAD_NOUN_BREAKS = frozenset({"of", "in", "on", "with", "reading"})

# A tag joins the prompt's words with these, so in_prompt needs no prompt word for them.
PROMPT_STOP_WORDS = frozenset({"a", "an", "the", "of", "with", "and", "in", "on", "at", "to", "for", "by", "from"})
_PROMPT_WORD = re.compile(r"[a-z0-9]+")
_PLURAL_CUTS = (("s", ""), ("es", ""), ("ies", "y"))

# On judged tile candidates this matched 0.3.0's "Is there a {TAG} in this image?" within every paired interval, and it
# needs no article or number (the Logit Tagger's tests-AB/ab_presence_wording.py, round 4).
PRESENCE_STATEMENT = "This image visibly contains {TAG}"
_TAG = "{TAG}"


def _full_initialisms(text: str) -> str:
    # One spelling per initialism, so "u.s.a" and "u.s.a." are one tag. NFC first, since a
    # decomposed accent is not a word character and would end the match mid letter.
    composed = unicodedata.normalize("NFC", text)

    return _INITIALISMS.sub(lambda match: match.group().rstrip(".") + ".", composed)


def clean_item(text: str) -> str:
    """Return one list item without its bullet, quotes and trailing periods, lowercased.

    Every initialism ends in a period, such as "flag of the u.s.a.".
    """
    body = _BULLET.sub("", text.strip()).strip(" \t" + _QUOTES)
    item = _full_initialisms(body.rstrip(". \t" + _QUOTES))

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


def drop_unfinished_tag(text: str) -> str:
    """Return the text through its last separator, or "" when it has none.

    A decode that fills its token budget stops mid tag, so the text after the last separator
    is not a whole tag.
    """
    separators = list(_SEPARATORS.finditer(text))

    if not separators:
        return ""
    return text[:separators[-1].start()]


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


def _prompt_parts(prompt: str) -> list[str]:
    parts: list[str] = []
    start = 0

    for match in _PROMPT_PARTS.finditer(prompt):
        if match.group("initialism"):
            continue
        parts.append(prompt[start:match.start()])
        start = match.end()
    parts.append(prompt[start:])
    return parts


def split_prompt(prompt: str) -> list[str]:
    """Split a prompt into its unique lowercase fragments, in the order they were written.

    The periods of an initialism, such as u.s.a. or d.c., divide nothing, so an initialism
    that ends a sentence joins the next sentence's fragment. Every initialism ends in a period.
    """
    fragments: list[str] = []
    seen: set[str] = set()

    for part in _prompt_parts(_full_initialisms(prompt)):
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


def _words(item: str) -> list[str]:
    return [word for word in _WORD_BREAKS.split(item) if word]


def head_noun(phrase: str) -> str:
    """Return the last word before the phrase's first word in HEAD_NOUN_BREAKS, lowercased.

    Words split on whitespace and hyphens, so "sun-dried tomato" gives "tomato" and "wheels of cheese"
    gives "wheels". A phrase that starts with a break word, or has no words, gives its lowercased
    stripped text.
    """
    text = phrase.lower().strip()
    head_words = list(itertools.takewhile(lambda word: word not in HEAD_NOUN_BREAKS, _words(text)))

    if not head_words:
        return text
    return head_words[-1]


# "words" stays the default, since a "head-noun" default rewrote 20 of 69 recorded
# ContextAnchoredTileRefine tile tag lists with no pack release.
def drop_subsets(items: list[str], *, rule: str = "words") -> list[str]:
    """Drop an item whose word set another item's word set holds, keeping the first of equal word sets.

    Words split on whitespace and hyphens. Rule "words" drops every such item, so "cat" beside
    "cat ear" is dropped. Rule "head-noun" drops it only when both items have the same head_noun,
    so "cat" beside "cat ear" is kept. Both rules drop "signage" beside "signage in chinese".
    """
    same_head_only = rule == "head-noun"
    word_sets: list[frozenset[str]] = []
    heads: list[str] = []
    kept: list[str] = []

    if rule not in ("words", "head-noun"):
        raise ValueError(f'drop_subsets got rule {rule!r}. Use "words" or "head-noun".')

    word_sets = [frozenset(_words(item)) for item in items]
    heads = [head_noun(item) for item in items]
    for index, (item, words, head) in enumerate(zip(items, word_sets, heads, strict=True)):
        held = any(words < other and (not same_head_only or head == other_head)
                   for other, other_head in zip(word_sets, heads, strict=True))
        if held or words in word_sets[:index]:
            continue
        kept.append(item)
    return kept


def _word_forms(text: str) -> dict[str, frozenset[str]]:
    # One cut per word would stem "trees" to "tre" and leave "tree" whole, so each word keeps every cut.
    forms: dict[str, frozenset[str]] = {}

    for word in _PROMPT_WORD.findall(text.lower()):
        cuts = {word}
        for suffix, ending in _PLURAL_CUTS:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                cuts.add(word[:-len(suffix)] + ending)
        forms[word] = frozenset(cuts)
    return forms


def prompt_word_forms(prompt: str) -> frozenset[str]:
    """Return every form of every prompt word, the set in_prompt matches a tag against.

    A word is a run of ASCII letters and digits, lowercased. A word ending in "s", "es" or "ies" also
    gives each form with that ending cut, so "trees" gives "trees", "tree" and "tre".
    """
    return frozenset(itertools.chain.from_iterable(_word_forms(prompt).values()))


def in_prompt(tag: str, word_forms: frozenset[str]) -> bool:
    """Return whether every content word of the tag shares a form with a prompt word.

    A content word is a word outside PROMPT_STOP_WORDS, and a tag with no content word is not in the prompt.
    """
    content = [forms for word, forms in _word_forms(tag).items() if word not in PROMPT_STOP_WORDS]

    return bool(content) and all(forms & word_forms for forms in content)


def tag_stop(*, max_tags: int | None, ungrounded_streak: int, word_forms: frozenset[str] | None = None,
             max_block: int = MAX_REPEAT_BLOCK) -> Callable[[list[str]], bool]:
    """Return the predicate that ends a greedy tag list, given the complete tags so far.

    It holds once the list reaches max_tags, once its last ungrounded_streak tags all fail in_prompt against
    word_forms, or once repeated_block finds a block of up to max_block tags. A max_tags of None or an
    ungrounded_streak of 0 turns that test off.
    """
    streak_on = ungrounded_streak > 0
    prompt_forms = word_forms if word_forms is not None else frozenset()

    if streak_on and word_forms is None:
        raise ValueError(f"tag_stop got ungrounded_streak={ungrounded_streak} and no word_forms. Pass "
                         f"word_forms=prompt_word_forms(prompt), or ungrounded_streak=0.")

    def stop(tags: list[str]) -> bool:
        trailing = tags[-ungrounded_streak:] if streak_on else []
        full = max_tags is not None and len(tags) >= max_tags
        off_prompt = (streak_on and len(trailing) == ungrounded_streak
                      and not any(in_prompt(tag, prompt_forms) for tag in trailing))

        return full or off_prompt or repeated_block(tags, max_block) > 0

    return stop


class MergedCandidates(NamedTuple):
    """The merged candidates, each one's origin, and each left-out name with its reason.

    An origin is "model", "prompt" or "both". A reason is "category noun", "repeat" or "over the cap".
    """

    candidates: tuple[str, ...]
    origins: tuple[str, ...]
    dropped: tuple[tuple[str, str], ...]


def merge_candidates(proposed: Sequence[str], prompt_tags: Sequence[str], *, cap: int,
                     drop: frozenset[str]) -> MergedCandidates:
    """Merge the model's tags, then the prompt's tags, each through normalize_item, into one candidate list.

    An empty name is skipped. A name in drop is left out as "category noun", a repeat as "repeat", and a new
    name past cap as "over the cap". A prompt tag the model also proposed turns that origin to "both".
    """
    index_of: dict[str, int] = {}
    origins: list[str] = []
    dropped: list[tuple[str, str]] = []

    for origin, items in (("model", proposed), ("prompt", prompt_tags)):
        for item in items:
            name = normalize_item(item)
            if not name:
                continue
            index = index_of.get(name)
            if name in drop:
                dropped.append((name, "category noun"))
            elif index is not None and origin == "prompt" and origins[index] == "model":
                origins[index] = "both"
            elif index is not None:
                dropped.append((name, "repeat"))
            elif len(index_of) >= cap:
                dropped.append((name, "over the cap"))
            else:
                index_of[name] = len(origins)
                origins.append(origin)
    return MergedCandidates(tuple(index_of), tuple(origins), tuple(dropped))


def passed_by_origin(candidates: Sequence[str], origins: Sequence[str], scores: Sequence[float], *,
                     thresholds: Mapping[str, float]) -> tuple[str, ...]:
    """Return the candidates whose score reaches their origin's threshold, in candidate order."""
    missing = sorted(set(origins).difference(thresholds))
    passed: list[str] = []

    if missing:
        raise ValueError(f"passed_by_origin has no threshold for origin {', '.join(map(repr, missing))}. "
                         f"Pass thresholds with a key for every origin.")
    for tag, origin, score in zip(candidates, origins, scores, strict=True):
        if score >= thresholds[origin]:
            passed.append(tag)
    return tuple(passed)


def presence_statement(tag: str, *, wording: str = PRESENCE_STATEMENT) -> str:
    """Return the wording with {TAG} replaced by the tag as given."""
    stripped_tag = tag.strip()

    if _TAG not in wording:
        raise ValueError(f"presence_statement got wording {wording!r}, which lacks {_TAG}. Put {_TAG} where "
                         f"the tag goes.")
    if not stripped_tag:
        raise ValueError(f"A tag needs at least one word, got {tag!r}.")
    return wording.replace(_TAG, stripped_tag)


class PresenceThresholds(NamedTuple):
    """The scores a presence statement must reach to keep its tag.

    `proposed` applies to a tag the model listed, and `declared` to a tag it did not list, such as a prompt's thing.
    """

    proposed: float
    declared: float


# On ContextAnchoredTileRefine's recorded fp8 tile runs this kept 12.32 correct and 0.54 wrong or vague tags per tile,
# precision 0.958 (tests-AB/ab_tags_paired.py).
BROAD_THRESHOLDS = PresenceThresholds(proposed=0.999, declared=0.9999)
# On ContextAnchoredTileRefine's recorded fp8 tile runs this kept 7.33 correct and 0.17 wrong or vague tags per tile,
# precision 0.977 (tests-AB/ab_tags_paired.py).
STRICT_THRESHOLDS = PresenceThresholds(proposed=0.99998, declared=0.99998)
