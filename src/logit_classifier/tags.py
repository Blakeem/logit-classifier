"""The tag text tools, which moved to logit_classifier.toolkit.tags in 0.3.0.

This path stays for code built on 0.2.x, which PyPI published. It raises no warning, so that code keeps a quiet log.
"""

from __future__ import annotations

from .toolkit.tags import (
    MAX_CANDIDATES,
    MAX_REPEAT_BLOCK,
    MAX_TAG_CHARS,
    MAX_TAG_WORDS,
    clean_item,
    complete_tags,
    drop_subsets,
    drop_unfinished_tag,
    normalize_item,
    parse_candidates,
    repeated_block,
    split_prompt,
)

__all__ = [
    "MAX_CANDIDATES",
    "MAX_REPEAT_BLOCK",
    "MAX_TAG_CHARS",
    "MAX_TAG_WORDS",
    "clean_item",
    "complete_tags",
    "drop_subsets",
    "drop_unfinished_tag",
    "normalize_item",
    "parse_candidates",
    "repeated_block",
    "split_prompt",
]
