"""The text tools for a generated caption that a diffusion model reads as conditioning.

The caption is encoded as text, so every artifact left in it reaches the diffusion model verbatim.
"""

from __future__ import annotations

import re

from .tags import _THINK_BLOCK

__all__ = ["clean_caption", "strip_thinking"]

# The openings of the lines in which the model narrates its own answer instead of describing the image.
_META_LINE = ("wait,", "wait ", "here's a revised", "here is a revised", "revised version", "let me", "i need to",
              "actually,")
_THINK_TAG = re.compile(r"</?think>")


def strip_thinking(text: str) -> str:
    """Cut Qwen3's reasoning turn off the front of an answer.

    An unstripped think block would reach the diffusion model as several hundred tokens of the model talking to
    itself. An answer that holds only an unclosed reasoning turn comes back empty, which makes describe_image's
    fallback chain fire on it.
    """
    body = ""

    if "<think>" not in text:
        return text.strip()
    body = _THINK_BLOCK.sub("", text)
    # A close tag the sub left behind has no open tag of its own, so only the text after the last one is the answer.
    if "</think>" in body:
        body = body.rsplit("</think>", 1)[-1]
    return _THINK_TAG.sub("", body).strip()


def clean_caption(text: str) -> str:
    """Remove the model's own formatting lines from a caption, so the diffusion model never reads them as content.

    These three failures were observed:

    - heading: a label such as "**Answer:**" or "**Image as compact list:**", which describes nothing
    - meta: the model narrating, such as "Wait, I need to rephrase to fit under 50 words."
    - repetition: a market crop that looped six items three times inside one 84 word caption

    A repeat is matched by its normalized line, so "chicken, left" and "chicken, right" both survive and only an exact
    repeat is dropped. Content is never rewritten. Only whole artifact lines are cut.
    """
    kept: list[str] = []
    seen: set[str] = set()
    dropped = False
    line = ""
    low = ""
    key = ""

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower().lstrip("*_#-\u2022 ")
        if low.startswith(_META_LINE) or (not kept and line.startswith("**") and line.endswith(":**")):
            dropped = True
            continue
        key = " ".join(low.rstrip(".").split())
        if key in seen:
            dropped = True
            continue
        seen.add(key)
        kept.append(line)
    # A text with nothing to remove comes back byte for byte, since a rejoin would drop the markdown line break
    # spaces and retokenize a caption that was already fine.
    return text if not dropped else "\n".join(kept).strip()
