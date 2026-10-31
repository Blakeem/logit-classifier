"""A free text reply about one image from a loaded CLIP's own generate, for a caption a diffusion model reads."""

from __future__ import annotations

from typing import Any

from ..captions import clean_caption, strip_thinking
from .generate import _tokenize_image, chat, clip_generate

# The last try for an image whose stop token fires at once (ContextAnchoredTileRefine's generate_caption).
_FALLBACK_INSTRUCTION = "Write one short sentence describing this image."


def describe_image(clip: Any, image: Any, instruction: str, *, max_tokens: int, node: str) -> str:
    """Return one greedy reply to instruction about image, with its reasoning turn cut off.

    A reply that clean_caption empties is asked again sampled, then greedy with a one sentence instruction, and a reply
    still empty raises RuntimeError. max_tokens covers the reasoning turn as well as the answer. A CLIP whose tokenizer
    does not read images raises VisionUnsupportedError.
    """
    tokens: Any = None
    text = ""

    if not callable(getattr(clip, "generate", None)) or not callable(getattr(clip, "decode", None)):
        raise RuntimeError(f"{node}: this CLIP cannot generate text. Load a Qwen3-VL text encoder with a Load CLIP "
                           f"type that keeps its vision tower, such as krea2.")
    tokens = _tokenize_image(clip, chat(instruction, image=True), image)
    text = _answer(clip, tokens, do_sample=False, max_length=max_tokens, repetition_penalty=1.05)
    if not clean_caption(text):
        text = _answer(clip, tokens, do_sample=True, max_length=max_tokens, temperature=0.7, top_k=64, top_p=0.95,
                       min_p=0.05, repetition_penalty=1.05, seed=42)
    if not clean_caption(text):
        tokens = _tokenize_image(clip, chat(_FALLBACK_INSTRUCTION, image=True), image)
        text = _answer(clip, tokens, do_sample=False, max_length=max_tokens, repetition_penalty=1.05)
    if not clean_caption(text):
        raise RuntimeError(f"{node}: the caption came back empty after every fallback.")
    return text


def _answer(clip: Any, tokens: Any, **generate_options: Any) -> str:
    return strip_thinking(clip.decode(clip_generate(clip, tokens, **generate_options)))
