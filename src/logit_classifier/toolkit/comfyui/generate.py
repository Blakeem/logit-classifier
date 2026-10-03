"""Greedy replies from a loaded CLIP's own generate, for text a ComfyUI pack reads, such as a list of candidate tags.

The classifier core never calls these tools, so nothing a classifier answers is generated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

from ...backends.base import VisionUnsupportedError
from ...backends.comfy_clip import _patched, _transformer
from ...prompt import inert
from ..tags import complete_tags, drop_unfinished_tag

IMAGE_SLOT = "<|vision_start|><|image_pad|><|vision_end|>"

_NO_IMAGE_TOKENIZER = (
    "this CLIP's tokenizer does not read images, so it cannot generate from one. Load a Qwen3-VL text encoder with a "
    "Load CLIP type that keeps its vision tower, such as krea2."
)


def chat(user: str, *, image: bool = False) -> str:
    """Return the Qwen chat text for one user turn, with the image slot first when image is true."""
    # The tokenizer reads a <|...|> spelling in plain text as a control token, so an unescaped prompt could
    # close the user turn.
    safe_user = inert(user)

    # A text starting with <|im_start|> skips core's template, whose empty think block made this model reply
    # empty or refuse.
    return f"<|im_start|>user\n{IMAGE_SLOT if image else ''}{safe_user}<|im_end|>\n<|im_start|>assistant\n"


def clip_generate(clip: Any, tokens: Any, **kwargs: Any) -> Any:
    """Run clip.generate with every keyword passed through, then core's cleanup of the CUDA graphs it captured.

    Core's Qwen3 graph decode keeps each layer's graph bound to the freed KV cache and asserts on the next generate in
    the same node (core issue #16441). The cleanup keeps graph decode on, which measured 100 to 16.5 ms per token
    (ContextAnchoredTileRefine tests-AB/tags-bench-log.md, section 3).
    """
    try:
        return clip.generate(tokens, **kwargs)
    finally:
        _drop_decode_graphs()


def _drop_decode_graphs() -> None:
    try:
        from comfy.model_prefetch import cleanup_prefetch_queues
    except ImportError:
        return
    cleanup_prefetch_queues()


@contextmanager
def stop_when(clip: Any, should_stop: Callable[[list[str]], bool]) -> Iterator[None]:
    """End core's greedy decode on a stop token once should_stop(the complete tags so far) is true.

    Core's generate loop takes no stop condition, so the token sample_token returns is replaced. Core copies that
    token into the sequence and breaks on it, in comfy/text_encoders/llama.py BaseGenerate.generate. A CLIP without
    core's sample_token or stop tokens passes through.
    """
    transformer = _transformer(clip)
    config = getattr(getattr(transformer, "model", None), "config", None)
    stop_tokens = getattr(config, "stop_tokens", None)
    history: list[int] = []

    if not callable(getattr(transformer, "sample_token", None)) or not stop_tokens:
        yield
        return
    stop_id = stop_tokens[0]
    original = transformer.sample_token

    def watch(*args: Any, **kwargs: Any) -> Any:
        token = original(*args, **kwargs)
        history.append(int(token.reshape(-1)[0]))
        if should_stop(complete_tags(clip.decode(history))):
            return token.new_full(token.shape, stop_id)
        return token

    with _patched(transformer, "sample_token", watch):
        yield


def generate_text(
    clip: Any,
    user: str,
    *,
    image: Any = None,
    max_tokens: int,
    should_stop: Callable[[list[str]], bool] | None = None,
    whole_tags: bool = True,
) -> str:
    """One greedy reply to user. With whole_tags, a reply that fills max_tokens is cut back to its last whole tag.

    A CLIP whose tokenizer does not read images raises VisionUnsupportedError for an image.
    """
    text = chat(user, image=image is not None)
    stopper: AbstractContextManager[None] = nullcontext() if should_stop is None else stop_when(clip, should_stop)
    tokens: Any = None
    ids: Any = None
    reply: str = ""

    tokens = clip.tokenize(text) if image is None else _tokenize_image(clip, text, image)
    with stopper:
        ids = clip_generate(clip, tokens, do_sample=False, max_length=max_tokens)
    reply = clip.decode(ids)
    # Core stops early only on a stop token, so a decode that fills the budget ends mid tag.
    if whole_tags and len(ids) >= max_tokens:
        return drop_unfinished_tag(reply)
    return reply


def _tokenize_image(clip: Any, text: str, image: Any) -> Any:
    """Tokenize text with one image, or raise VisionUnsupportedError when the tokenizer placed no image entry."""
    tokens: Any = None
    holds_image = False

    try:
        tokens = clip.tokenize(text, images=[image])
    except TypeError as error:
        raise VisionUnsupportedError(_NO_IMAGE_TOKENIZER) from error
    # A tokenizer that takes images=... through **kwargs and ignores it writes only text entries, and core's image
    # entry is a dict in the token's first slot.
    holds_image = any(isinstance(token[0], dict) for rows in tokens.values() for row in rows for token in row)
    if not holds_image:
        raise VisionUnsupportedError(_NO_IMAGE_TOKENIZER)
    return tokens
