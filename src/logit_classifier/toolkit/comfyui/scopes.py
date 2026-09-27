"""Context managers that patch a loaded CLIP for the duration of a node's run, and restore it after."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ...backends._torch_window import _determinism
from ...backends.comfy_clip import _clip_parts, _resident


@contextmanager
def shared_vision_encode(clip: Any) -> Iterator[None]:
    """Run the vision tower once per picture tensor for the duration, shared by generate and classify.

    Each encode runs inside the determinism window, where classify always ran it, so its
    scores are unchanged. A transformer without core's preprocess_embed passes through.
    """
    import torch

    transformer = _clip_parts(clip)[1]
    original = getattr(transformer, "preprocess_embed", None)
    shadowed = "preprocess_embed" in vars(transformer)
    previous = vars(transformer).get("preprocess_embed")
    cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}

    if not callable(original):
        yield
        return

    def preprocess_embed(embed: Any, device: Any) -> Any:
        data = embed.get("data")
        if embed.get("type") != "image" or not torch.is_tensor(data):
            return original(embed, device=device)
        # The tensor is held beside its result, so its id cannot be reused while cached.
        key = (id(data), tuple(data.shape), data.dtype, str(device))
        if key not in cache:
            with _determinism():
                cache[key] = (data, original(embed, device=device))
        return cache[key][1]

    transformer.preprocess_embed = preprocess_embed
    try:
        yield
    finally:
        cache.clear()
        if shadowed:
            transformer.preprocess_embed = previous
        else:
            del transformer.preprocess_embed


@contextmanager
def skip_resident_loads(clip: Any) -> Iterator[None]:
    """Make clip.load_model return clip.patcher at once while the CLIP is resident.

    Core's load_models_gpu has no fast path for a loaded model and costs about 0.1 s per call
    (ContextAnchoredTileRefine tests-AB/tags-bench-log.md, section 4). The skip also skips the
    memory_required reservation core makes in CLIP.load_model.
    """
    original = clip.load_model
    shadowed = "load_model" in vars(clip)

    def load_model(*args: Any, **kwargs: Any) -> Any:
        if _resident(clip):
            return clip.patcher
        return original(*args, **kwargs)

    clip.load_model = load_model
    try:
        yield
    finally:
        if shadowed:
            clip.load_model = original
        else:
            del clip.load_model
