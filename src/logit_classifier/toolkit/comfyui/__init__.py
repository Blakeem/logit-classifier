"""The tools for a ComfyUI node that holds a loaded CLIP.

Every module here imports comfy, server and torch inside the functions that need them, so importing
this package in a plain Python process loads none of them.
"""

from __future__ import annotations

from ... import COMFY_SOCKET_TYPE
from ...backends.comfy_clip import PACKED_TOKEN_CEILING, ComfyClipBackend
from .classifier import comfy_classifier
from .generate import IMAGE_SLOT, chat, clip_generate, generate_text, stop_when
from .progress import routed_progress_bars, run_outcome, send_status, token_fill
from .scopes import shared_vision_encode, skip_resident_loads
from .tagger import (
    ImageTagTrace,
    PromptTagTrace,
    TagSettings,
    fit_image,
    tag_image,
    tag_prompt,
    thing_scores,
    to_english,
)

__all__ = [
    "COMFY_SOCKET_TYPE",
    "IMAGE_SLOT",
    "PACKED_TOKEN_CEILING",
    "ComfyClipBackend",
    "ImageTagTrace",
    "PromptTagTrace",
    "TagSettings",
    "chat",
    "clip_generate",
    "comfy_classifier",
    "fit_image",
    "generate_text",
    "routed_progress_bars",
    "run_outcome",
    "send_status",
    "shared_vision_encode",
    "skip_resident_loads",
    "stop_when",
    "tag_image",
    "tag_prompt",
    "thing_scores",
    "to_english",
    "token_fill",
]
