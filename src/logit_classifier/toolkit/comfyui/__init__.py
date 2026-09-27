"""The tools for a ComfyUI node that holds a loaded CLIP.

Every module here imports comfy, server, torch and comfy_execution inside the functions that need them, so importing
this package in a plain Python process loads none of them.
"""

from __future__ import annotations

from ...backends.comfy_clip import PACKED_TOKEN_CEILING, ComfyClipBackend
from .classifier import comfy_classifier
from .generate import IMAGE_SLOT, chat, clip_generate, generate_text, stop_when
from .progress import routed_progress_bars, run_outcome, send_status, token_fill
from .scopes import shared_vision_encode, skip_resident_loads
from .tagger import (
    PromptTags,
    TagSettings,
    TagTrace,
    english_text,
    fit_picture,
    prompt_tags,
    tag_picture,
    thing_scores,
)

# The ComfyUI socket type a node pack declares for a loaded classifier. It lives
# here so the library and the packs cannot drift apart on the spelling.
COMFY_SOCKET_TYPE = "LOGIT_CLASSIFIER"

__all__ = [
    "COMFY_SOCKET_TYPE",
    "IMAGE_SLOT",
    "PACKED_TOKEN_CEILING",
    "ComfyClipBackend",
    "PromptTags",
    "TagSettings",
    "TagTrace",
    "chat",
    "clip_generate",
    "comfy_classifier",
    "english_text",
    "fit_picture",
    "generate_text",
    "prompt_tags",
    "routed_progress_bars",
    "run_outcome",
    "send_status",
    "shared_vision_encode",
    "skip_resident_loads",
    "stop_when",
    "tag_picture",
    "thing_scores",
    "token_fill",
]
