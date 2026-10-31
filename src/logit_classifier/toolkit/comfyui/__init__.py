"""The tools for a ComfyUI node that holds a loaded CLIP.

Every module here imports comfy, server and torch inside the functions that need them, so importing
this package in a plain Python process loads none of them.
"""

from __future__ import annotations

from ... import COMFY_SOCKET_TYPE
from ...backends.comfy_clip import PACKED_TOKEN_CEILING, ComfyClipBackend
from .classifier import comfy_classifier
from .describe import describe_image
from .generate import IMAGE_SLOT, chat, clip_generate, generate_text, stop_when
from .locate import POSITION_THRESHOLD, STRIP_PIXELS, StripBox, TagPlaces, locate_tags, render_placed, strip_boxes
from .progress import routed_progress_bars, run_outcome, send_status, token_fill
from .scopes import shared_vision_encode, skip_resident_loads
from .style import PromptStyleTrace, StyleSettings, prompt_style, style_line, style_scores
from .tagger import (
    ImageTagTrace,
    PromptTagTrace,
    TagSettings,
    fit_image,
    resample_image,
    tag_image,
    tag_prompt,
    thing_scores,
    to_english,
)

__all__ = [
    "COMFY_SOCKET_TYPE",
    "IMAGE_SLOT",
    "PACKED_TOKEN_CEILING",
    "POSITION_THRESHOLD",
    "STRIP_PIXELS",
    "ComfyClipBackend",
    "ImageTagTrace",
    "PromptStyleTrace",
    "PromptTagTrace",
    "StripBox",
    "StyleSettings",
    "TagPlaces",
    "TagSettings",
    "chat",
    "clip_generate",
    "comfy_classifier",
    "describe_image",
    "fit_image",
    "generate_text",
    "locate_tags",
    "prompt_style",
    "render_placed",
    "resample_image",
    "routed_progress_bars",
    "run_outcome",
    "send_status",
    "shared_vision_encode",
    "skip_resident_loads",
    "stop_when",
    "strip_boxes",
    "style_line",
    "style_scores",
    "tag_image",
    "tag_prompt",
    "thing_scores",
    "to_english",
    "token_fill",
]
