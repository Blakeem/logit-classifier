"""One Classifier over the CLIP a ComfyUI workflow already loaded."""

from __future__ import annotations

from typing import Any

from ...backends.base import UnsupportedModelError, VisionUnsupportedError
from ...backends.comfy_clip import ComfyClipBackend
from ...classifier import Classifier
from ...config import Config


def comfy_classifier(
    clip: Any, *, node: str | None = None, needs_images: bool = True, batch_branches: bool = True
) -> Classifier:
    """One Classifier over the workflow's CLIP, with no running prior and no calibration file.

    Each error starts with the node's name when one is given, and names the Load CLIP fix.
    """
    prefix = f"{node}: " if node else ""
    backend: ComfyClipBackend | None = None

    try:
        backend = ComfyClipBackend(clip, batch_branches=batch_branches)
    except UnsupportedModelError as error:
        raise UnsupportedModelError(
            f"{prefix}this CLIP is not a Qwen3-VL text encoder, so the node cannot read it. Load a Qwen3-VL text "
            "encoder, such as qwen3vl_4b_fp8_scaled.safetensors, with a Load CLIP type that keeps its vision tower, "
            "such as krea2."
        ) from error
    if needs_images and not backend.sees_images:
        raise VisionUnsupportedError(
            f"{prefix}this CLIP was loaded without its vision tower, so the node cannot read images. Load the "
            "Qwen3-VL text encoder with a Load CLIP type that keeps it, such as krea2. The flux and flux2 types "
            "drop it."
        )
    # A node reuses one classifier across a batch, and the running prior would make an item's answer depend on the
    # items before it.
    return Classifier(Config(use_prior_debias=False, calibration_path=None), backend)
