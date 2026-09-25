"""Pulling an image out of the request state.

The state stays JSON, so an image arrives as a data URL, a bare base64 string or a
path. It is lifted out before the state is rendered as text, because base64 in the
prompt would cost thousands of tokens and say nothing.
"""

from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .deps import require
from .errors import LogitClassifierError

if TYPE_CHECKING:
    from PIL.Image import Image

IMAGE_KEYS = ("image", "screenshot")

# What a state that held only an image renders as. A host passing its image beside the
# state renders an empty state as this too, so both paths build the same prompt.
IMAGE_ONLY_STATE = "(see image)"


class ImageError(LogitClassifierError, ValueError):
    """The state named an image that could not be read."""


def _normalise(opened: Image) -> Image:
    pil_image = require("PIL.Image", "service")
    image_ops = require("PIL.ImageOps", "service")
    upright = image_ops.exif_transpose(opened)
    has_alpha = upright.mode in ("RGBA", "LA", "PA") or "transparency" in upright.info

    if not has_alpha:
        return cast("Image", upright.convert("RGB"))
    # White matches Qwen's reference qwen_vl_utils, which composites transparency onto white.
    background = pil_image.new("RGBA", upright.size, (255, 255, 255, 255))
    return cast("Image", pil_image.alpha_composite(background, upright.convert("RGBA")).convert("RGB"))


def _open_rgb(source: io.BytesIO | Path, what: str) -> Image:
    pil_image = require("PIL.Image", "service")
    unreadable = (OSError, ValueError, pil_image.DecompressionBombError)
    limit = pil_image.MAX_IMAGE_PIXELS

    try:
        opened = pil_image.open(source)
    except unreadable as error:
        raise ImageError(f"{what} is not a readable image: {error}") from error
    # Pillow raises only above twice its limit and decodes with a warning below that. The size
    # comes from the header, so this check runs before any pixel is decoded.
    width, height = opened.size
    if limit is not None and width * height > limit:
        raise ImageError(f"{what} is {width}x{height}, above the {limit} pixel limit")
    try:
        return _normalise(opened)
    except unreadable as error:
        raise ImageError(f"{what} is not a readable image: {error}") from error


def _from_base64(value: str) -> Image:
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ImageError(f"image is neither a readable path nor valid base64: {error}") from error
    return _open_rgb(io.BytesIO(raw), "the decoded base64")


def _decode(value: str, key: str, *, allow_paths: bool) -> Image:
    """Decode a data URL, a path to a file, or bare base64, by trying rather than guessing.

    Length cannot tell a path from base64, since a small PNG encodes shorter than many
    paths. Whether the file exists can.
    """
    path: Path | None = None

    if value.startswith("data:"):
        _, _, payload = value.partition(",")
        if not payload:
            raise ImageError("data URL carried no payload")
        return _from_base64(payload)
    # A path probe over HTTP would let a client read server files, and a UNC path would
    # send the service account's credentials to a host the client names.
    if not allow_paths:
        try:
            return _from_base64(value)
        except ImageError as error:
            raise ImageError(
                f"state.{key} is not a data URL or base64 image. The HTTP service accepts "
                f"only those two forms: {error}"
            ) from error
    try:
        candidate = Path(value)
        path = candidate if candidate.is_file() else None
    except (OSError, ValueError):
        path = None  # too long or illegal as a path, so it is not one
    if path is None:
        return _from_base64(value)
    # UnidentifiedImageError subclasses OSError, so reading the file stays outside
    # the probe above, which would report a real file as bad base64.
    return _open_rgb(path, str(path))


def image_key(state: Any) -> str | None:
    """Return the key a mapping state carries its image under, or None.

    Only a non-empty string is an image, so any other value stays state content.
    """
    if not isinstance(state, dict):
        return None

    for key in IMAGE_KEYS:
        value = state.get(key)
        if isinstance(value, str) and value:
            return key
    return None


def extract_image(state: Any, *, allow_paths: bool = True) -> tuple[Any, Image | None]:
    """Return the state with any image removed, and the image itself."""
    key = image_key(state)
    remainder: dict[str, Any] = {}

    if key is None:
        return state, None
    remainder = {k: v for k, v in state.items() if k != key}
    return (remainder or IMAGE_ONLY_STATE), _decode(state[key], key, allow_paths=allow_paths)
