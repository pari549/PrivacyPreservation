"""Small, testable validation and service-call helpers for the Streamlit UI."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.application.non_recoverable_service import process_non_recoverable_image
from src.non_recoverable.baseline_transformations import (
    MAX_QUANTIZATION_LEVELS,
    MAX_SMOOTHING_STRENGTH,
    MIN_QUANTIZATION_LEVELS,
)

MAX_UPLOAD_PIXELS = 25_000_000
ALLOWED_EXTENSIONS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}
GRAYSCALE_MODES = {"1", "L", "LA", "I", "I;16", "I;16B", "I;16L", "F"}


class UploadValidationError(ValueError):
    """An upload is empty, unsupported, malformed, or outside safe limits."""


class RecoverableConfigurationError(RuntimeError):
    """Server-side recovery storage or key configuration is incomplete."""


@dataclass(frozen=True)
class ImageUploadInfo:
    """Validated upload details and orientation-correct preview bytes."""

    width: int
    height: int
    color_mode: str
    image_format: str
    is_grayscale: bool
    preview_png: bytes


def inspect_image_upload(content: bytes, filename: str) -> ImageUploadInfo:
    """Validate PNG/JPEG bytes and report EXIF-oriented dimensions and mode."""
    if not isinstance(content, (bytes, bytearray, memoryview)) or not content:
        raise UploadValidationError("The uploaded file is empty or unreadable.")
    suffix = Path(filename or "").suffix.lower()
    expected_format = ALLOWED_EXTENSIONS.get(suffix)
    if expected_format is None:
        raise UploadValidationError("Upload a PNG, JPG, or JPEG image.")
    try:
        with Image.open(BytesIO(bytes(content)), formats=("PNG", "JPEG")) as image:
            if image.format != expected_format:
                raise UploadValidationError("The file extension does not match its image format.")
            encoded_width, encoded_height = image.size
            if encoded_width <= 0 or encoded_height <= 0:
                raise UploadValidationError("The image dimensions must be positive.")
            if encoded_width * encoded_height > MAX_UPLOAD_PIXELS:
                raise UploadValidationError(
                    f"The image exceeds the {MAX_UPLOAD_PIXELS:,}-pixel upload limit."
                )
            source_mode = image.mode
            image.load()
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
            if width <= 0 or height <= 0 or width * height > MAX_UPLOAD_PIXELS:
                raise UploadValidationError("The oriented image dimensions exceed the supported limit.")
            preview = BytesIO()
            oriented.save(preview, format="PNG")
            return ImageUploadInfo(
                width=int(width),
                height=int(height),
                color_mode=source_mode,
                image_format=expected_format,
                is_grayscale=_is_grayscale_source(oriented),
                preview_png=preview.getvalue(),
            )
    except UploadValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise UploadValidationError("The uploaded image could not be safely decoded.") from error


def _is_grayscale_source(image: Image.Image) -> bool:
    if image.mode in GRAYSCALE_MODES:
        return True
    if image.mode == "P":
        palette = image.getpalette("RGB")
        if palette:
            used_indices = np.unique(np.asarray(image))
            if any(int(index) * 3 + 2 >= len(palette) for index in used_indices):
                return False
            return all(
                palette[int(index) * 3]
                == palette[int(index) * 3 + 1]
                == palette[int(index) * 3 + 2]
                for index in used_indices
            )
    return False


def validate_recoverable_upload(info: ImageUploadInfo) -> None:
    """Require the existing recovery implementation's exact input geometry."""
    if info.is_grayscale:
        raise UploadValidationError("Recoverable mode requires RGB; grayscale images are not supported.")
    if (info.height, info.width) != (28, 28):
        raise UploadValidationError(
            "Recoverable mode accepts only 28×28 RGB images. Images are not resized."
        )


def recoverable_service_configuration(
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    """Return explicitly configured private storage and key paths.

    Configuration is read only from the server environment; the UI never asks
    users for filesystem paths and never reads or returns key bytes.
    """
    config = os.environ if environ is None else environ
    storage_value = config.get("PRIVACYPRESERVATION_PRIVATE_STORAGE", "").strip()
    key_value = config.get("PRIVACYPRESERVATION_KEY_PATH", "").strip()
    if not storage_value or not key_value:
        raise RecoverableConfigurationError(
            "Recoverable mode is not configured by the application administrator."
        )
    storage_input, key_input = Path(storage_value).expanduser(), Path(key_value).expanduser()
    if not storage_input.is_absolute() or not key_input.is_absolute():
        raise RecoverableConfigurationError("Recovery configuration must use absolute server-side paths.")
    if storage_input.exists() and storage_input.is_symlink():
        raise RecoverableConfigurationError("The configured private storage location is unavailable.")
    if key_input.is_symlink() or not key_input.is_file():
        raise RecoverableConfigurationError("The configured recovery key is unavailable.")
    storage_resolved, key_resolved = storage_input.resolve(), key_input.resolve()
    project_root = Path(__file__).resolve().parents[2]
    try:
        storage_resolved.relative_to(project_root)
    except ValueError:
        pass
    else:
        raise RecoverableConfigurationError(
            "Recoverable artifacts must use private storage outside the project directory."
        )
    return storage_resolved, key_resolved


def call_recoverable_transform(service, content: bytes, info: ImageUploadInfo):
    """Validate an upload and invoke the existing recoverable service once."""
    validate_recoverable_upload(info)
    return service.transform(content)


def call_recoverable_recovery(service, run_id: str):
    """Invoke recovery only for a service-issued opaque UUID run identifier."""
    import re

    if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("No valid active recovery run is selected.")
    return service.recover(run_id)


def call_non_recoverable_transform(
    content: bytes,
    *,
    quantization_levels: int,
    smoothing_strength: float,
    grayscale: bool,
):
    """Validate the UI's global settings and call the existing service API."""
    if isinstance(quantization_levels, bool) or not isinstance(quantization_levels, int):
        raise ValueError("Quantization levels must be an integer.")
    if not MIN_QUANTIZATION_LEVELS <= quantization_levels <= MAX_QUANTIZATION_LEVELS:
        raise ValueError("Quantization level is outside the supported range.")
    if (
        isinstance(smoothing_strength, bool)
        or not isinstance(smoothing_strength, (int, float))
    ):
        raise ValueError("Smoothing strength must be numeric.")
    if not 0 <= float(smoothing_strength) <= MAX_SMOOTHING_STRENGTH:
        raise ValueError("Smoothing strength is outside the supported range.")
    return process_non_recoverable_image(
        content,
        quantization_levels=quantization_levels,
        smoothing_strength=float(smoothing_strength),
        grayscale_image=grayscale,
    )


def safe_service_error(error: Exception, *, mode: str) -> str:
    """Return a user-safe error category without paths, metadata, or key data."""
    category = type(error).__name__
    if mode == "recoverable":
        return (
            "Recoverable processing could not complete. Check that the administrator-configured "
            f"key and private storage are available. (Error type: {category})"
        )
    return f"Image processing could not complete. Check the upload and settings. (Error type: {category})"
