"""Application service for uploaded-image non-recoverable transformations.

The service validates upload bytes and explicit transformation settings, then
uses the project's existing universal loader, block feature extractor and
adaptive transform. It performs no Jaya optimization, persistence, key
handling, or recovery operation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
import math
import numbers
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image

from src.evaluation.non_recoverable_metrics import (
    mean_absolute_error,
    peak_signal_to_noise_ratio,
    structural_similarity,
)
from src.non_recoverable.adaptive_transformations import adaptive_combined_transformation
from src.non_recoverable.baseline_transformations import (
    MAX_QUANTIZATION_LEVELS,
    MAX_SMOOTHING_STRENGTH,
    MIN_QUANTIZATION_LEVELS,
)
from src.non_recoverable.feature_matrix import ImageFeatureMatrix, extract_feature_matrix
from src.non_recoverable.image_input import (
    DEFAULT_MAX_IMAGE_PIXELS,
    PreparedImage,
    load_image,
)


class NonRecoverableServiceError(RuntimeError):
    """Raised when a valid request fails during feature or transform processing."""


@dataclass(frozen=True)
class NonRecoverableImageResult:
    """Previews, measurements, and settings returned for one processed upload."""

    original_image: np.ndarray
    transformed_image: np.ndarray
    transformed_image_bytes: bytes
    mime_type: str
    image_format: str
    dimensions: tuple[int, int]
    channels: int
    preprocessing_metadata: dict[str, object]
    feature_summary: dict[str, object]
    transformation_settings: dict[str, object]
    metrics: dict[str, float | None]
    processing_time_seconds: float
    warnings: tuple[str, ...]


def _read_upload(image: bytes | bytearray | memoryview | str | Path) -> bytes:
    if isinstance(image, (bytes, bytearray, memoryview)):
        content = bytes(image)
    elif isinstance(image, (str, Path)):
        path = Path(image)
        if path.is_symlink() or not path.is_file():
            raise ValueError("image path must refer to a regular, non-symlink file")
        try:
            with path.open("rb") as handle:
                content = handle.read()
        except OSError as error:
            raise ValueError(f"Could not read input image: {error}") from error
    else:
        raise TypeError("image must be image bytes or a safe temporary file path")
    if not content:
        raise ValueError("uploaded image is empty")
    return content


def _sequence(value: object, name: str) -> list[object]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional sequence")
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise ValueError(f"{name} must be a scalar or one-dimensional sequence")


def _quantization_values(value: object, block_count: int) -> tuple[list[int], dict[str, object]]:
    if isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)):
        levels = int(value)
        if not MIN_QUANTIZATION_LEVELS <= levels <= MAX_QUANTIZATION_LEVELS:
            raise ValueError(
                f"quantization_levels must be in [{MIN_QUANTIZATION_LEVELS}, {MAX_QUANTIZATION_LEVELS}]"
            )
        return [levels] * block_count, {"mode": "global_uniform", "value": levels}

    values = _sequence(value, "quantization_levels")
    if len(values) != block_count:
        raise ValueError(f"quantization_levels must contain exactly {block_count} per-block values")
    levels_per_block: list[int] = []
    for index, item in enumerate(values):
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"quantization_levels[{index}] must be an integer")
        level = int(item)
        if not MIN_QUANTIZATION_LEVELS <= level <= MAX_QUANTIZATION_LEVELS:
            raise ValueError(
                f"quantization_levels[{index}] must be in [{MIN_QUANTIZATION_LEVELS}, {MAX_QUANTIZATION_LEVELS}]"
            )
        levels_per_block.append(level)
    return levels_per_block, {"mode": "per_block", "values": levels_per_block.copy()}


def _smoothing_values(value: object, block_count: int) -> tuple[list[float], dict[str, object]]:
    if isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)):
        strength = float(value)
        if not math.isfinite(strength) or not 0.0 <= strength <= MAX_SMOOTHING_STRENGTH:
            raise ValueError(f"smoothing_strength must be finite and in [0, {MAX_SMOOTHING_STRENGTH}]")
        return [strength] * block_count, {"mode": "global_uniform", "value": strength}

    values = _sequence(value, "smoothing_strength")
    if len(values) != block_count:
        raise ValueError(f"smoothing_strength must contain exactly {block_count} per-block values")
    strengths: list[float] = []
    for index, item in enumerate(values):
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, float, np.number)):
            raise ValueError(f"smoothing_strength[{index}] must be a finite number")
        strength = float(item)
        if not math.isfinite(strength) or not 0.0 <= strength <= MAX_SMOOTHING_STRENGTH:
            raise ValueError(
                f"smoothing_strength[{index}] must be finite and in [0, {MAX_SMOOTHING_STRENGTH}]"
            )
        strengths.append(strength)
    return strengths, {"mode": "per_block", "values": strengths.copy()}


def _encode_png(image: np.ndarray) -> bytes:
    pixels = np.clip(np.rint(np.asarray(image, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)
    if pixels.ndim == 2:
        pil_image = Image.fromarray(pixels)
    elif pixels.ndim == 3 and pixels.shape[2] == 3:
        pil_image = Image.fromarray(pixels)
    else:  # Defensive: loader currently returns only grayscale or RGB.
        raise ValueError("transformed result must be grayscale or RGB")
    output = BytesIO()
    pil_image.save(output, format="PNG")
    return output.getvalue()


def _summarize_features(matrix: ImageFeatureMatrix) -> dict[str, object]:
    means = np.mean(matrix.values, axis=0, dtype=np.float64)
    minima = np.min(matrix.values, axis=0)
    maxima = np.max(matrix.values, axis=0)
    height = int(matrix.image_metadata["original_height"])
    width = int(matrix.image_metadata["original_width"])
    block_size = int(matrix.image_metadata["block_size"])
    return {
        "block_count": int(matrix.values.shape[0]),
        "block_rows": math.ceil(height / block_size),
        "block_columns": math.ceil(width / block_size),
        "valid_source_pixel_count": int(sum(block["valid_pixel_count"] for block in matrix.blocks)),
        "feature_names": list(matrix.feature_names),
        "feature_means": {name: float(value) for name, value in zip(matrix.feature_names, means)},
        "feature_minima": {name: float(value) for name, value in zip(matrix.feature_names, minima)},
        "feature_maxima": {name: float(value) for name, value in zip(matrix.feature_names, maxima)},
    }


def process_non_recoverable_image(
    image: bytes | bytearray | memoryview | str | Path,
    *,
    quantization_levels: int | Sequence[int] | np.ndarray,
    smoothing_strength: float | Sequence[float] | np.ndarray,
    block_size: int = 4,
    grayscale_image: bool | None = None,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> NonRecoverableImageResult:
    """Process one uploaded image using explicit global or per-block settings.

    Global values are expanded to every row-major image block and are labeled
    ``global_uniform`` in the result. Per-block sequences must have one value
    per block. Feature extraction is descriptive only; it does not select
    transformation parameters. The transformed preview is normalized float32;
    ``transformed_image_bytes`` is an explicitly quantized PNG representation.
    """
    started = time.perf_counter()
    content = _read_upload(image)
    with tempfile.TemporaryDirectory(prefix="privacy-nonrecoverable-") as temp_dir:
        temporary_input = Path(temp_dir) / "uploaded-image"
        temporary_input.write_bytes(content)
        prepared: PreparedImage = load_image(
            temporary_input,
            block_size=block_size,
            grayscale_image=grayscale_image,
            max_image_pixels=max_image_pixels,
        )

    try:
        matrix = extract_feature_matrix(prepared)
        height, width = prepared.image.shape[:2]
        block_count = math.ceil(height / block_size) * math.ceil(width / block_size)
        if matrix.values.shape[0] != block_count:
            raise ValueError("feature rows do not match the loaded image block geometry")
        levels, level_record = _quantization_values(quantization_levels, block_count)
        strengths, strength_record = _smoothing_values(smoothing_strength, block_count)
        transformed = adaptive_combined_transformation(
            prepared,
            levels,
            strengths,
            block_size=block_size,
        )
    except (ValueError, TypeError):
        raise
    except Exception as error:
        raise NonRecoverableServiceError(
            f"Feature extraction or non-recoverable transformation failed: {type(error).__name__}: {error}"
        ) from error

    transformed = np.asarray(transformed)
    if (
        transformed.shape != prepared.image.shape
        or transformed.dtype != prepared.image.dtype
        or not np.all(np.isfinite(transformed))
        or np.any(transformed < 0.0)
        or np.any(transformed > 1.0)
    ):
        raise NonRecoverableServiceError(
            "Transformation output must preserve image shape and dtype and contain finite [0, 1] pixels"
        )

    warnings: list[str] = [
        "Feature extraction does not determine transformation settings or establish privacy effectiveness.",
        "The transformation is lossy; no inverse or recovery operation is provided.",
    ]
    if len(set(strengths)) > 1:
        warnings.append(
            "Different neighboring smoothing strengths can still produce visible block-boundary discontinuities."
        )

    metrics: dict[str, float | None] = {
        "mae": mean_absolute_error(prepared.image, transformed),
        "psnr_db": peak_signal_to_noise_ratio(prepared.image, transformed),
        "ssim": None,
    }
    try:
        metrics["ssim"] = structural_similarity(prepared.image, transformed)
    except ValueError as error:
        warnings.append(f"SSIM unavailable for this image: {error}")

    encoded = _encode_png(transformed)
    channels = 1 if prepared.image.ndim == 2 else int(prepared.image.shape[2])
    settings = {
        "method": "adaptive_quantization_then_adaptive_gaussian_smoothing",
        "block_size": block_size,
        "block_count": block_count,
        "quantization_levels": level_record,
        "smoothing_strength": strength_record,
    }
    return NonRecoverableImageResult(
        original_image=prepared.image.copy(),
        transformed_image=np.asarray(transformed).copy(),
        transformed_image_bytes=encoded,
        mime_type="image/png",
        image_format="PNG",
        dimensions=(int(height), int(width)),
        channels=channels,
        preprocessing_metadata=dict(prepared.metadata),
        feature_summary=_summarize_features(matrix),
        transformation_settings=settings,
        metrics=metrics,
        processing_time_seconds=time.perf_counter() - started,
        warnings=tuple(warnings),
    )
