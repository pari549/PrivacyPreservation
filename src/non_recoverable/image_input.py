"""General image input, normalization, padding, and block feature utilities.

Images are kept as grayscale (H, W) or RGB (H, W, 3) arrays. Pixel values are
normalized to float32 [0, 1]. To ensure no source pixels are discarded, images
whose dimensions are not divisible by the block size are padded on the bottom
and right using edge replication. Block operations use this padded canvas;
``restore_original_dimensions`` crops their output back to the original extent.
Padding and normalization details are recorded in ``PreparedImage.metadata``.

Preprocessing only adapts image representation and geometry; it does not provide
privacy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.image_processing.block_splitter import validate_block_coverage
from src.image_processing.entropy import block_entropy
from src.image_processing.gradients import edge_density, gradient_maps, grayscale
from src.image_processing.roi_detection import roi_score
from src.image_processing.variance import block_statistics

DEFAULT_MAX_IMAGE_PIXELS = 25_000_000
DEFAULT_BACKGROUND_COLOR = (255, 255, 255)


class ImageInputError(ValueError):
    """Raised when an input image cannot be decoded or safely prepared."""


class ImageSizeLimitError(ImageInputError):
    """Raised when the configured maximum image pixel count is exceeded."""


@dataclass(frozen=True)
class PreparedImage:
    """Normalized image, edge-padded canvas, and reproducibility metadata."""

    image: np.ndarray
    padded_image: np.ndarray
    metadata: dict[str, object]


def _validate_positive_limit(max_image_pixels: int) -> None:
    if isinstance(max_image_pixels, bool) or not isinstance(max_image_pixels, int) or max_image_pixels <= 0:
        raise ValueError("max_image_pixels must be a positive integer")


def _validate_block_size(block_size: int) -> None:
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")


def _validate_image_array(
    image: np.ndarray, max_image_pixels: int
) -> tuple[np.ndarray, int]:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    array = image
    if array.ndim == 2:
        channels = 1
    elif array.ndim == 3 and array.shape[2] in (1, 3):
        channels = int(array.shape[2])
        if channels == 1:
            array = array[..., 0]
    else:
        raise ValueError("Expected a grayscale (H, W) or RGB (H, W, 3) image")

    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("Image dimensions must be positive")
    pixel_count = int(array.shape[0]) * int(array.shape[1])
    if pixel_count > max_image_pixels:
        raise ImageSizeLimitError(
            f"Image has {pixel_count:,} pixels, exceeding max_image_pixels={max_image_pixels:,}"
        )
    if array.dtype == np.bool_ or not (
        np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError("Image dtype must be a real integer or floating-point type")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("Image pixels must all be finite")
    return array, channels


def _resolve_value_range(
    image: np.ndarray, value_range: tuple[float, float] | None
) -> tuple[float, float]:
    if value_range is not None:
        if len(value_range) != 2:
            raise ValueError("value_range must contain exactly (minimum, maximum)")
        low, high = float(value_range[0]), float(value_range[1])
    elif np.issubdtype(image.dtype, np.unsignedinteger):
        limits = np.iinfo(image.dtype)
        low, high = 0.0, float(limits.max)
    elif np.issubdtype(image.dtype, np.signedinteger):
        raise ValueError("Signed integer images require an explicit value_range")
    else:
        minimum, maximum = float(np.min(image)), float(np.max(image))
        if minimum >= 0.0 and maximum <= 1.0:
            low, high = 0.0, 1.0
        elif minimum >= 0.0 and maximum <= 255.0:
            low, high = 0.0, 255.0
        else:
            raise ValueError(
                "Floating-point images outside [0, 1] or [0, 255] require value_range"
            )

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("value_range must be finite and increasing")
    if np.issubdtype(image.dtype, np.floating) and not np.all(np.isfinite(image)):
        raise ValueError("Image pixels must all be finite")
    actual_min, actual_max = float(np.min(image)), float(np.max(image))
    if actual_min < low or actual_max > high:
        raise ValueError(
            f"Image pixel range [{actual_min}, {actual_max}] falls outside value_range [{low}, {high}]"
        )
    return low, high


def prepare_image(
    image: np.ndarray,
    block_size: int = 4,
    value_range: tuple[float, float] | None = None,
    *,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    source_format: str | None = None,
    source_mode: str | None = None,
    conversion: str | None = None,
    alpha_policy: str = "not applicable to array input",
    background_color: tuple[int, int, int] = DEFAULT_BACKGROUND_COLOR,
    exif_orientation_applied: bool = False,
    exif_orientation: int | None = None,
    encoded_width: int | None = None,
    encoded_height: int | None = None,
) -> PreparedImage:
    """Validate, normalize, and edge-pad one grayscale or RGB image.

    Unsigned integer inputs use their dtype range by default. Floating-point
    inputs in [0, 1] or [0, 255] use that range. Other ranges, including signed
    integers, must be specified explicitly as ``value_range=(low, high)``.
    """
    _validate_block_size(block_size)
    _validate_positive_limit(max_image_pixels)
    background_color = _validate_background_color(background_color)
    array, channels = _validate_image_array(image, max_image_pixels)
    low, high = _resolve_value_range(array, value_range)
    normalized = ((array.astype(np.float64) - low) / (high - low)).astype(np.float32)
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0) or np.any(normalized > 1):
        raise ValueError("Normalization produced pixels outside finite [0, 1]")

    height, width = map(int, normalized.shape[:2])
    pad_bottom = (-height) % block_size
    pad_right = (-width) % block_size
    padding = ((0, pad_bottom), (0, pad_right))
    if normalized.ndim == 3:
        padding += ((0, 0),)
    padded = np.pad(normalized, padding, mode="edge")
    padded_height, padded_width = map(int, padded.shape[:2])
    validate_block_coverage(padded_height, padded_width, block_size)

    metadata: dict[str, object] = {
        "original_height": height,
        "original_width": width,
        "encoded_height": encoded_height,
        "encoded_width": encoded_width,
        "original_channels": channels,
        "input_dtype": str(array.dtype),
        "input_value_range": [low, high],
        "normalized_dtype": "float32",
        "normalized_value_range": [0.0, 1.0],
        "normalization": "(pixel - input_min) / (input_max - input_min)",
        "block_size": block_size,
        "padded_height": padded_height,
        "padded_width": padded_width,
        "pad_bottom": pad_bottom,
        "pad_right": pad_right,
        "edge_policy": "replicate bottom and right edge pixels",
        "source_format": source_format,
        "source_mode": source_mode,
        "conversion": conversion,
        "original_file_format": source_format,
        "original_file_mode": source_mode,
        "channel_conversion_policy": conversion or "preserve array channel layout",
        "alpha_handling_policy": alpha_policy,
        "background_color_rgb": list(background_color),
        "exif_orientation_policy": "apply EXIF transpose before channel conversion",
        "exif_orientation_applied": bool(exif_orientation_applied),
        "exif_orientation_value": exif_orientation,
        "max_image_pixels": max_image_pixels,
    }
    return PreparedImage(image=normalized, padded_image=padded, metadata=metadata)


def load_image(
    path: str | Path,
    block_size: int = 4,
    *,
    grayscale_image: bool | None = None,
    value_range: tuple[float, float] | None = None,
    background_color: tuple[int, int, int] = DEFAULT_BACKGROUND_COLOR,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> PreparedImage:
    """Load a Pillow-supported image file and prepare its pixels.

    EXIF orientation is applied before dimensions and channel conversion are
    finalized. Transparent pixels are alpha-composited over ``background_color``
    (RGB byte values, white by default) before optional grayscale conversion.
    Encoded dimensions are checked against ``max_image_pixels`` before image
    conversion or full pixel processing.
    """
    _validate_block_size(block_size)
    _validate_positive_limit(max_image_pixels)
    background_color = _validate_background_color(background_color)
    path = Path(path)
    try:
        with Image.open(path) as opened:
            source_format, source_mode = opened.format, opened.mode
            encoded_width, encoded_height = opened.size
            _validate_dimensions(encoded_width, encoded_height, max_image_pixels)
            orientation = opened.getexif().get(274, 1)
            has_transparency = "A" in opened.getbands() or "transparency" in opened.info
            oriented = ImageOps.exif_transpose(opened)
            width, height = oriented.size
            _validate_dimensions(width, height, max_image_pixels)
            grayscale_modes = {"1", "L", "LA", "I", "I;16", "I;16B", "I;16L", "F"}
            use_grayscale = source_mode in grayscale_modes if grayscale_image is None else grayscale_image
            if has_transparency:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (*background_color, 255))
                composited = Image.alpha_composite(background, rgba).convert("RGB")
                if use_grayscale:
                    pixels = np.asarray(composited.convert("L")).copy()
                    conversion = "alpha-composite over RGB background, then Pillow convert('L')"
                else:
                    pixels = np.asarray(composited).copy()
                    conversion = "alpha-composite over RGB background, output RGB"
                alpha_policy = "alpha-composite over configured opaque RGB background before channel conversion"
            else:
                alpha_policy = "source has no alpha or transparency; no compositing performed"
                preserve_16bit_gray = use_grayscale and oriented.mode in {"I;16", "I;16B", "I;16L"}
                if preserve_16bit_gray:
                    pixels = np.asarray(oriented).copy()
                    conversion = "preserve native 16-bit grayscale pixels"
                elif use_grayscale:
                    pixels = np.asarray(oriented.convert("L")).copy()
                    conversion = "Pillow convert('L')"
                else:
                    pixels = np.asarray(oriented.convert("RGB")).copy()
                    conversion = "Pillow convert('RGB')"
            orientation_applied = orientation in {2, 3, 4, 5, 6, 7, 8}
    except Image.DecompressionBombError as error:
        raise ImageSizeLimitError(f"Image exceeds Pillow's decompression safety limit: {path}") from error
    except (UnidentifiedImageError, OSError) as error:
        raise ImageInputError(f"Could not decode image file {path}: {error}") from error

    return prepare_image(
        pixels,
        block_size,
        value_range,
        max_image_pixels=max_image_pixels,
        source_format=source_format,
        source_mode=source_mode,
        conversion=conversion,
        alpha_policy=alpha_policy,
        background_color=background_color,
        exif_orientation_applied=orientation_applied,
        exif_orientation=orientation,
        encoded_width=encoded_width,
        encoded_height=encoded_height,
    )


def _validate_dimensions(width: int, height: int, max_image_pixels: int) -> None:
    if width <= 0 or height <= 0:
        raise ImageInputError("Decoded image dimensions must be positive")
    pixel_count = int(width) * int(height)
    if pixel_count > max_image_pixels:
        raise ImageSizeLimitError(
            f"Image has {pixel_count:,} pixels, exceeding max_image_pixels={max_image_pixels:,}"
        )


def _validate_background_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(color, (tuple, list)) or len(color) != 3:
        raise ValueError("background_color must be three integer RGB values")
    values = []
    for channel in color:
        if isinstance(channel, bool) or not isinstance(channel, (int, np.integer)) or not 0 <= channel <= 255:
            raise ValueError("background_color channels must be integers in [0, 255]")
        values.append(int(channel))
    return tuple(values)


def iter_image_blocks(
    prepared: PreparedImage,
) -> Iterator[tuple[int, int, int, np.ndarray]]:
    """Yield ``(block_id, top, left, block)`` over the complete padded canvas."""
    if not isinstance(prepared, PreparedImage):
        raise TypeError("prepared must be a PreparedImage")
    block_size = int(prepared.metadata["block_size"])
    block_id = 0
    for row in range(0, prepared.padded_image.shape[0], block_size):
        for column in range(0, prepared.padded_image.shape[1], block_size):
            yield (
                block_id,
                row,
                column,
                prepared.padded_image[row:row + block_size, column:column + block_size],
            )
            block_id += 1


def restore_original_dimensions(image: np.ndarray, metadata: dict[str, object]) -> np.ndarray:
    """Crop a padded-canvas result to the recorded original image dimensions."""
    array = np.asarray(image)
    original_height = metadata.get("original_height")
    original_width = metadata.get("original_width")
    padded_height = metadata.get("padded_height")
    padded_width = metadata.get("padded_width")
    dimensions = (original_height, original_width, padded_height, padded_width)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
        raise ValueError("Metadata must contain positive integer original and padded dimensions")
    if array.ndim not in (2, 3):
        raise ValueError("Image result must be a 2D grayscale or 3D channel array")
    if array.shape[0] != padded_height or array.shape[1] != padded_width:
        raise ValueError("Image dimensions do not match the recorded padded dimensions")
    if original_height > padded_height or original_width > padded_width:
        raise ValueError("Original dimensions cannot exceed padded dimensions")
    return array[:original_height, :original_width, ...] if array.ndim == 3 else array[:original_height, :original_width]


def extract_image_block_features(prepared: PreparedImage) -> list[dict[str, float | int]]:
    """Extract existing image-processing features from all padded blocks.

    The existing variance, gradient, entropy, edge-density, and ROI primitives
    are reused. Pixel statistics, entropy, and ROI foreground calculations use
    each block's valid source region only. The Sobel magnitude map is computed
    over the unpadded source image (using OpenCV's default border rule), then
    sampled only at valid pixels. Canny edge density is computed on the valid
    block crop, so OpenCV's normal local boundary handling applies at block
    edges. No feature value is calculated from edge-replicated padding.
    """
    if not isinstance(prepared, PreparedImage):
        raise TypeError("prepared must be a PreparedImage")
    image = prepared.image
    block_size = int(prepared.metadata["block_size"])
    gray = grayscale(image)
    _, _, magnitude = gradient_maps(image)
    magnitude = np.clip(magnitude, 0, 1)
    byte_gray = np.clip(np.rint(gray * 255), 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(byte_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    global_otsu = float(otsu / 255)
    original_height = int(prepared.metadata["original_height"])
    original_width = int(prepared.metadata["original_width"])
    records: list[dict[str, float | int]] = []

    for block_id, row, column, _ in iter_image_blocks(prepared):
        valid_height = max(0, min(block_size, original_height - row))
        valid_width = max(0, min(block_size, original_width - column))
        valid_block = image[row:row + valid_height, column:column + valid_width]
        valid_magnitude = magnitude[row:row + valid_height, column:column + valid_width]
        mean, variance, std = block_statistics(valid_block)
        gradient = float(valid_magnitude.mean())
        records.append({
            "block_id": block_id,
            "row": row,
            "column": column,
            "valid_height": valid_height,
            "valid_width": valid_width,
            "valid_pixel_count": valid_height * valid_width,
            "mean": mean,
            "variance": variance,
            "std": std,
            "gradient": gradient,
            "edge_density": edge_density(valid_block),
            "entropy": block_entropy(valid_block),
            "roi_score": roi_score(valid_block, gradient, global_otsu),
        })
    return records
