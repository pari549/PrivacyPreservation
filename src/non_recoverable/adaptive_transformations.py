"""Per-block baseline transforms for normalized grayscale and RGB images.

Block parameters follow the Stage 3A row-major geometry: left to right within
each block row, then top to bottom. A ``PreparedImage`` supplies its recorded
block size; raw arrays use ``block_size=4`` unless overridden. Partial blocks
at the bottom and right edges are processed over their valid source pixels.

Adaptive smoothing filters each block with an expanded crop containing a halo
of at least the Gaussian kernel support, then copies only the block core. This
prevents artificial filter boundaries at tile edges; real image boundaries
use the same reflected boundary mode as the baseline filter. A discontinuity
in sigma between neighboring block cores can still produce a visible change
in smoothing across their shared edge. No blending is applied, so each core
retains its specified strength.

The combined adaptive operation applies adaptive quantization first, then
adaptive Gaussian smoothing using the same row-major block parameter mapping.
These operations do not guarantee privacy or irreversibility.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from src.non_recoverable.baseline_transformations import (
    MAX_QUANTIZATION_LEVELS,
    MAX_SMOOTHING_STRENGTH,
    MIN_QUANTIZATION_LEVELS,
    ImageInput,
    uniform_gaussian_smoothing,
    uniform_quantization,
)
from src.non_recoverable.image_input import PreparedImage


DEFAULT_BLOCK_SIZE = 4


def _image_and_block_size(
    image: ImageInput, block_size: int | None
) -> tuple[np.ndarray, int]:
    if isinstance(image, PreparedImage):
        array = image.image
        prepared_block_size = image.metadata.get("block_size")
        if (
            isinstance(prepared_block_size, bool)
            or not isinstance(prepared_block_size, (int, np.integer))
            or prepared_block_size <= 0
        ):
            raise ValueError("PreparedImage metadata has an invalid block_size")
        if block_size is not None:
            if (
                isinstance(block_size, (bool, np.bool_))
                or not isinstance(block_size, (int, np.integer))
                or block_size != prepared_block_size
            ):
                raise ValueError("block_size must be an integer matching PreparedImage geometry")
        resolved_size = int(prepared_block_size)
    elif isinstance(image, np.ndarray):
        array = image
        resolved_size = DEFAULT_BLOCK_SIZE if block_size is None else block_size
    else:
        raise TypeError("image must be a NumPy array or PreparedImage")

    if (
        isinstance(resolved_size, (bool, np.bool_))
        or not isinstance(resolved_size, (int, np.integer))
        or resolved_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")
    # Delegate normalized range, dtype, and dimensionality checks to the
    # existing baseline implementation without changing its behavior.
    validated = uniform_gaussian_smoothing(array, strength=0.0)
    return validated, int(resolved_size)


def _block_grid(height: int, width: int, block_size: int) -> tuple[int, int]:
    return math.ceil(height / block_size), math.ceil(width / block_size)


def _validate_block_values(
    values: Sequence[object] | np.ndarray,
    expected_count: int,
    *,
    name: str,
    minimum: float,
    maximum: float,
    integer: bool,
) -> tuple[int, ...] | tuple[float, ...]:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional sequence")
        sequence = values.tolist()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        sequence = list(values)
    else:
        raise TypeError(f"{name} must be a one-dimensional sequence")
    if len(sequence) != expected_count:
        raise ValueError(
            f"{name} must contain exactly {expected_count} values for this image"
        )

    validated: list[int | float] = []
    for index, value in enumerate(sequence):
        if integer:
            valid_type = isinstance(value, (int, np.integer)) and not isinstance(
                value, (bool, np.bool_)
            )
        else:
            valid_type = isinstance(value, (int, float, np.number)) and not isinstance(
                value, (bool, np.bool_)
            )
        if not valid_type or not np.isfinite(value) or not minimum <= value <= maximum:
            kind = "integer" if integer else "finite numeric"
            raise ValueError(
                f"{name}[{index}] must be a {kind} in [{minimum}, {maximum}]"
            )
        validated.append(int(value) if integer else float(value))
    return tuple(validated)  # type: ignore[return-value]


def _validated_quantization_levels(
    levels_per_block: Sequence[object] | np.ndarray,
    block_count: int,
) -> tuple[int, ...]:
    return _validate_block_values(
        levels_per_block,
        block_count,
        name="levels_per_block",
        minimum=MIN_QUANTIZATION_LEVELS,
        maximum=MAX_QUANTIZATION_LEVELS,
        integer=True,
    )  # type: ignore[return-value]


def _validated_smoothing_strengths(
    strengths_per_block: Sequence[object] | np.ndarray,
    block_count: int,
) -> tuple[float, ...]:
    return _validate_block_values(
        strengths_per_block,
        block_count,
        name="strengths_per_block",
        minimum=0.0,
        maximum=MAX_SMOOTHING_STRENGTH,
        integer=False,
    )  # type: ignore[return-value]


def adaptive_quantization(
    image: ImageInput,
    levels_per_block: Sequence[object] | np.ndarray,
    *,
    block_size: int | None = None,
) -> np.ndarray:
    """Quantize each row-major block with its own 2–65,536 level setting."""
    array, block_size = _image_and_block_size(image, block_size)
    block_rows, block_columns = _block_grid(*array.shape[:2], block_size)
    levels = _validated_quantization_levels(
        levels_per_block, block_rows * block_columns
    )
    result = np.empty_like(array)
    index = 0
    for row in range(block_rows):
        top = row * block_size
        bottom = min(top + block_size, array.shape[0])
        for column in range(block_columns):
            left = column * block_size
            right = min(left + block_size, array.shape[1])
            result[top:bottom, left:right, ...] = uniform_quantization(
                array[top:bottom, left:right, ...], levels[index]
            )
            index += 1
    return result


def adaptive_gaussian_smoothing(
    image: ImageInput,
    strengths_per_block: Sequence[object] | np.ndarray,
    *,
    block_size: int | None = None,
) -> np.ndarray:
    """Smooth each block core at its sigma using neighboring image context.

    For each block, the source crop expands by ``ceil(3*sigma)+2`` pixels on
    every side (clipped at image boundaries). The baseline Gaussian filter is
    run on that crop and only the block core is copied to the output. The halo
    exceeds OpenCV's default approximately-three-sigma kernel radius, avoiding
    tile-edge boundary effects inside the core. Sigma changes at a block edge
    can still create an output discontinuity, as documented in this module.
    """
    array, block_size = _image_and_block_size(image, block_size)
    block_rows, block_columns = _block_grid(*array.shape[:2], block_size)
    strengths = _validated_smoothing_strengths(
        strengths_per_block, block_rows * block_columns
    )
    result = np.empty_like(array)
    index = 0
    height, width = array.shape[:2]
    for row in range(block_rows):
        top = row * block_size
        bottom = min(top + block_size, height)
        for column in range(block_columns):
            left = column * block_size
            right = min(left + block_size, width)
            strength = strengths[index]
            halo = int(math.ceil(3.0 * strength)) + 2
            crop_top = max(0, top - halo)
            crop_bottom = min(height, bottom + halo)
            crop_left = max(0, left - halo)
            crop_right = min(width, right + halo)
            crop = array[crop_top:crop_bottom, crop_left:crop_right, ...]
            smoothed = uniform_gaussian_smoothing(crop, strength)
            result[top:bottom, left:right, ...] = smoothed[
                top - crop_top:bottom - crop_top,
                left - crop_left:right - crop_left,
                ...,
            ]
            index += 1
    return result


def adaptive_combined_transformation(
    image: ImageInput,
    levels_per_block: Sequence[object] | np.ndarray,
    strengths_per_block: Sequence[object] | np.ndarray,
    *,
    block_size: int | None = None,
) -> np.ndarray:
    """Apply adaptive block quantization first, then adaptive block smoothing."""
    array, resolved_block_size = _image_and_block_size(image, block_size)
    block_rows, block_columns = _block_grid(*array.shape[:2], resolved_block_size)
    block_count = block_rows * block_columns
    levels = _validated_quantization_levels(levels_per_block, block_count)
    strengths = _validated_smoothing_strengths(strengths_per_block, block_count)
    quantized = adaptive_quantization(
        array, levels, block_size=resolved_block_size
    )
    return adaptive_gaussian_smoothing(
        quantized, strengths, block_size=resolved_block_size
    )
