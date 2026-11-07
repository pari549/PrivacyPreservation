"""Deterministic baseline transforms for normalized grayscale and RGB images.

These operations change image values but do not provide a privacy guarantee or
claim irreversibility. Input arrays must be floating-point grayscale ``(H, W)``
or channel-last ``(H, W, 1|3)`` images with finite values in ``[0, 1]``. A
``PreparedImage`` is accepted and its unpadded normalized ``image`` is used.
Outputs preserve the input dimensions, channel layout, and dtype.

Uniform quantization uses ``round-half-to-even`` on equally spaced levels
including both 0 and 1. Gaussian smoothing uses OpenCV's Gaussian kernel and
``BORDER_REFLECT_101`` boundary extension. The combined operation quantizes
first, then smooths; its output therefore need not remain on the quantization
grid.
"""
from __future__ import annotations

from typing import TypeAlias

import cv2
import numpy as np

from src.non_recoverable.image_input import PreparedImage


MIN_QUANTIZATION_LEVELS = 2
MAX_QUANTIZATION_LEVELS = 65_536
MAX_SMOOTHING_STRENGTH = 100.0
ImageInput: TypeAlias = np.ndarray | PreparedImage


def _validate_image(image: ImageInput) -> np.ndarray:
    """Return a validated normalized floating image without changing dtype."""
    if isinstance(image, PreparedImage):
        array = image.image
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError("image must be a NumPy array or PreparedImage")

    if array.ndim == 2:
        pass
    elif array.ndim == 3 and array.shape[2] in (1, 3):
        pass
    else:
        raise ValueError("Expected grayscale (H, W) or channel-last (H, W, 1|3) image")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("Image dimensions must be positive")
    if array.dtype not in (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Image dtype must be float16, float32, or float64")
    if not np.all(np.isfinite(array)):
        raise ValueError("Image pixels must be finite")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError("Image pixels must lie in [0, 1]")
    return array


def _validate_levels(levels: int) -> int:
    if (
        isinstance(levels, (bool, np.bool_))
        or not isinstance(levels, (int, np.integer))
        or not MIN_QUANTIZATION_LEVELS <= levels <= MAX_QUANTIZATION_LEVELS
    ):
        raise ValueError(
            f"levels must be an integer in [{MIN_QUANTIZATION_LEVELS}, "
            f"{MAX_QUANTIZATION_LEVELS}]"
        )
    return int(levels)


def _validate_smoothing_strength(strength: float) -> float:
    if (
        isinstance(strength, (bool, np.bool_))
        or not isinstance(strength, (int, float, np.number))
        or not np.isfinite(strength)
        or strength < 0
        or strength > MAX_SMOOTHING_STRENGTH
    ):
        raise ValueError(
            f"strength must be finite and in [0, {MAX_SMOOTHING_STRENGTH}]"
        )
    return float(strength)


def uniform_quantization(image: ImageInput, levels: int = 256) -> np.ndarray:
    """Map pixels to ``levels`` evenly spaced values over ``[0, 1]``.

    The supported level count is 2 through 65,536 inclusive. Formula:
    ``round_half_even(x * (levels - 1)) / (levels - 1)``. The returned array
    keeps the original shape and floating dtype.
    """
    array = _validate_image(image)
    levels = _validate_levels(levels)
    denominator = levels - 1
    quantized = np.rint(array.astype(np.float64) * denominator) / denominator
    return np.clip(quantized, 0.0, 1.0).astype(array.dtype, copy=False)


def uniform_gaussian_smoothing(
    image: ImageInput, strength: float = 1.0
) -> np.ndarray:
    """Apply Gaussian smoothing with ``strength`` as the kernel sigma in pixels.

    Sigma must be finite and in ``[0, 100]``. At zero, return an unchanged
    copy. Nonzero sigma uses OpenCV GaussianBlur with reflected boundaries;
    channels are filtered independently, and the original dtype is restored.
    """
    array = _validate_image(image)
    strength = _validate_smoothing_strength(strength)
    if strength == 0.0:
        return array.copy()

    working_dtype = np.float64 if array.dtype == np.float64 else np.float32
    working = array.astype(working_dtype, copy=False)
    if working.ndim == 2:
        smoothed = cv2.GaussianBlur(
            working,
            (0, 0),
            sigmaX=strength,
            sigmaY=strength,
            borderType=cv2.BORDER_REFLECT_101,
        )
    else:
        channels = [
            cv2.GaussianBlur(
                working[..., index],
                (0, 0),
                sigmaX=strength,
                sigmaY=strength,
                borderType=cv2.BORDER_REFLECT_101,
            )
            for index in range(working.shape[2])
        ]
        smoothed = np.stack(channels, axis=2)
    return np.clip(smoothed, 0.0, 1.0).astype(array.dtype, copy=False)


def combined_transformation(
    image: ImageInput,
    *,
    quantization_levels: int = 256,
    smoothing_strength: float = 1.0,
) -> np.ndarray:
    """Apply uniform quantization, then Gaussian smoothing.

    Both configurations and the input are validated before either operation
    runs. Output shape and dtype match the image input.
    """
    array = _validate_image(image)
    levels = _validate_levels(quantization_levels)
    strength = _validate_smoothing_strength(smoothing_strength)
    quantized = uniform_quantization(array, levels)
    return uniform_gaussian_smoothing(quantized, strength)
