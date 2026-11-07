"""Evaluation metrics and result structures for non-recoverable experiments.

Image metrics assume normalized floating images in ``[0, 1]``. Attack
comparisons preserve measurements and direction, and report a relative
performance reduction only when both measurements and the original baseline
are valid. These comparisons are evaluation measures, not privacy guarantees.
No attack model or experimental value is provided by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numbers
import time
from collections.abc import Callable, Mapping
from typing import Literal

import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim

from src.non_recoverable.image_input import PreparedImage


MetricDirection = Literal["higher_is_better", "lower_is_better"]
_DIRECTIONS = {"higher_is_better", "lower_is_better"}


def _image_array(image: np.ndarray | PreparedImage, name: str) -> np.ndarray:
    if isinstance(image, PreparedImage):
        array = image.image
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError(f"{name} must be a NumPy array or PreparedImage")
    if array.ndim != 2 and not (array.ndim == 3 and array.shape[2] == 3):
        raise ValueError(f"{name} must be grayscale (H, W) or RGB (H, W, 3)")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} dtype must be floating point with values in [0, 1]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} pixels must be finite")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} pixels must lie in [0, 1]")
    return array.astype(np.float64, copy=False)


def _compatible_images(
    reference: np.ndarray | PreparedImage,
    candidate: np.ndarray | PreparedImage,
) -> tuple[np.ndarray, np.ndarray]:
    original = _image_array(reference, "original image")
    transformed = _image_array(candidate, "transformed image")
    if original.shape != transformed.shape:
        raise ValueError("Original and transformed image dimensions/channels must match")
    return original, transformed


def mean_absolute_error(
    original: np.ndarray | PreparedImage,
    transformed: np.ndarray | PreparedImage,
) -> float:
    """Return mean absolute pixel error in ``[0, 1]``; 0 means identical."""
    reference, candidate = _compatible_images(original, transformed)
    return float(np.mean(np.abs(reference - candidate), dtype=np.float64))


def peak_signal_to_noise_ratio(
    original: np.ndarray | PreparedImage,
    transformed: np.ndarray | PreparedImage,
) -> float:
    """Return PSNR in dB for data range 1; identical images yield ``+inf``."""
    reference, candidate = _compatible_images(original, transformed)
    difference = reference - candidate
    mse = float(np.mean(difference * difference, dtype=np.float64))
    if mse == 0.0:
        return math.inf
    return float(10.0 * math.log10(1.0 / mse))


def structural_similarity(
    original: np.ndarray | PreparedImage,
    transformed: np.ndarray | PreparedImage,
    *,
    win_size: int | None = None,
) -> float:
    """Return SSIM with data range 1 and an odd local window.

    By default, use the largest odd window no greater than 7 or the smaller
    image dimension. The window must be at least 3 for non-identical images.
    Identical images smaller than 3 pixels in either dimension return 1.0,
    since their equality is exact and a local SSIM window cannot be formed.
    Grayscale values are compared directly; RGB uses scikit-image's channel
    axis handling. SSIM is conventionally in ``[-1, 1]``, where 1 is identical.
    """
    reference, candidate = _compatible_images(original, transformed)
    min_dimension = min(reference.shape[:2])
    if win_size is not None:
        if (
            isinstance(win_size, bool)
            or not isinstance(win_size, (int, np.integer))
            or win_size < 3
            or win_size % 2 == 0
            or win_size > min_dimension
        ):
            raise ValueError("win_size must be an odd integer from 3 to the image dimensions")
        resolved_window = int(win_size)
    else:
        resolved_window = min(7, min_dimension)
        if resolved_window % 2 == 0:
            resolved_window -= 1

    if np.array_equal(reference, candidate):
        return 1.0
    if resolved_window < 3:
        raise ValueError("Non-identical images need both dimensions at least 3 for SSIM")
    return float(
        skimage_ssim(
            reference,
            candidate,
            data_range=1.0,
            win_size=resolved_window,
            channel_axis=-1 if reference.ndim == 3 else None,
        )
    )


def calculate_utility_metrics(
    original: np.ndarray | PreparedImage,
    transformed: np.ndarray | PreparedImage,
    *,
    ssim_win_size: int | None = None,
) -> "UtilityMetrics":
    """Calculate PSNR, SSIM and MAE without mixing them with privacy results."""
    return UtilityMetrics(
        psnr_db=peak_signal_to_noise_ratio(original, transformed),
        ssim=structural_similarity(original, transformed, win_size=ssim_win_size),
        mae=mean_absolute_error(original, transformed),
    )


def _observation(value: object, label: str) -> "AttackObservation":
    if value is None:
        return AttackObservation(
            value=None,
            reported_value=None,
            valid=False,
            status="missing",
            reason=f"{label} measurement is missing",
        )
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        reported = value if isinstance(value, str) else repr(value)
        return AttackObservation(
            value=None,
            reported_value=reported,
            valid=False,
            status="invalid",
            reason=f"{label} measurement is not a real number",
        )
    number = float(value)
    if not math.isfinite(number):
        reported = "NaN" if math.isnan(number) else ("-Infinity" if number < 0 else "Infinity")
        return AttackObservation(
            value=None,
            reported_value=reported,
            valid=False,
            status="invalid",
            reason=f"{label} measurement is not finite",
        )
    return AttackObservation(
        value=number,
        reported_value=number,
        valid=True,
        status="valid",
        reason=None,
    )


@dataclass(frozen=True)
class AttackObservation:
    """One attack metric value and its validity state."""

    value: float | None
    reported_value: float | str | None
    valid: bool
    status: str
    reason: str | None


@dataclass(frozen=True)
class AttackComparison:
    """Separate original/transformed measurements and their directional delta."""

    attack_name: str
    metric_name: str
    metric_direction: MetricDirection
    applicable: bool | None
    applicability_note: str | None
    original: AttackObservation
    transformed: AttackObservation
    relative_performance_reduction: float | None
    status: str
    limitations: tuple[str, ...]


def compare_attack_performance(
    original_performance: object,
    transformed_performance: object,
    *,
    metric_direction: MetricDirection,
    applicable: bool | None,
    attack_name: str = "unspecified_attack",
    metric_name: str = "unspecified_metric",
) -> AttackComparison:
    """Compare an attack metric without assuming its direction or validity.

    For higher-is-better metrics, reduction is ``(original - transformed) /
    abs(original)``. For lower-is-better metrics, it is ``(transformed -
    original) / abs(original)``. Positive values indicate worse attack
    performance on transformed images; negative values indicate better attack
    performance. A missing, non-finite, zero, or otherwise invalid original
    baseline never produces a reduction.
    """
    if not isinstance(metric_direction, str) or metric_direction not in _DIRECTIONS:
        raise ValueError("metric_direction must be higher_is_better or lower_is_better")
    if applicable not in (True, False, None):
        raise ValueError("applicable must be True, False, or None")
    if not isinstance(attack_name, str) or not attack_name.strip():
        raise ValueError("attack_name must be a non-empty string")
    if not isinstance(metric_name, str) or not metric_name.strip():
        raise ValueError("metric_name must be a non-empty string")

    original = _observation(original_performance, "original attack")
    transformed = _observation(transformed_performance, "transformed attack")
    limitations: list[str] = [
        "Relative performance reduction is an evaluation measure, not a universal privacy guarantee."
    ]
    reduction: float | None = None
    if applicable is False:
        status = "not_applicable"
        limitations.append("The attack was marked inapplicable to this dataset.")
    elif applicable is None:
        status = "applicability_unknown"
        limitations.append("Attack applicability has not been established.")
    elif not original.valid:
        status = "invalid_original_measurement" if original.status == "invalid" else "missing_original_measurement"
        limitations.append(original.reason or "Original attack measurement is unavailable.")
    elif original.value == 0.0:
        status = "zero_original_baseline"
        limitations.append("Relative reduction is undefined for a zero original baseline.")
    elif not transformed.valid:
        status = "invalid_transformed_measurement" if transformed.status == "invalid" else "missing_transformed_measurement"
        limitations.append(transformed.reason or "Transformed attack measurement is unavailable.")
    else:
        assert original.value is not None and transformed.value is not None
        difference = (
            original.value - transformed.value
            if metric_direction == "higher_is_better"
            else transformed.value - original.value
        )
        candidate = difference / abs(original.value)
        if math.isfinite(candidate):
            reduction = float(candidate)
            status = "valid"
        else:
            status = "invalid_relative_reduction"
            limitations.append("Relative reduction overflowed and is undefined.")

    return AttackComparison(
        attack_name=attack_name,
        metric_name=metric_name,
        metric_direction=metric_direction,
        applicable=applicable,
        applicability_note=None,
        original=original,
        transformed=transformed,
        relative_performance_reduction=reduction,
        status=status,
        limitations=tuple(limitations),
    )


def evaluate_attack(
    attack_name: str,
    metric_name: str,
    original_performance: object,
    transformed_performance: object,
    *,
    metric_direction: MetricDirection,
    applicable: bool | None,
    applicability_note: str | None = None,
) -> AttackComparison:
    """Create a named attack evaluation record from externally obtained results."""
    if not isinstance(attack_name, str) or not attack_name.strip():
        raise ValueError("attack_name must be a non-empty string")
    if not isinstance(metric_name, str) or not metric_name.strip():
        raise ValueError("metric_name must be a non-empty string")
    if applicability_note is not None and not isinstance(applicability_note, str):
        raise ValueError("applicability_note must be a string or None")
    comparison = compare_attack_performance(
        original_performance,
        transformed_performance,
        metric_direction=metric_direction,
        applicable=applicable,
        attack_name=attack_name,
        metric_name=metric_name,
    )
    return AttackComparison(
        attack_name=attack_name,
        metric_name=metric_name,
        metric_direction=comparison.metric_direction,
        applicable=comparison.applicable,
        applicability_note=applicability_note,
        original=comparison.original,
        transformed=comparison.transformed,
        relative_performance_reduction=comparison.relative_performance_reduction,
        status=comparison.status,
        limitations=comparison.limitations,
    )


@dataclass(frozen=True)
class UtilityMetrics:
    """Image utility measurements; PSNR can be positive infinity if identical."""

    psnr_db: float | None = None
    ssim: float | None = None
    mae: float | None = None
    downstream_metrics: Mapping[str, float | None] | None = None
    runtime_seconds: float | None = None
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationRecord:
    """Experiment metadata with privacy and utility results kept separate."""

    dataset_name: str
    image_category: str
    transformation_name: str
    transformation_configuration: Mapping[str, object]
    privacy: AttackComparison | None = None
    utility: UtilityMetrics | None = None
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a plain nested dictionary, preserving undefined values as null."""
        return asdict(self)


def create_evaluation_record(
    *,
    dataset_name: str,
    image_category: str,
    transformation_name: str,
    transformation_configuration: Mapping[str, object],
    privacy: AttackComparison | None = None,
    utility: UtilityMetrics | None = None,
    limitations: tuple[str, ...] = (),
) -> EvaluationRecord:
    """Create a validated result envelope; no experiment values are inferred."""
    for name, value in (
        ("dataset_name", dataset_name),
        ("image_category", image_category),
        ("transformation_name", transformation_name),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if not isinstance(transformation_configuration, Mapping):
        raise TypeError("transformation_configuration must be a mapping")
    if privacy is not None and not isinstance(privacy, AttackComparison):
        raise TypeError("privacy must be an AttackComparison or None")
    if utility is not None and not isinstance(utility, UtilityMetrics):
        raise TypeError("utility must be UtilityMetrics or None")
    if any(not isinstance(note, str) for note in limitations):
        raise ValueError("limitations must contain only strings")
    return EvaluationRecord(
        dataset_name=dataset_name,
        image_category=image_category,
        transformation_name=transformation_name,
        transformation_configuration=dict(transformation_configuration),
        privacy=privacy,
        utility=utility,
        limitations=tuple(limitations),
    )


def measure_runtime(
    function: Callable[..., object],
    *args: object,
    repeats: int = 1,
    **kwargs: object,
) -> dict[str, float | int]:
    """Measure wall-clock execution time without persisting outputs."""
    if not callable(function):
        raise TypeError("function must be callable")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    start = time.perf_counter()
    for _ in range(repeats):
        function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return {
        "total_seconds": float(elapsed),
        "mean_seconds": float(elapsed / repeats),
        "repeats": repeats,
    }
