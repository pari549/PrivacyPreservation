"""Map existing block priorities to bounded non-recoverable settings."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.non_recoverable.baseline_transformations import (
    MAX_QUANTIZATION_LEVELS,
    MAX_SMOOTHING_STRENGTH,
    MIN_QUANTIZATION_LEVELS,
)
from src.non_recoverable.feature_matrix import ImageFeatureMatrix
from src.optimization.jaya_optimizer import normalize_bounded_weights
from src.optimization.objective_functions import FEATURE_NAMES, block_priorities


# run_jaya seeds its initial population with equal weights for these four
# established priority features. The service uses this vector when no saved
# Jaya weights are supplied.
DEFAULT_FEATURE_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


@dataclass(frozen=True)
class BlockTransformationParameters:
    """Feature priorities and the resulting parameters in row-major order."""

    feature_names: tuple[str, ...]
    priorities: tuple[float, ...]
    scaled_priorities: tuple[float, ...]
    feature_weights: tuple[float, ...]
    levels_per_block: tuple[int, ...]
    strengths_per_block: tuple[float, ...]


def map_block_features_to_parameters(
    feature_matrix: ImageFeatureMatrix,
    *,
    quantization_levels: int | None = None,
    smoothing_strength: float | None = None,
    feature_weights: Sequence[float] | Mapping[str, object] | None = None,
) -> BlockTransformationParameters:
    """Adapt scalar setting ceilings using the project's weighted priorities.

    Priority uses the existing Jaya feature set and ``block_priorities``
    implementation. It is scaled with the same ``priority / (1 + max)`` rule
    used by the existing optimization objectives. Higher priority moves
    quantization toward its minimum and smoothing toward the supplied scalar;
    values stay inside the existing transformation bounds. A scalar of ``None``
    leaves that parameter unspecified and returns an empty tuple for it.
    """
    if not isinstance(feature_matrix, ImageFeatureMatrix):
        raise TypeError("feature_matrix must be an ImageFeatureMatrix")
    values = np.asarray(feature_matrix.values, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != len(feature_matrix.feature_names)
        or not np.all(np.isfinite(values))
        or len(feature_matrix.blocks) != values.shape[0]
    ):
        raise ValueError("feature_matrix must contain finite, aligned block rows")
    if len(set(feature_matrix.feature_names)) != len(feature_matrix.feature_names):
        raise ValueError("feature names must be unique")
    missing = [name for name in FEATURE_NAMES if name not in feature_matrix.feature_names]
    if missing:
        raise ValueError(f"feature_matrix is missing priority feature(s): {', '.join(missing)}")

    if quantization_levels is not None and (
        isinstance(quantization_levels, (bool, np.bool_))
        or not isinstance(quantization_levels, (int, np.integer))
        or not MIN_QUANTIZATION_LEVELS <= quantization_levels <= MAX_QUANTIZATION_LEVELS
    ):
        raise ValueError(
            f"quantization_levels must be an integer in [{MIN_QUANTIZATION_LEVELS}, "
            f"{MAX_QUANTIZATION_LEVELS}]"
        )
    if smoothing_strength is not None and (
        isinstance(smoothing_strength, (bool, np.bool_))
        or not isinstance(smoothing_strength, (int, float, np.number))
        or not np.isfinite(smoothing_strength)
        or not 0.0 <= smoothing_strength <= MAX_SMOOTHING_STRENGTH
    ):
        raise ValueError(
            f"smoothing_strength must be finite and in [0, {MAX_SMOOTHING_STRENGTH}]"
        )

    weights = _validated_feature_weights(feature_weights)
    records = [
        {name: float(values[row, feature_matrix.feature_names.index(name)]) for name in FEATURE_NAMES}
        for row in range(values.shape[0])
    ]
    priorities = np.asarray(block_priorities(records, weights), dtype=np.float64)
    if (
        priorities.shape != (values.shape[0],)
        or not np.all(np.isfinite(priorities))
        or np.any(priorities < 0.0)
    ):
        raise ValueError("block priority features must produce finite non-negative priorities")
    maximum_priority = float(np.max(priorities))
    scaled = priorities / (1.0 + maximum_priority)

    levels: tuple[int, ...] = ()
    if quantization_levels is not None:
        levels = tuple(
            int(value)
            for value in np.rint(
                MIN_QUANTIZATION_LEVELS
                + (int(quantization_levels) - MIN_QUANTIZATION_LEVELS) * (1.0 - scaled)
            )
        )

    strengths: tuple[float, ...] = ()
    if smoothing_strength is not None:
        strengths = tuple(float(value) for value in float(smoothing_strength) * scaled)

    return BlockTransformationParameters(
        feature_names=FEATURE_NAMES,
        priorities=tuple(float(value) for value in priorities),
        scaled_priorities=tuple(float(value) for value in scaled),
        feature_weights=tuple(float(value) for value in weights),
        levels_per_block=levels,
        strengths_per_block=strengths,
    )


def _validated_feature_weights(
    feature_weights: Sequence[float] | Mapping[str, object] | None,
) -> np.ndarray:
    """Resolve equal defaults or named Jaya output into the existing simplex."""
    if feature_weights is None:
        raw_values: object = DEFAULT_FEATURE_WEIGHTS
    elif isinstance(feature_weights, Mapping):
        # Accept either the complete run_jaya document or its optimized_weights
        # mapping, and the generic optimizer's `weights` mapping.
        if "optimized_weights" in feature_weights:
            raw_values = feature_weights["optimized_weights"]
        elif "weights" in feature_weights:
            raw_values = feature_weights["weights"]
        else:
            raw_values = feature_weights
        if not isinstance(raw_values, Mapping):
            raise ValueError("optimizer result must contain a named feature-weight mapping")
        canonical: dict[str, object] = {}
        for name, value in raw_values.items():
            canonical_name = "roi_score" if name == "roi" else name
            if canonical_name in canonical:
                raise ValueError("optimizer result contains duplicate ROI weight names")
            canonical[canonical_name] = value
        if set(canonical) != set(FEATURE_NAMES):
            raise ValueError(
                "optimizer weights must contain variance, gradient, entropy, and roi/roi_score"
            )
        raw_values = [canonical[name] for name in FEATURE_NAMES]
    elif isinstance(feature_weights, (str, bytes)):
        raise ValueError("feature_weights must be a four-value sequence or named mapping")
    else:
        raw_values = feature_weights

    if isinstance(raw_values, np.ndarray):
        if raw_values.ndim != 1:
            raise ValueError("feature_weights must be a one-dimensional sequence")
        items = raw_values.tolist()
    else:
        try:
            items = list(raw_values)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("feature_weights must be a four-value sequence") from error
    if len(items) != len(FEATURE_NAMES) or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.number))
        for value in items
    ):
        raise ValueError(f"feature_weights must contain {len(FEATURE_NAMES)} numeric values")
    try:
        values = np.asarray(items, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("optimizer feature weights must be finite numeric values") from error
    if values.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError(f"feature_weights must contain {len(FEATURE_NAMES)} finite values")
    try:
        return normalize_bounded_weights(values)
    except (TypeError, ValueError, ArithmeticError) as error:
        raise ValueError("feature_weights cannot be normalized by the existing Jaya bounds") from error
