"""Caller-supplied-objective Jaya plumbing for non-recoverable features.

No research objective or image transformation is defined here. The caller
must provide an objective appropriate to the experiment and interpret the
result as optimization output, not as a privacy guarantee.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.non_recoverable.feature_matrix import ImageFeatureMatrix
from src.optimization.jaya_optimizer import jaya_optimize


FeatureObjective = Callable[[np.ndarray, np.ndarray], float]


def project_bounded_simplex(
    values: np.ndarray,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    """Project a vector onto ``sum(w)=1`` and ``lower <= w_i <= upper``."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("values must be a non-empty finite one-dimensional vector")
    if (
        isinstance(lower, (bool, np.bool_))
        or isinstance(upper, (bool, np.bool_))
        or not isinstance(lower, (int, float, np.number))
        or not isinstance(upper, (int, float, np.number))
        or not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower < 0
        or lower >= upper
    ):
        raise ValueError("bounds must be finite with 0 <= lower < upper")
    count = vector.size
    if lower * count > 1.0 or upper * count < 1.0:
        raise ValueError("bounds do not permit weights to sum to one")

    left = float(np.min(vector) - upper)
    right = float(np.max(vector) - lower)
    for _ in range(100):
        shift = (left + right) / 2.0
        projected = np.clip(vector - shift, lower, upper)
        if float(projected.sum()) > 1.0:
            left = shift
        else:
            right = shift

    projected = np.clip(vector - (left + right) / 2.0, lower, upper)
    residual = 1.0 - float(projected.sum())
    for index in range(count):
        adjustment = float(np.clip(residual, lower - projected[index], upper - projected[index]))
        projected[index] += adjustment
        residual -= adjustment
    if abs(residual) > 1e-10:
        raise ArithmeticError("Unable to satisfy bounded simplex constraints")
    return projected


def optimize_feature_weights(
    feature_matrix: ImageFeatureMatrix,
    objective: FeatureObjective,
    *,
    population_size: int = 20,
    iterations: int = 100,
    seed: int = 42,
    lower_weight: float = 0.0,
    upper_weight: float = 1.0,
) -> dict[str, object]:
    """Optimize normalized feature weights using a supplied scalar objective.

    ``objective(weights, matrix)`` must return a finite scalar to maximize.
    The Jaya implementation and the objective's scientific meaning are the
    caller's responsibility; this adapter performs no transformation.
    """
    if not isinstance(feature_matrix, ImageFeatureMatrix):
        raise TypeError("feature_matrix must be an ImageFeatureMatrix")
    if not callable(objective):
        raise TypeError("objective must be callable")
    matrix = np.asarray(feature_matrix.values, dtype=np.float64)
    if (
        not feature_matrix.feature_names
        or any(not isinstance(name, str) or not name for name in feature_matrix.feature_names)
        or len(set(feature_matrix.feature_names)) != len(feature_matrix.feature_names)
    ):
        raise ValueError("Feature names must be non-empty, unique strings")
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_matrix.feature_names):
        raise ValueError("Feature matrix shape does not match its feature names")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("Feature matrix must be non-empty and finite")
    if len(feature_matrix.blocks) != matrix.shape[0]:
        raise ValueError("Block metadata count must match feature matrix rows")

    for name, value, minimum in (
        ("population_size", population_size, 2),
        ("iterations", iterations, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    dimensions = matrix.shape[1]
    if (
        isinstance(lower_weight, (bool, np.bool_))
        or isinstance(upper_weight, (bool, np.bool_))
        or not isinstance(lower_weight, (int, float, np.number))
        or not isinstance(upper_weight, (int, float, np.number))
        or not np.isfinite(lower_weight)
        or not np.isfinite(upper_weight)
        or lower_weight < 0
        or lower_weight >= upper_weight
        or lower_weight * dimensions > 1.0
        or upper_weight * dimensions < 1.0
    ):
        raise ValueError("weight bounds cannot form a normalized simplex")

    def repair(candidate: np.ndarray) -> np.ndarray:
        return project_bounded_simplex(candidate, lower_weight, upper_weight)

    def checked_objective(weights: np.ndarray) -> float:
        score = objective(weights.copy(), matrix.copy())
        if isinstance(score, (bool, np.bool_)) or not np.isscalar(score):
            raise ValueError("objective must return a finite scalar")
        score = float(score)
        if not np.isfinite(score):
            raise ValueError("objective must return a finite scalar")
        return score

    result = jaya_optimize(
        checked_objective,
        dimensions,
        bounds=(lower_weight, upper_weight),
        population_size=population_size,
        iterations=iterations,
        seed=seed,
        repair=repair,
    )
    best = np.asarray(result["best_solution"], dtype=np.float64)
    return {
        "feature_names": list(feature_matrix.feature_names),
        "weights": {name: float(weight) for name, weight in zip(feature_matrix.feature_names, best)},
        "best_fitness": result["best_fitness"],
        "iterations": result["iterations"],
        "convergence_history": result["convergence_history"],
        "best_solution_history": result["best_solution_history"],
    }
