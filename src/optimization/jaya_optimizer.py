"""Standard Jaya optimizer with bounded candidate repair support."""
from __future__ import annotations
from collections.abc import Callable
import numpy as np

Objective = Callable[[np.ndarray], float]
Repair = Callable[[np.ndarray], np.ndarray]

def normalize_bounded_weights(values: np.ndarray, lower: float = .10, upper: float = .70) -> np.ndarray:
    """Project four weights onto the bounded simplex: sum=1, lower<=w<=upper."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (4,) or lower * 4 > 1 or upper * 4 < 1 or lower >= upper:
        raise ValueError("Infeasible bounded simplex or expected four weights")
    lo, hi = float(np.min(values) - upper), float(np.max(values) - lower)
    for _ in range(80):
        shift = (lo + hi) / 2
        projected = np.clip(values - shift, lower, upper)
        if projected.sum() > 1: lo = shift
        else: hi = shift
    projected = np.clip(values - (lo + hi) / 2, lower, upper)
    # Correct tiny floating point residual on a coordinate with available room.
    residual = 1.0 - float(projected.sum())
    for index in range(4):
        adjustment = np.clip(residual, lower - projected[index], upper - projected[index])
        projected[index] += adjustment
        residual -= adjustment
    return projected

def jaya_optimize(objective: Objective, dimensions: int,
                  bounds: tuple[float, float] | tuple[np.ndarray, np.ndarray] = (0.0, 1.0),
                  population_size: int = 20, iterations: int = 100, seed: int = 42,
                  initial_population: np.ndarray | None = None, repair: Repair | None = None) -> dict[str, object]:
    """Maximize an objective with standard Jaya updates and boundary handling."""
    if dimensions <= 0 or population_size < 2 or iterations < 1:
        raise ValueError("dimensions/iterations must be positive and population_size at least 2")
    low = np.broadcast_to(np.asarray(bounds[0], dtype=np.float64), (dimensions,))
    high = np.broadcast_to(np.asarray(bounds[1], dtype=np.float64), (dimensions,))
    if np.any(low >= high): raise ValueError("Each lower bound must be below its upper bound")
    rng = np.random.default_rng(seed)
    population = rng.uniform(low, high, (population_size, dimensions)) if initial_population is None else np.asarray(initial_population, dtype=np.float64).copy()
    if population.shape != (population_size, dimensions): raise ValueError("Initial population shape mismatch")
    def constrain(candidate: np.ndarray) -> np.ndarray:
        bounded = np.clip(candidate, low, high)
        return np.clip(repair(bounded), low, high) if repair else bounded
    population = np.asarray([constrain(row) for row in population])
    fitness = np.asarray([objective(candidate) for candidate in population], dtype=np.float64)
    history = [float(np.max(fitness))]
    best_history = [population[int(np.argmax(fitness))].copy()]
    for _ in range(iterations):
        best = population[int(np.argmax(fitness))].copy()
        worst = population[int(np.argmin(fitness))].copy()
        for i in range(population_size):
            r1, r2 = rng.random(dimensions), rng.random(dimensions)
            candidate = constrain(population[i] + r1 * (best - np.abs(population[i])) - r2 * (worst - np.abs(population[i])))
            candidate_fitness = float(objective(candidate))
            if candidate_fitness > fitness[i]: population[i], fitness[i] = candidate, candidate_fitness
        best_index = int(np.argmax(fitness))
        history.append(float(fitness[best_index])); best_history.append(population[best_index].copy())
    best_index = int(np.argmax(fitness)); worst_index = int(np.argmin(fitness))
    return {"best_solution": population[best_index].copy(), "best_fitness": float(fitness[best_index]),
            "worst_solution": population[worst_index].copy(), "worst_fitness": float(fitness[worst_index]),
            "iterations": iterations, "convergence_history": history, "best_solution_history": np.asarray(best_history)}
