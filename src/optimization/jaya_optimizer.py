"""Standard Jaya optimizer with bounded candidate repair support."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
import math
import numbers
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


@dataclass(frozen=True)
class UtilityConstraint:
    """One configured utility threshold and its violation normalization scale.

    ``direction`` is ``"minimum"`` for metric >= threshold or ``"maximum"``
    for metric <= threshold. ``normalization_scale`` is a positive, experiment
    configured scale in the metric's units; it prevents incompatible raw
    metric magnitudes from dominating total constraint violation by accident.
    """

    metric_name: str
    direction: str
    threshold: float
    normalization_scale: float

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str) or not self.metric_name.strip():
            raise ValueError("metric_name must be a non-empty string")
        if self.direction not in ("minimum", "maximum"):
            raise ValueError("direction must be 'minimum' or 'maximum'")
        for name, value in (
            ("threshold", self.threshold),
            ("normalization_scale", self.normalization_scale),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                raise ValueError(f"{name} must be a finite real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite real number")
        if self.normalization_scale <= 0:
            raise ValueError("normalization_scale must be greater than zero")


@dataclass(frozen=True)
class CandidateEvaluation:
    """Result of evaluating one candidate transform configuration.

    Valid records require finite privacy fitness and finite named utility
    measurements. Failed attack/evaluation runs should return ``invalid()``;
    they receive no fitness credit and rank below every valid candidate.
    """

    privacy_fitness: float | None
    utility_metrics: dict[str, object] = field(default_factory=dict)
    valid: bool = True
    invalid_reason: str | None = None

    @classmethod
    def invalid(cls, reason: str) -> "CandidateEvaluation":
        """Create a well-formed invalid candidate with no privacy score."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("invalid candidate reason must be a non-empty string")
        return cls(
            privacy_fitness=None,
            utility_metrics={},
            valid=False,
            invalid_reason=reason,
        )


@dataclass(frozen=True)
class CandidateRank:
    """Feasibility tier and normalized violation used for candidate ordering."""

    tier: int
    feasible: bool
    total_normalized_violation: float | None
    privacy_fitness: float | None
    invalid_reason: str | None = None

    @property
    def sort_key(self) -> tuple[int, float, float]:
        if self.tier == 0:
            return (0, -math.inf, -math.inf)
        if self.tier == 1:
            assert self.total_normalized_violation is not None
            assert self.privacy_fitness is not None
            return (1, -self.total_normalized_violation, self.privacy_fitness)
        assert self.privacy_fitness is not None
        return (2, 0.0, self.privacy_fitness)

    def as_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "feasible": self.feasible,
            "total_normalized_violation": self.total_normalized_violation,
            "privacy_fitness": self.privacy_fitness,
            "invalid_reason": self.invalid_reason,
        }


def _validate_constraints(
    constraints: tuple[UtilityConstraint, ...] | list[UtilityConstraint],
) -> tuple[UtilityConstraint, ...]:
    if not isinstance(constraints, (tuple, list)):
        raise TypeError("constraints must be a sequence of UtilityConstraint values")
    configured = tuple(constraints)
    if any(not isinstance(item, UtilityConstraint) for item in configured):
        raise TypeError("constraints must contain only UtilityConstraint values")
    names = [item.metric_name for item in configured]
    if len(names) != len(set(names)):
        raise ValueError("utility constraint metric names must be unique")
    return configured


def rank_candidate(
    evaluation: CandidateEvaluation,
    constraints: tuple[UtilityConstraint, ...] | list[UtilityConstraint],
) -> CandidateRank:
    """Rank valid candidates lexicographically; invalid candidates are last.

    For a minimum constraint, normalized violation is
    ``max(0, (threshold - value) / normalization_scale)``. For a maximum
    constraint it is ``max(0, (value - threshold) / normalization_scale)``.
    Total violation is the sum over configured constraints. Scales must be
    chosen in experiment configuration and expressed in each metric's units.
    """
    configured = _validate_constraints(constraints)
    if not isinstance(evaluation, CandidateEvaluation):
        raise TypeError("evaluation must be a CandidateEvaluation")
    if not isinstance(evaluation.valid, (bool, np.bool_)):
        raise ValueError("evaluation.valid must be boolean")
    if not isinstance(evaluation.utility_metrics, dict):
        raise ValueError("utility_metrics must be a dictionary")

    if not evaluation.valid:
        if evaluation.privacy_fitness is not None:
            raise ValueError("invalid candidates must not contain privacy fitness")
        if not isinstance(evaluation.invalid_reason, str) or not evaluation.invalid_reason.strip():
            raise ValueError("invalid candidates require a non-empty invalid_reason")
        return CandidateRank(0, False, None, None, evaluation.invalid_reason)
    if evaluation.invalid_reason is not None:
        raise ValueError("valid candidates cannot contain invalid_reason")
    if isinstance(evaluation.privacy_fitness, (bool, np.bool_)) or not isinstance(
        evaluation.privacy_fitness, numbers.Real
    ):
        raise ValueError("valid candidate privacy_fitness must be a finite real number")
    fitness = float(evaluation.privacy_fitness)
    if not math.isfinite(fitness):
        raise ValueError("valid candidate privacy_fitness must be finite")

    checked_metrics: dict[str, float] = {}
    for name, value in evaluation.utility_metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("utility metric names must be non-empty strings")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"utility metric {name!r} must be a finite real number")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"utility metric {name!r} must be finite")
        checked_metrics[name] = numeric_value

    violations: list[float] = []
    for constraint in configured:
        if constraint.metric_name not in checked_metrics:
            raise ValueError(f"required utility metric {constraint.metric_name!r} is missing")
        value = checked_metrics[constraint.metric_name]
        if constraint.direction == "minimum":
            violation = max(0.0, (constraint.threshold - value) / constraint.normalization_scale)
        else:
            violation = max(0.0, (value - constraint.threshold) / constraint.normalization_scale)
        if not math.isfinite(violation):
            raise ValueError("normalized constraint violation is non-finite")
        violations.append(violation)
    total_violation = math.fsum(violations)
    if not math.isfinite(total_violation):
        raise ValueError("total normalized constraint violation is non-finite")
    feasible = total_violation == 0.0
    return CandidateRank(
        tier=2 if feasible else 1,
        feasible=feasible,
        total_normalized_violation=total_violation,
        privacy_fitness=fitness,
    )


def jaya_optimize_constrained(
    evaluator: Callable[[np.ndarray], CandidateEvaluation],
    dimensions: int,
    constraints: tuple[UtilityConstraint, ...] | list[UtilityConstraint],
    bounds: tuple[float, float] | tuple[np.ndarray, np.ndarray] = (0.0, 1.0),
    population_size: int = 20,
    iterations: int = 100,
    seed: int = 42,
    initial_population: np.ndarray | None = None,
    repair: Repair | None = None,
) -> dict[str, object]:
    """Run Jaya using feasibility-aware ranking without changing scalar callers.

    Evaluator exceptions and malformed/non-finite candidate results become
    explicit invalid candidates with no fitness. Ranking order is feasible
    first (then higher privacy fitness), valid infeasible second (then lower
    normalized violation, then fitness), and invalid last. Convergence history
    stores the best rank summary at initialization and after each iteration;
    it is not a scalar fitness series. If every candidate is invalid, the best
    fitness is ``None`` and the invalid reason is returned.
    """
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if isinstance(dimensions, bool) or not isinstance(dimensions, (int, np.integer)) or dimensions <= 0:
        raise ValueError("dimensions must be a positive integer")
    if isinstance(population_size, bool) or not isinstance(population_size, (int, np.integer)) or population_size < 2:
        raise ValueError("population_size must be an integer of at least 2")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    configured_constraints = _validate_constraints(constraints)
    dimensions = int(dimensions)
    population_size = int(population_size)
    iterations = int(iterations)

    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError("bounds must contain exactly lower and upper bounds")
    try:
        low = np.broadcast_to(np.asarray(bounds[0], dtype=np.float64), (dimensions,)).copy()
        high = np.broadcast_to(np.asarray(bounds[1], dtype=np.float64), (dimensions,)).copy()
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError("bounds must contain broadcastable lower and upper bounds") from error
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(low >= high):
        raise ValueError("bounds must be finite and each lower bound below its upper bound")

    rng = np.random.default_rng(int(seed))
    if initial_population is None:
        population = rng.uniform(low, high, (population_size, dimensions))
    else:
        try:
            population = np.asarray(initial_population, dtype=np.float64).copy()
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("initial_population must be a finite numeric matrix") from error
        if population.shape != (population_size, dimensions):
            raise ValueError("Initial population shape mismatch")
        if not np.all(np.isfinite(population)):
            raise ValueError("initial_population must contain only finite values")

    def constrain(candidate: np.ndarray) -> np.ndarray:
        bounded = np.clip(candidate, low, high)
        repaired = np.asarray(repair(bounded), dtype=np.float64) if repair else bounded
        if repaired.shape != (dimensions,) or not np.all(np.isfinite(repaired)):
            raise ValueError("repair must return a finite vector matching dimensions")
        return np.clip(repaired, low, high)

    population = np.asarray([constrain(row) for row in population], dtype=np.float64)

    def evaluate(candidate: np.ndarray) -> tuple[CandidateEvaluation, CandidateRank]:
        try:
            record = evaluator(candidate.copy())
            rank = rank_candidate(record, configured_constraints)
            return record, rank
        except Exception as error:
            reason = f"{type(error).__name__}: {str(error)[:300]}"
            invalid = CandidateEvaluation.invalid(reason)
            return invalid, rank_candidate(invalid, configured_constraints)

    evaluations_and_ranks = [evaluate(candidate) for candidate in population]

    def select_best() -> int:
        return max(range(population_size), key=lambda index: evaluations_and_ranks[index][1].sort_key)

    def select_worst() -> int:
        return min(range(population_size), key=lambda index: evaluations_and_ranks[index][1].sort_key)

    def history_record(iteration: int) -> dict[str, object]:
        best_index = select_best()
        rank = evaluations_and_ranks[best_index][1]
        return {"iteration": iteration, "best_rank": rank.as_dict()}

    history = [history_record(0)]
    best_history = [population[select_best()].copy()]
    for iteration in range(1, iterations + 1):
        best = population[select_best()].copy()
        worst = population[select_worst()].copy()
        for index in range(population_size):
            r1 = rng.random(dimensions)
            r2 = rng.random(dimensions)
            candidate = constrain(
                population[index]
                + r1 * (best - np.abs(population[index]))
                - r2 * (worst - np.abs(population[index]))
            )
            candidate_evaluation, candidate_rank = evaluate(candidate)
            if candidate_rank.sort_key > evaluations_and_ranks[index][1].sort_key:
                population[index] = candidate
                evaluations_and_ranks[index] = (candidate_evaluation, candidate_rank)
        history.append(history_record(iteration))
        best_history.append(population[select_best()].copy())

    best_index = select_best()
    worst_index = select_worst()
    best_evaluation, best_rank = evaluations_and_ranks[best_index]
    worst_evaluation, worst_rank = evaluations_and_ranks[worst_index]
    return {
        "best_solution": population[best_index].copy(),
        "best_fitness": best_evaluation.privacy_fitness,
        "best_evaluation": best_evaluation,
        "best_rank": best_rank.as_dict(),
        "worst_solution": population[worst_index].copy(),
        "worst_evaluation": worst_evaluation,
        "worst_rank": worst_rank.as_dict(),
        "iterations": iterations,
        "convergence_history": history,
        "best_solution_history": np.asarray(best_history),
    }
