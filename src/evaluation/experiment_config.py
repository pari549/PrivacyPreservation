"""Validated, serializable configuration for non-recoverable experiments.

This module records experiment choices only. It contains no measured results,
attack implementation, privacy objective, or experiment-wide defaults for
privacy/utility thresholds. Measured attack baselines are represented
separately by :class:`AttackBaselineMeasurement` and must pass explicit
validation before being accepted as privacy evidence.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import numbers
from types import MappingProxyType
from typing import Any

import numpy as np

from src.non_recoverable.baseline_transformations import (
    MAX_QUANTIZATION_LEVELS,
    MAX_SMOOTHING_STRENGTH,
    MIN_QUANTIZATION_LEVELS,
)
from src.non_recoverable.transformation_parameter_codec import (
    INITIAL_8BIT_QUANTIZATION_MENU,
    TransformationParameterCodec,
)
from src.optimization.jaya_optimizer import UtilityConstraint


_DIRECTIONS = {"higher_is_better", "lower_is_better"}
_CONSTRAINT_DIRECTIONS = {"minimum", "maximum"}
_APPLICABILITY = {"applicable", "not_applicable"}
_BASELINE_STATUSES = {"validated", "unvalidated"}
_TRANSFORM_METHODS = {
    "uniform_quantization",
    "uniform_gaussian_smoothing",
    "combined_transformation",
    "adaptive_quantization",
    "adaptive_gaussian_smoothing",
    "adaptive_combined_transformation",
}


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _freeze_json(value: object, path: str = "configuration") -> object:
    """Validate and recursively freeze JSON data; reject ambiguous/non-JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} dictionary keys must be strings")
            frozen[key] = _freeze_json(child, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child, f"{path}[{index}]") for index, child in enumerate(value))
    raise ValueError(f"{path} contains unsupported JSON value {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _require_keys(data: Mapping[str, Any], required: set[str], allowed: set[str], name: str) -> None:
    missing = required - data.keys()
    extra = data.keys() - allowed
    if missing:
        raise ValueError(f"{name} missing required fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"{name} has unsupported fields: {', '.join(sorted(extra))}")


@dataclass(frozen=True)
class DatasetConfiguration:
    name: str
    version: str
    image_category: str
    split_identifiers: tuple[str, ...]
    preprocessing: Mapping[str, object]
    dimension_policy: str
    dimensions: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "image_category"):
            _identifier(getattr(self, field_name), f"dataset.{field_name}")
        if not isinstance(self.split_identifiers, (tuple, list)) or not self.split_identifiers:
            raise ValueError("dataset.split_identifiers must be a non-empty sequence")
        splits = tuple(_identifier(item, "dataset split identifier") for item in self.split_identifiers)
        if len(set(splits)) != len(splits):
            raise ValueError("dataset split identifiers must be unique")
        if not isinstance(self.preprocessing, Mapping):
            raise ValueError("dataset.preprocessing must be a JSON object")
        frozen = _freeze_json(self.preprocessing, "dataset.preprocessing")
        if not isinstance(frozen, Mapping):
            raise ValueError("dataset.preprocessing must be a JSON object")
        policy = _identifier(self.dimension_policy, "dataset.dimension_policy")
        if policy not in ("arbitrary", "fixed"):
            raise ValueError("dataset.dimension_policy must be 'arbitrary' or 'fixed'")
        dimensions = self.dimensions
        if policy == "fixed":
            if not isinstance(dimensions, (tuple, list)) or len(dimensions) != 2:
                raise ValueError("fixed dimension policy requires [height, width]")
            dimensions = tuple(_positive_int(v, "dataset dimension") for v in dimensions)
        elif dimensions is not None:
            raise ValueError("dimensions must be omitted when dimension_policy is 'arbitrary'")
        object.__setattr__(self, "name", _identifier(self.name, "dataset.name"))
        object.__setattr__(self, "version", _identifier(self.version, "dataset.version"))
        object.__setattr__(self, "image_category", _identifier(self.image_category, "dataset.image_category"))
        object.__setattr__(self, "split_identifiers", splits)
        object.__setattr__(self, "preprocessing", frozen)
        object.__setattr__(self, "dimension_policy", policy)
        object.__setattr__(self, "dimensions", dimensions)


@dataclass(frozen=True)
class BaselineAcceptanceCriterion:
    direction: str
    threshold: float

    def __post_init__(self) -> None:
        if not isinstance(self.direction, str) or self.direction not in _CONSTRAINT_DIRECTIONS:
            raise ValueError("baseline criterion direction must be 'minimum' or 'maximum'")
        object.__setattr__(self, "threshold", _finite_number(self.threshold, "baseline threshold"))

    def accepts(self, value: float) -> bool:
        return value >= self.threshold if self.direction == "minimum" else value <= self.threshold


@dataclass(frozen=True)
class AttackConfiguration:
    name: str
    implementation_identifier: str
    metric_name: str
    metric_direction: str
    applicable_image_categories: tuple[str, ...]
    applicability_decision: str
    baseline_acceptance: BaselineAcceptanceCriterion
    evaluation_protocol_identifier: str
    model_policy: str

    def __post_init__(self) -> None:
        for field_name in ("name", "implementation_identifier", "metric_name", "evaluation_protocol_identifier"):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), f"attack.{field_name}"))
        if not isinstance(self.metric_direction, str) or self.metric_direction not in _DIRECTIONS:
            raise ValueError("attack.metric_direction must be higher_is_better or lower_is_better")
        if not isinstance(self.applicable_image_categories, (tuple, list)) or not self.applicable_image_categories:
            raise ValueError("attack.applicable_image_categories must explicitly list at least one category")
        categories = tuple(_identifier(c, "attack applicable image category") for c in self.applicable_image_categories)
        if len(set(categories)) != len(categories):
            raise ValueError("attack applicable image categories must be unique")
        if not isinstance(self.applicability_decision, str) or self.applicability_decision not in _APPLICABILITY:
            raise ValueError("attack.applicability_decision must be applicable or not_applicable")
        if not isinstance(self.baseline_acceptance, BaselineAcceptanceCriterion):
            raise ValueError("attack.baseline_acceptance must be configured explicitly")
        model_policy = _identifier(self.model_policy, "attack.model_policy")
        if model_policy not in ("fixed", "retrained"):
            raise ValueError("attack.model_policy must be 'fixed' or 'retrained'")
        object.__setattr__(self, "applicable_image_categories", categories)
        object.__setattr__(self, "model_policy", model_policy)


@dataclass(frozen=True)
class AttackBaselineMeasurement:
    """Measured original-image baseline, separate from configuration.

    ``validation_status='validated'`` is an explicit caller assertion that the
    stated evaluation protocol was completed. This class checks identifiers,
    finiteness, applicability and the configured acceptance criterion; it
    cannot independently verify that the attack was run correctly.
    """

    attack_name: str
    metric_name: str
    value: float
    validation_status: str
    evaluation_protocol_identifier: str

    def validate_for(self, attack: AttackConfiguration, image_category: str) -> float:
        if not isinstance(attack, AttackConfiguration):
            raise TypeError("attack must be an AttackConfiguration")
        if not isinstance(self.validation_status, str) or self.validation_status not in _BASELINE_STATUSES:
            raise ValueError("baseline validation_status must be validated or unvalidated")
        if self.validation_status != "validated":
            raise ValueError("unvalidated original attack baseline cannot be privacy evidence")
        if attack.applicability_decision != "applicable":
            raise ValueError("attack is not configured as applicable to this experiment")
        if image_category not in attack.applicable_image_categories:
            raise ValueError("attack is not declared applicable to this image category")
        if self.attack_name != attack.name or self.metric_name != attack.metric_name:
            raise ValueError("baseline attack or metric identifier does not match configuration")
        if self.evaluation_protocol_identifier != attack.evaluation_protocol_identifier:
            raise ValueError("baseline evaluation protocol does not match configuration")
        value = _finite_number(self.value, "original attack baseline")
        if value == 0.0:
            raise ValueError("zero original attack baseline cannot establish relative privacy improvement")
        if not attack.baseline_acceptance.accepts(value):
            raise ValueError("original attack baseline does not meet its configured acceptance criterion")
        return value


@dataclass(frozen=True)
class UtilityMetricConfiguration:
    name: str
    direction: str
    threshold: float
    normalization_scale: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "utility.name"))
        if not isinstance(self.direction, str) or self.direction not in _CONSTRAINT_DIRECTIONS:
            raise ValueError("utility.direction must be 'minimum' or 'maximum'")
        object.__setattr__(self, "threshold", _finite_number(self.threshold, "utility threshold"))
        scale = _finite_number(self.normalization_scale, "utility normalization_scale")
        if scale <= 0:
            raise ValueError("utility normalization_scale must be greater than zero")
        object.__setattr__(self, "normalization_scale", scale)

    def to_optimizer_constraint(self) -> UtilityConstraint:
        return UtilityConstraint(self.name, self.direction, self.threshold, self.normalization_scale)


@dataclass(frozen=True)
class TransformationConfiguration:
    method: str
    quantization_levels: tuple[int, ...]
    maximum_smoothing_strength: float
    experiment_bit_depth: int | None

    def __post_init__(self) -> None:
        method = _identifier(self.method, "transformation.method")
        if method not in _TRANSFORM_METHODS:
            raise ValueError("transformation.method must name a supported baseline or adaptive transform")
        if isinstance(self.quantization_levels, (str, bytes)) or not isinstance(self.quantization_levels, (tuple, list)):
            raise ValueError("transformation.quantization_levels must be a sequence")
        levels = tuple(self.quantization_levels)
        if len(levels) < 2 or any(
            isinstance(level, (bool, np.bool_))
            or not isinstance(level, (int, np.integer))
            or not MIN_QUANTIZATION_LEVELS <= level <= MAX_QUANTIZATION_LEVELS
            for level in levels
        ):
            raise ValueError("quantization levels must be valid integers supported by the transformation API")
        if tuple(sorted(set(int(level) for level in levels))) != tuple(int(level) for level in levels):
            raise ValueError("quantization levels must be strictly increasing and unique")
        maximum = _finite_number(self.maximum_smoothing_strength, "maximum_smoothing_strength")
        if maximum <= 0 or maximum > MAX_SMOOTHING_STRENGTH:
            raise ValueError(f"maximum_smoothing_strength must be in (0, {MAX_SMOOTHING_STRENGTH}]")
        if self.experiment_bit_depth is not None:
            object.__setattr__(self, "experiment_bit_depth", _positive_int(
                self.experiment_bit_depth, "transformation.experiment_bit_depth"
            ))
        if levels == INITIAL_8BIT_QUANTIZATION_MENU and self.experiment_bit_depth != 8:
            raise ValueError("the initial pilot quantization menu requires experiment_bit_depth=8")
        # Instantiate the production codec as the final compatibility check.
        TransformationParameterCodec(
            max_sigma=maximum,
            quantization_levels=levels,
            experiment_bit_depth=self.experiment_bit_depth,
        )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "quantization_levels", tuple(int(v) for v in levels))
        object.__setattr__(self, "maximum_smoothing_strength", maximum)

    def build_codec(self) -> TransformationParameterCodec:
        """Construct the Stage 3F-1 codec from this validated configuration."""
        return TransformationParameterCodec(
            max_sigma=self.maximum_smoothing_strength,
            quantization_levels=self.quantization_levels,
            experiment_bit_depth=self.experiment_bit_depth,
        )


@dataclass(frozen=True)
class JayaConfiguration:
    population_size: int
    iteration_count: int
    random_seed: int
    lower_bounds: tuple[float, float]
    upper_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        population = _positive_int(self.population_size, "jaya.population_size")
        if population < 2:
            raise ValueError("jaya.population_size must be at least 2")
        iterations = _positive_int(self.iteration_count, "jaya.iteration_count")
        if isinstance(self.random_seed, (bool, np.bool_)) or not isinstance(self.random_seed, (int, np.integer)) or self.random_seed < 0:
            raise ValueError("jaya.random_seed must be a non-negative integer")
        lower = self._bounds(self.lower_bounds, "lower_bounds")
        upper = self._bounds(self.upper_bounds, "upper_bounds")
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise ValueError("each Jaya lower bound must be below its upper bound")
        if lower != (0.0, 0.0) or upper != (1.0, 1.0):
            raise ValueError("the Stage 3F-1 codec requires both decision-vector bounds to be [0, 1]")
        object.__setattr__(self, "population_size", population)
        object.__setattr__(self, "iteration_count", iterations)
        object.__setattr__(self, "random_seed", int(self.random_seed))
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)

    @staticmethod
    def _bounds(values: Sequence[float], name: str) -> tuple[float, float]:
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError(f"jaya.{name} must contain two finite values")
        return tuple(_finite_number(value, f"jaya.{name}") for value in values)  # type: ignore[return-value]


@dataclass(frozen=True)
class ExperimentConfiguration:
    dataset: DatasetConfiguration
    attacks: tuple[AttackConfiguration, ...]
    utility_metrics: tuple[UtilityMetricConfiguration, ...]
    transformation: TransformationConfiguration
    jaya: JayaConfiguration
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, (bool, np.bool_)) or not isinstance(self.schema_version, int) or self.schema_version != 1:
            raise ValueError("unsupported experiment configuration schema_version")
        if not isinstance(self.dataset, DatasetConfiguration):
            raise TypeError("dataset must be a DatasetConfiguration")
        if not isinstance(self.transformation, TransformationConfiguration):
            raise TypeError("transformation must be a TransformationConfiguration")
        if not isinstance(self.jaya, JayaConfiguration):
            raise TypeError("jaya must be a JayaConfiguration")
        attacks = self._sequence(self.attacks, AttackConfiguration, "attacks")
        utilities = self._sequence(self.utility_metrics, UtilityMetricConfiguration, "utility_metrics")
        if not attacks:
            raise ValueError("at least one attack configuration is required")
        if not utilities:
            raise ValueError("at least one utility metric configuration is required")
        if len({attack.name for attack in attacks}) != len(attacks):
            raise ValueError("attack names must be unique within one experiment")
        if len({metric.name for metric in utilities}) != len(utilities):
            raise ValueError("utility metric names must be unique within one experiment")
        for attack in attacks:
            if self.dataset.image_category not in attack.applicable_image_categories:
                raise ValueError(
                    f"attack {attack.name!r} must explicitly list the configured dataset image category"
                )
        object.__setattr__(self, "attacks", attacks)
        object.__setattr__(self, "utility_metrics", utilities)

    @staticmethod
    def _sequence(values: object, item_type: type, name: str) -> tuple[Any, ...]:
        if not isinstance(values, (tuple, list)):
            raise ValueError(f"{name} must be a sequence")
        result = tuple(values)
        if any(not isinstance(value, item_type) for value in result):
            raise ValueError(f"{name} contains an invalid configuration record")
        return result

    def optimizer_constraints(self) -> tuple[UtilityConstraint, ...]:
        return tuple(item.to_optimizer_constraint() for item in self.utility_metrics)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset": {
                "name": self.dataset.name,
                "version": self.dataset.version,
                "image_category": self.dataset.image_category,
                "split_identifiers": list(self.dataset.split_identifiers),
                "preprocessing": _thaw_json(self.dataset.preprocessing),
                "dimension_policy": self.dataset.dimension_policy,
                "dimensions": list(self.dataset.dimensions) if self.dataset.dimensions else None,
            },
            "attacks": [
                {
                    "name": item.name,
                    "implementation_identifier": item.implementation_identifier,
                    "metric_name": item.metric_name,
                    "metric_direction": item.metric_direction,
                    "applicable_image_categories": list(item.applicable_image_categories),
                    "applicability_decision": item.applicability_decision,
                    "baseline_acceptance": {
                        "direction": item.baseline_acceptance.direction,
                        "threshold": item.baseline_acceptance.threshold,
                    },
                    "evaluation_protocol_identifier": item.evaluation_protocol_identifier,
                    "model_policy": item.model_policy,
                }
                for item in self.attacks
            ],
            "utility_metrics": [
                {"name": item.name, "direction": item.direction, "threshold": item.threshold,
                 "normalization_scale": item.normalization_scale}
                for item in self.utility_metrics
            ],
            "transformation": {
                "method": self.transformation.method,
                "quantization_levels": list(self.transformation.quantization_levels),
                "maximum_smoothing_strength": self.transformation.maximum_smoothing_strength,
                "experiment_bit_depth": self.transformation.experiment_bit_depth,
            },
            "jaya": {
                "population_size": self.jaya.population_size,
                "iteration_count": self.jaya.iteration_count,
                "random_seed": self.jaya.random_seed,
                "lower_bounds": list(self.jaya.lower_bounds),
                "upper_bounds": list(self.jaya.upper_bounds),
            },
        }

    def to_json(self) -> str:
        """Serialize reproducibly using sorted keys, compact separators and finite JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, serialized: str) -> "ExperimentConfiguration":
        if not isinstance(serialized, str):
            raise TypeError("serialized configuration must be a string")
        try:
            data = json.loads(serialized, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("serialized configuration is not valid finite JSON") from error
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: object) -> "ExperimentConfiguration":
        if not isinstance(data, Mapping):
            raise ValueError("experiment configuration must be a JSON object")
        _require_keys(data, {"schema_version", "dataset", "attacks", "utility_metrics", "transformation", "jaya"},
                      {"schema_version", "dataset", "attacks", "utility_metrics", "transformation", "jaya"}, "experiment")
        dataset_data = data["dataset"]
        if not isinstance(dataset_data, Mapping):
            raise ValueError("dataset must be an object")
        _require_keys(dataset_data, {"name", "version", "image_category", "split_identifiers", "preprocessing", "dimension_policy", "dimensions"},
                      {"name", "version", "image_category", "split_identifiers", "preprocessing", "dimension_policy", "dimensions"}, "dataset")
        dataset = DatasetConfiguration(**dataset_data)

        attacks_data = data["attacks"]
        if not isinstance(attacks_data, list):
            raise ValueError("attacks must be an array")
        attacks: list[AttackConfiguration] = []
        for item in attacks_data:
            if not isinstance(item, Mapping):
                raise ValueError("each attack must be an object")
            fields = {"name", "implementation_identifier", "metric_name", "metric_direction", "applicable_image_categories", "applicability_decision", "baseline_acceptance", "evaluation_protocol_identifier", "model_policy"}
            _require_keys(item, fields, fields, "attack")
            criterion = item["baseline_acceptance"]
            if not isinstance(criterion, Mapping):
                raise ValueError("attack.baseline_acceptance must be an object")
            _require_keys(criterion, {"direction", "threshold"}, {"direction", "threshold"}, "baseline_acceptance")
            attacks.append(AttackConfiguration(**{**item, "baseline_acceptance": BaselineAcceptanceCriterion(**criterion)}))

        utilities_data = data["utility_metrics"]
        if not isinstance(utilities_data, list):
            raise ValueError("utility_metrics must be an array")
        utilities = []
        for item in utilities_data:
            if not isinstance(item, Mapping):
                raise ValueError("each utility metric must be an object")
            fields = {"name", "direction", "threshold", "normalization_scale"}
            _require_keys(item, fields, fields, "utility metric")
            utilities.append(UtilityMetricConfiguration(**item))

        transform_data = data["transformation"]
        if not isinstance(transform_data, Mapping):
            raise ValueError("transformation must be an object")
        fields = {"method", "quantization_levels", "maximum_smoothing_strength", "experiment_bit_depth"}
        _require_keys(transform_data, fields, fields, "transformation")
        transformation = TransformationConfiguration(**transform_data)

        jaya_data = data["jaya"]
        if not isinstance(jaya_data, Mapping):
            raise ValueError("jaya must be an object")
        fields = {"population_size", "iteration_count", "random_seed", "lower_bounds", "upper_bounds"}
        _require_keys(jaya_data, fields, fields, "jaya")
        jaya = JayaConfiguration(**jaya_data)
        return cls(dataset, tuple(attacks), tuple(utilities), transformation, jaya, data["schema_version"])


__all__ = [
    "AttackBaselineMeasurement",
    "AttackConfiguration",
    "BaselineAcceptanceCriterion",
    "DatasetConfiguration",
    "ExperimentConfiguration",
    "JayaConfiguration",
    "TransformationConfiguration",
    "UtilityMetricConfiguration",
]
