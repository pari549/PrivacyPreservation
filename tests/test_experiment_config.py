import unittest
from dataclasses import replace

from src.evaluation.experiment_config import (
    AttackBaselineMeasurement,
    AttackConfiguration,
    BaselineAcceptanceCriterion,
    DatasetConfiguration,
    ExperimentConfiguration,
    JayaConfiguration,
    TransformationConfiguration,
    UtilityMetricConfiguration,
)
from src.non_recoverable.transformation_parameter_codec import INITIAL_8BIT_QUANTIZATION_MENU


def make_configuration():
    """Synthetic configuration fixture; contains no experimental measurements."""
    dataset = DatasetConfiguration(
        name="test_dataset",
        version="fixture-v1",
        image_category="synthetic_rgb",
        split_identifiers=("test-only-split",),
        preprocessing={"normalization": {"method": "unit_interval", "range": [0, 1]}},
        dimension_policy="arbitrary",
    )
    attack = AttackConfiguration(
        name="placeholder_attack",
        implementation_identifier="tests.placeholder.v1",
        metric_name="placeholder_metric",
        metric_direction="higher_is_better",
        applicable_image_categories=("synthetic_rgb",),
        applicability_decision="applicable",
        baseline_acceptance=BaselineAcceptanceCriterion("minimum", 0.5),
        evaluation_protocol_identifier="tests.protocol.v1",
        model_policy="fixed",
    )
    utility = UtilityMetricConfiguration("ssim", "minimum", 0.8, 0.1)
    transform = TransformationConfiguration(
        "adaptive_combined_transformation",
        INITIAL_8BIT_QUANTIZATION_MENU,
        2.0,
        8,
    )
    jaya = JayaConfiguration(8, 5, 123, (0.0, 0.0), (1.0, 1.0))
    return ExperimentConfiguration(dataset, (attack,), (utility,), transform, jaya)


class ExperimentConfigurationTest(unittest.TestCase):
    def test_json_round_trip_is_deterministic_and_codec_compatible(self):
        config = make_configuration()
        serialized = config.to_json()
        restored = ExperimentConfiguration.from_json(serialized)
        self.assertEqual(restored.to_json(), serialized)
        self.assertEqual(restored.to_dict(), config.to_dict())
        codec = restored.transformation.build_codec()
        self.assertEqual(codec.quantization_levels, INITIAL_8BIT_QUANTIZATION_MENU)
        self.assertEqual(codec.max_sigma, 2.0)
        self.assertEqual(restored.optimizer_constraints()[0].metric_name, "ssim")

    def test_repeated_serialization_and_input_mapping_order_are_stable(self):
        first = make_configuration()
        reversed_preprocessing = {"normalization": {"range": [0, 1], "method": "unit_interval"}}
        data = first.to_dict()
        data["dataset"]["preprocessing"] = reversed_preprocessing
        second = ExperimentConfiguration.from_dict(data)
        self.assertEqual(first.to_json(), second.to_json())

    def test_required_fields_and_schema_are_strict(self):
        data = make_configuration().to_dict()
        del data["dataset"]["version"]
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            ExperimentConfiguration.from_dict(data)
        data = make_configuration().to_dict()
        data["schema_version"] = True
        with self.assertRaises(ValueError):
            ExperimentConfiguration.from_dict(data)
        with self.assertRaises(ValueError):
            ExperimentConfiguration.from_json('{"schema_version":NaN}')

    def test_invalid_attack_directions_applicability_and_protocol_are_rejected(self):
        with self.assertRaises(ValueError):
            replace(make_configuration().attacks[0], metric_direction="sideways")
        with self.assertRaises(ValueError):
            replace(make_configuration().attacks[0], applicability_decision="unknown")
        with self.assertRaises(ValueError):
            replace(make_configuration().attacks[0], applicable_image_categories=())
        data = make_configuration().to_dict()
        data["attacks"][0]["metric_direction"] = "sideways"
        with self.assertRaises(ValueError):
            ExperimentConfiguration.from_dict(data)

    def test_thresholds_scales_and_nonfinite_values_are_rejected(self):
        for criterion in (
            {"direction": "invalid", "threshold": 1.0},
            {"direction": "minimum", "threshold": float("nan")},
        ):
            with self.subTest(criterion=criterion), self.assertRaises(ValueError):
                BaselineAcceptanceCriterion(**criterion)
        for utility_args in (
            ("ssim", "sideways", 0.8, 0.1),
            ("ssim", "minimum", float("inf"), 0.1),
            ("ssim", "minimum", 0.8, 0),
            ("ssim", "minimum", 0.8, float("nan")),
        ):
            with self.subTest(utility_args=utility_args), self.assertRaises(ValueError):
                UtilityMetricConfiguration(*utility_args)
        with self.assertRaises(ValueError):
            DatasetConfiguration("x", "v", "cat", ("test",), {"bad": float("inf")}, "arbitrary")

    def test_invalid_transformation_menu_and_sigma_are_rejected(self):
        base = make_configuration().transformation
        for changes in (
            {"quantization_levels": (2, 2)},
            {"quantization_levels": (1, 4)},
            {"maximum_smoothing_strength": 0},
            {"maximum_smoothing_strength": float("nan")},
            {"maximum_smoothing_strength": 101},
            {"method": "invented_transform"},
            {"experiment_bit_depth": None},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(base, **changes)

    def test_jaya_configuration_bounds_and_parameters_are_validated(self):
        for args in (
            (1, 5, 0, (0, 0), (1, 1)),
            (4, 0, 0, (0, 0), (1, 1)),
            (4, 5, -1, (0, 0), (1, 1)),
            (4, 5, 0, (0, 0), (1, float("inf"))),
            (4, 5, 0, (0, 0), (1, 2)),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                JayaConfiguration(*args)

    def test_unvalidated_or_unacceptable_baselines_cannot_be_evidence(self):
        attack = make_configuration().attacks[0]
        unvalidated = AttackBaselineMeasurement(
            attack.name, attack.metric_name, 0.9, "unvalidated", attack.evaluation_protocol_identifier
        )
        with self.assertRaisesRegex(ValueError, "unvalidated"):
            unvalidated.validate_for(attack, "synthetic_rgb")
        below_criterion = AttackBaselineMeasurement(
            attack.name, attack.metric_name, 0.4, "validated", attack.evaluation_protocol_identifier
        )
        with self.assertRaisesRegex(ValueError, "acceptance criterion"):
            below_criterion.validate_for(attack, "synthetic_rgb")
        zero_baseline = AttackBaselineMeasurement(
            attack.name, attack.metric_name, 0.0, "validated", attack.evaluation_protocol_identifier
        )
        with self.assertRaisesRegex(ValueError, "zero original"):
            zero_baseline.validate_for(attack, "synthetic_rgb")

    def test_validated_baseline_requires_matching_attack_metric_category_and_protocol(self):
        attack = make_configuration().attacks[0]
        measurement = AttackBaselineMeasurement(
            attack.name, attack.metric_name, 0.75, "validated", attack.evaluation_protocol_identifier
        )
        self.assertEqual(measurement.validate_for(attack, "synthetic_rgb"), 0.75)
        with self.assertRaises(ValueError):
            measurement.validate_for(attack, "medical_image")
        with self.assertRaises(ValueError):
            replace(measurement, evaluation_protocol_identifier="other").validate_for(
                attack, "synthetic_rgb"
            )

    def test_serialization_rejects_non_json_values_and_unknown_fields(self):
        config_dict = make_configuration().to_dict()
        config_dict["unexpected"] = True
        with self.assertRaises(ValueError):
            ExperimentConfiguration.from_dict(config_dict)
        config_dict = make_configuration().to_dict()
        config_dict["dataset"]["preprocessing"] = {"nested": {1: "not a string key"}}
        with self.assertRaises(ValueError):
            ExperimentConfiguration.from_dict(config_dict)


if __name__ == "__main__":
    unittest.main()
