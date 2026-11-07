"""Tests for utility metrics and attack-performance evaluation interfaces."""
from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.non_recoverable_metrics import (
    UtilityMetrics,
    calculate_utility_metrics,
    compare_attack_performance,
    create_evaluation_record,
    evaluate_attack,
    mean_absolute_error,
    peak_signal_to_noise_ratio,
    structural_similarity,
)
from src.non_recoverable.image_input import prepare_image


class ImageUtilityMetricTest(unittest.TestCase):
    def test_identical_grayscale_and_rgb_images_have_expected_metrics(self) -> None:
        for image in (
            np.random.default_rng(1).random((12, 15), dtype=np.float32),
            np.random.default_rng(2).random((12, 15, 3), dtype=np.float32),
        ):
            self.assertEqual(mean_absolute_error(image, image), 0.0)
            self.assertEqual(peak_signal_to_noise_ratio(image, image), float("inf"))
            self.assertEqual(structural_similarity(image, image), 1.0)

    def test_different_images_have_deterministic_metric_values(self) -> None:
        original = np.zeros((9, 13), dtype=np.float64)
        transformed = np.full((9, 13), 0.25, dtype=np.float64)
        metrics = calculate_utility_metrics(original, transformed)

        self.assertAlmostEqual(metrics.mae, 0.25)
        self.assertAlmostEqual(metrics.psnr_db, 20.0 * np.log10(4.0))
        self.assertEqual(metrics.ssim, calculate_utility_metrics(original, transformed).ssim)
        self.assertLess(metrics.ssim, 1.0)

    def test_metrics_accept_prepared_images(self) -> None:
        image = prepare_image(np.arange(100, dtype=np.uint8).reshape(10, 10), block_size=4)
        self.assertEqual(mean_absolute_error(image, image), 0.0)
        self.assertEqual(structural_similarity(image, image), 1.0)

    def test_invalid_dimensions_dtype_range_nan_and_infinity_are_rejected(self) -> None:
        valid = np.zeros((8, 9), dtype=np.float32)
        invalid_pairs = (
            (valid, np.zeros((9, 8), dtype=np.float32)),
            (valid, np.zeros((8, 9, 3), dtype=np.float32)),
            (valid.astype(np.uint8), valid.astype(np.uint8)),
            (np.full((8, 9), -0.1, dtype=np.float32), valid),
            (np.full((8, 9), 1.1, dtype=np.float32), valid),
            (np.full((8, 9), np.nan, dtype=np.float32), valid),
            (valid, np.full((8, 9), np.inf, dtype=np.float32)),
        )
        for original, transformed in invalid_pairs:
            with self.subTest(shape_a=original.shape, dtype_a=original.dtype), self.assertRaises(ValueError):
                mean_absolute_error(original, transformed)

    def test_ssim_small_image_edge_cases(self) -> None:
        identical = np.array([[0.5]], dtype=np.float32)
        self.assertEqual(structural_similarity(identical, identical), 1.0)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            structural_similarity(identical, np.array([[0.4]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "win_size"):
            structural_similarity(np.zeros((5, 5)), np.ones((5, 5)), win_size=4)


class AttackComparisonTest(unittest.TestCase):
    def test_higher_is_better_attack_metric_reports_relative_reduction(self) -> None:
        result = compare_attack_performance(
            0.8, 0.4, metric_direction="higher_is_better", applicable=True
        )
        self.assertEqual(result.status, "valid")
        self.assertAlmostEqual(result.relative_performance_reduction, 0.5)
        self.assertEqual(result.original.value, 0.8)
        self.assertEqual(result.transformed.value, 0.4)

    def test_lower_is_better_attack_metric_uses_correct_direction(self) -> None:
        result = evaluate_attack(
            "test_attack",
            "error_rate",
            0.2,
            0.3,
            metric_direction="lower_is_better",
            applicable=True,
            applicability_note="Applicable to generated test data only.",
        )
        self.assertAlmostEqual(result.relative_performance_reduction, 0.5)
        self.assertEqual(result.attack_name, "test_attack")
        self.assertEqual(result.metric_name, "error_rate")
        self.assertEqual(result.applicability_note, "Applicable to generated test data only.")

    def test_improved_attack_performance_has_negative_reduction(self) -> None:
        higher = compare_attack_performance(
            0.5, 0.75, metric_direction="higher_is_better", applicable=True
        )
        lower = compare_attack_performance(
            0.5, 0.25, metric_direction="lower_is_better", applicable=True
        )
        self.assertLess(higher.relative_performance_reduction, 0)
        self.assertLess(lower.relative_performance_reduction, 0)

    def test_missing_zero_and_invalid_original_baselines_never_report_reduction(self) -> None:
        cases = (
            (None, 0.4, "missing_original_measurement"),
            (0.0, 0.4, "zero_original_baseline"),
            (float("nan"), 0.4, "invalid_original_measurement"),
            (float("inf"), 0.4, "invalid_original_measurement"),
        )
        for original, transformed, expected_status in cases:
            with self.subTest(original=original):
                result = compare_attack_performance(
                    original,
                    transformed,
                    metric_direction="higher_is_better",
                    applicable=True,
                )
                self.assertEqual(result.status, expected_status)
                self.assertIsNone(result.relative_performance_reduction)
                if isinstance(original, float) and np.isnan(original):
                    self.assertEqual(result.original.reported_value, "NaN")
                if isinstance(original, float) and np.isinf(original):
                    self.assertEqual(result.original.reported_value, "Infinity")

    def test_missing_or_invalid_transformed_measurement_is_explicit(self) -> None:
        for value, expected_status in (
            (None, "missing_transformed_measurement"),
            (float("nan"), "invalid_transformed_measurement"),
        ):
            result = compare_attack_performance(
                0.8, value, metric_direction="higher_is_better", applicable=True
            )
            self.assertEqual(result.status, expected_status)
            self.assertIsNone(result.relative_performance_reduction)
            self.assertIsNone(result.transformed.value)

    def test_attack_applicability_is_explicit(self) -> None:
        not_applicable = compare_attack_performance(
            0.8, 0.4, metric_direction="higher_is_better", applicable=False
        )
        unknown = compare_attack_performance(
            0.8, 0.4, metric_direction="higher_is_better", applicable=None
        )
        self.assertEqual(not_applicable.status, "not_applicable")
        self.assertEqual(unknown.status, "applicability_unknown")
        self.assertIsNone(not_applicable.relative_performance_reduction)
        self.assertIsNone(unknown.relative_performance_reduction)

    def test_invalid_metric_direction_and_boolean_measurements_are_rejected_or_marked(self) -> None:
        with self.assertRaises(ValueError):
            compare_attack_performance(1, 0, metric_direction="accuracy", applicable=True)
        result = compare_attack_performance(
            True, 0.4, metric_direction="higher_is_better", applicable=True
        )
        self.assertEqual(result.status, "invalid_original_measurement")
        self.assertIsNone(result.relative_performance_reduction)


class EvaluationRecordTest(unittest.TestCase):
    def test_privacy_and_utility_remain_separate_and_values_are_not_invented(self) -> None:
        attack = evaluate_attack(
            "provided_attack",
            "balanced_accuracy",
            None,
            None,
            metric_direction="higher_is_better",
            applicable=None,
        )
        record = create_evaluation_record(
            dataset_name="test_dataset",
            image_category="synthetic",
            transformation_name="uniform_quantization",
            transformation_configuration={"levels": 8},
            privacy=attack,
            utility=UtilityMetrics(),
            limitations=("No attack experiment has been run.",),
        )
        payload = record.to_dict()
        self.assertEqual(payload["privacy"]["status"], "applicability_unknown")
        self.assertIsNone(payload["privacy"]["relative_performance_reduction"])
        self.assertIsNone(payload["utility"]["psnr_db"])
        self.assertIsNone(payload["utility"]["ssim"])
        self.assertIsNone(payload["utility"]["mae"])
        self.assertIn("utility", payload)
        self.assertIn("privacy", payload)


if __name__ == "__main__":
    unittest.main()
