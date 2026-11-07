"""Focused tests for non-recoverable feature matrices and Jaya plumbing."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.non_recoverable.feature_matrix import (
    DEFAULT_FEATURE_NAMES,
    ImageFeatureMatrix,
    extract_feature_matrix,
)
from src.non_recoverable.image_input import PreparedImage, prepare_image
from src.non_recoverable.optimizer import (
    optimize_feature_weights,
    project_bounded_simplex,
)


class FeatureMatrixTest(unittest.TestCase):
    def test_rgb_and_partial_blocks_preserve_row_major_geometry(self) -> None:
        prepared = prepare_image(
            np.random.default_rng(17).integers(0, 256, (5, 7, 3), dtype=np.uint8),
            block_size=4,
        )
        matrix = extract_feature_matrix(prepared)

        self.assertEqual(matrix.values.shape, (4, len(DEFAULT_FEATURE_NAMES)))
        self.assertEqual(matrix.feature_names, DEFAULT_FEATURE_NAMES)
        self.assertEqual(
            [(block["row"], block["column"]) for block in matrix.blocks],
            [(0, 0), (0, 4), (4, 0), (4, 4)],
        )
        self.assertEqual(
            [block["valid_pixel_count"] for block in matrix.blocks], [16, 12, 4, 3]
        )
        self.assertEqual(sum(block["valid_pixel_count"] for block in matrix.blocks), 35)
        self.assertEqual(matrix.image_metadata, prepared.metadata)
        self.assertTrue(np.all(np.isfinite(matrix.values)))

    def test_grayscale_and_alternate_dimensions_produce_expected_block_counts(self) -> None:
        prepared = prepare_image(np.arange(6 * 10, dtype=np.uint8).reshape(6, 10), 3)
        matrix = extract_feature_matrix(prepared, ("entropy", "mean"))

        self.assertEqual(matrix.values.shape, (8, 2))
        self.assertEqual(matrix.feature_names, ("entropy", "mean"))
        self.assertEqual(matrix.blocks[-1]["valid_pixel_count"], 3)
        self.assertEqual(sum(block["valid_pixel_count"] for block in matrix.blocks), 60)

    def test_feature_order_follows_requested_columns_deterministically(self) -> None:
        prepared = prepare_image(np.arange(16, dtype=np.uint8).reshape(4, 4))
        first = extract_feature_matrix(prepared, ("roi_score", "mean", "variance"))
        second = extract_feature_matrix(prepared, ("roi_score", "mean", "variance"))

        self.assertEqual(first.feature_names, ("roi_score", "mean", "variance"))
        np.testing.assert_array_equal(first.values, second.values)

    def test_invalid_names_empty_records_missing_features_and_nonfinite_values_rejected(self) -> None:
        prepared = prepare_image(np.arange(16, dtype=np.uint8).reshape(4, 4))
        for names in ((), ("mean", "mean"), ("missing",)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                extract_feature_matrix(prepared, names)

        with patch("src.non_recoverable.feature_matrix.extract_image_block_features", return_value=[]):
            with self.assertRaisesRegex(ValueError, "no blocks"):
                extract_feature_matrix(prepared)

        valid_record = {
            "block_id": 0,
            "row": 0,
            "column": 0,
            "valid_height": 4,
            "valid_width": 4,
            "valid_pixel_count": 16,
            **{name: 0.5 for name in DEFAULT_FEATURE_NAMES},
        }
        for bad_value in (float("nan"), float("inf"), "not-numeric"):
            row = dict(valid_record, mean=bad_value)
            with patch(
                "src.non_recoverable.feature_matrix.extract_image_block_features",
                return_value=[row],
            ), self.subTest(bad_value=bad_value), self.assertRaises(ValueError):
                extract_feature_matrix(prepared)

        missing_record = dict(valid_record)
        del missing_record["entropy"]
        with patch(
            "src.non_recoverable.feature_matrix.extract_image_block_features",
            return_value=[missing_record],
        ), self.assertRaisesRegex(ValueError, "Unknown block feature"):
            extract_feature_matrix(prepared)

    def test_inconsistent_block_records_are_rejected(self) -> None:
        prepared = prepare_image(np.arange(16, dtype=np.uint8).reshape(4, 4))
        record = {
            "block_id": 1,
            "row": 0,
            "column": 0,
            "valid_height": 4,
            "valid_width": 4,
            "valid_pixel_count": 16,
            **{name: 0.5 for name in DEFAULT_FEATURE_NAMES},
        }
        with patch(
            "src.non_recoverable.feature_matrix.extract_image_block_features",
            return_value=[record],
        ), self.assertRaisesRegex(ValueError, "inconsistent"):
            extract_feature_matrix(prepared)


class FeatureWeightOptimizerTest(unittest.TestCase):
    def setUp(self) -> None:
        prepared = prepare_image(
            np.random.default_rng(9).integers(0, 256, (8, 12, 3), dtype=np.uint8),
            block_size=4,
        )
        self.features = extract_feature_matrix(prepared, ("mean", "variance", "entropy"))

    def test_simplex_projection_obeys_bounds_and_normalization(self) -> None:
        weights = project_bounded_simplex(np.array([-2.0, 0.3, 4.0]), 0.1, 0.7)
        self.assertTrue(np.all(weights >= 0.1))
        self.assertTrue(np.all(weights <= 0.7))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)

    def test_optimizer_runs_with_caller_objective_and_correct_dimensions(self) -> None:
        observed_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def objective(weights: np.ndarray, matrix: np.ndarray) -> float:
            observed_shapes.append((weights.shape, matrix.shape))
            return float(weights @ np.array([0.1, 0.8, 0.1]))

        result = optimize_feature_weights(
            self.features,
            objective,
            population_size=8,
            iterations=6,
            seed=21,
            lower_weight=0.1,
            upper_weight=0.7,
        )

        weights = np.asarray(list(result["weights"].values()))
        self.assertEqual(result["feature_names"], ["mean", "variance", "entropy"])
        self.assertTrue(observed_shapes)
        self.assertEqual(observed_shapes[0], ((3,), self.features.values.shape))
        self.assertTrue(np.all(weights >= 0.1))
        self.assertTrue(np.all(weights <= 0.7))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        self.assertEqual(len(result["convergence_history"]), 7)

    def test_fixed_seed_is_reproducible(self) -> None:
        objective = lambda weights, matrix: float(weights @ np.array([0.2, 0.5, 0.3]))
        first = optimize_feature_weights(self.features, objective, population_size=6, iterations=5, seed=4)
        second = optimize_feature_weights(self.features, objective, population_size=6, iterations=5, seed=4)
        self.assertEqual(first["weights"], second["weights"])
        self.assertEqual(first["convergence_history"], second["convergence_history"])
        np.testing.assert_array_equal(first["best_solution_history"], second["best_solution_history"])

    def test_invalid_configurations_and_objective_results_are_rejected(self) -> None:
        objective = lambda weights, matrix: float(weights.sum())
        for kwargs in (
            {"population_size": 1},
            {"iterations": 0},
            {"seed": 1.5},
            {"lower_weight": 0.4, "upper_weight": 0.5},
            {"lower_weight": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                optimize_feature_weights(self.features, objective, **kwargs)

        with self.assertRaises(ValueError):
            optimize_feature_weights(self.features, lambda weights, matrix: float("nan"))

        bad_matrix = ImageFeatureMatrix(
            values=np.array([[0.1, np.inf, 0.3]]),
            feature_names=("mean", "variance", "entropy"),
            blocks=({"block_id": 0},),
            image_metadata={},
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            optimize_feature_weights(bad_matrix, objective)

        wrong_rows = ImageFeatureMatrix(
            values=np.ones((2, 3)),
            feature_names=("mean", "variance", "entropy"),
            blocks=({"block_id": 0},),
            image_metadata={},
        )
        with self.assertRaisesRegex(ValueError, "Block metadata count"):
            optimize_feature_weights(wrong_rows, objective)


if __name__ == "__main__":
    unittest.main()
