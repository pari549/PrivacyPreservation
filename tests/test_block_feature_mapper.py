"""Tests for feature-driven per-block transformation parameter selection."""
from __future__ import annotations

import unittest

import numpy as np

from src.non_recoverable.block_feature_mapper import map_block_features_to_parameters
from src.non_recoverable.block_feature_mapper import DEFAULT_FEATURE_WEIGHTS
from src.non_recoverable.baseline_transformations import (
    MAX_QUANTIZATION_LEVELS,
    MAX_SMOOTHING_STRENGTH,
    MIN_QUANTIZATION_LEVELS,
)
from src.non_recoverable.feature_matrix import DEFAULT_FEATURE_NAMES, ImageFeatureMatrix
from src.non_recoverable.optimizer import optimize_feature_weights
from src.optimization.objective_functions import FEATURE_NAMES


class BlockFeatureMapperTest(unittest.TestCase):
    def matrix(self, priority_values: tuple[tuple[float, ...], ...]) -> ImageFeatureMatrix:
        rows = []
        for variance, gradient, entropy, roi_score in priority_values:
            rows.append(
                {
                    "mean": 0.5,
                    "variance": variance,
                    "std": variance**0.5,
                    "gradient": gradient,
                    "edge_density": gradient,
                    "entropy": entropy,
                    "roi_score": roi_score,
                }
            )
        values = np.asarray(
            [[row[name] for name in DEFAULT_FEATURE_NAMES] for row in rows],
            dtype=np.float64,
        )
        return ImageFeatureMatrix(
            values=values,
            feature_names=DEFAULT_FEATURE_NAMES,
            blocks=tuple(
                {
                    "block_id": index,
                    "row": 0,
                    "column": index * 4,
                    "valid_height": 4,
                    "valid_width": 4,
                    "valid_pixel_count": 16,
                }
                for index in range(len(rows))
            ),
            image_metadata={"block_size": 4, "original_height": 4, "original_width": 4 * len(rows)},
        )

    def test_existing_priority_features_change_selected_parameters(self) -> None:
        matrix = self.matrix(
            (
                (0.0, 0.0, 0.0, 0.0),
                (0.02, 0.1, 1.0, 0.2),
                (0.08, 0.3, 2.0, 0.6),
                (0.16, 0.6, 3.5, 0.9),
            )
        )

        result = map_block_features_to_parameters(
            matrix, quantization_levels=64, smoothing_strength=8.0
        )

        self.assertEqual(result.feature_names, ("variance", "gradient", "entropy", "roi_score"))
        self.assertEqual(tuple(sorted(result.priorities)), result.priorities)
        self.assertEqual(tuple(sorted(result.scaled_priorities)), result.scaled_priorities)
        self.assertGreater(len(set(result.levels_per_block)), 1)
        self.assertGreater(len(set(result.strengths_per_block)), 1)
        self.assertTrue(all(a >= b for a, b in zip(result.levels_per_block, result.levels_per_block[1:])))
        self.assertTrue(all(a <= b for a, b in zip(result.strengths_per_block, result.strengths_per_block[1:])))
        np.testing.assert_allclose(result.feature_weights, DEFAULT_FEATURE_WEIGHTS)

    def test_named_jaya_weights_change_priorities_and_parameters(self) -> None:
        matrix = self.matrix(
            (
                (0.25, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        variance_result = {
            "optimized_weights": {
                "variance": 0.7,
                "gradient": 0.1,
                "entropy": 0.1,
                "roi": 0.1,
            }
        }
        roi_result = {
            "optimized_weights": {
                "variance": 0.1,
                "gradient": 0.1,
                "entropy": 0.1,
                "roi": 0.7,
            }
        }

        variance_parameters = map_block_features_to_parameters(
            matrix,
            quantization_levels=64,
            smoothing_strength=8.0,
            feature_weights=variance_result,
        )
        roi_parameters = map_block_features_to_parameters(
            matrix,
            quantization_levels=64,
            smoothing_strength=8.0,
            feature_weights=roi_result,
        )

        np.testing.assert_allclose(
            variance_parameters.feature_weights, (0.7, 0.1, 0.1, 0.1)
        )
        np.testing.assert_allclose(roi_parameters.feature_weights, (0.1, 0.1, 0.1, 0.7))
        self.assertNotEqual(variance_parameters.priorities, roi_parameters.priorities)
        self.assertNotEqual(
            variance_parameters.levels_per_block, roi_parameters.levels_per_block
        )
        self.assertNotEqual(
            variance_parameters.strengths_per_block, roi_parameters.strengths_per_block
        )

    def test_existing_feature_optimizer_result_flows_into_mapper(self) -> None:
        full_matrix = self.matrix(
            (
                (0.0, 0.0, 0.0, 0.0),
                (0.02, 0.1, 1.0, 0.2),
                (0.08, 0.3, 2.0, 0.6),
                (0.16, 0.6, 3.5, 0.9),
            )
        )
        indices = [full_matrix.feature_names.index(name) for name in FEATURE_NAMES]
        priority_matrix = ImageFeatureMatrix(
            values=full_matrix.values[:, indices],
            feature_names=FEATURE_NAMES,
            blocks=full_matrix.blocks,
            image_metadata=full_matrix.image_metadata,
        )

        # This synthetic objective tests result plumbing only; it makes no
        # privacy claim and is not used by the application.
        optimizer_result = optimize_feature_weights(
            priority_matrix,
            lambda weights, matrix: float(weights[0]),
            population_size=6,
            iterations=3,
            seed=23,
        )
        mapped = map_block_features_to_parameters(
            full_matrix,
            quantization_levels=64,
            smoothing_strength=8.0,
            feature_weights=optimizer_result,
        )

        self.assertEqual(mapped.feature_names, FEATURE_NAMES)
        self.assertTrue(all(0.1 <= weight <= 0.7 for weight in mapped.feature_weights))
        self.assertAlmostEqual(sum(mapped.feature_weights), 1.0, places=12)
        self.assertNotEqual(mapped.feature_weights, DEFAULT_FEATURE_WEIGHTS)

    def test_generated_parameters_respect_existing_transform_bounds(self) -> None:
        matrix = self.matrix(
            ((0.0, 0.0, 0.0, 0.0), (0.25, 1.0, 4.0, 1.0))
        )
        result = map_block_features_to_parameters(
            matrix,
            quantization_levels=MAX_QUANTIZATION_LEVELS,
            smoothing_strength=MAX_SMOOTHING_STRENGTH,
        )

        self.assertTrue(
            all(MIN_QUANTIZATION_LEVELS <= value <= MAX_QUANTIZATION_LEVELS
                for value in result.levels_per_block)
        )
        self.assertTrue(
            all(0.0 <= value <= MAX_SMOOTHING_STRENGTH
                for value in result.strengths_per_block)
        )

    def test_explicit_per_block_override_can_leave_either_parameter_unmapped(self) -> None:
        matrix = self.matrix(((0.0, 0.0, 0.0, 0.0), (0.25, 1.0, 4.0, 1.0)))
        levels_only = map_block_features_to_parameters(
            matrix, quantization_levels=8, smoothing_strength=None
        )
        smoothing_only = map_block_features_to_parameters(
            matrix, quantization_levels=None, smoothing_strength=2.0
        )

        self.assertEqual(len(levels_only.levels_per_block), 2)
        self.assertEqual(levels_only.strengths_per_block, ())
        self.assertEqual(smoothing_only.levels_per_block, ())
        self.assertEqual(len(smoothing_only.strengths_per_block), 2)

    def test_priority_features_are_required_and_invalid_bases_are_rejected(self) -> None:
        matrix = self.matrix(((0.1, 0.2, 1.0, 0.3),))
        no_entropy = ImageFeatureMatrix(
            values=matrix.values[:, [i for i, name in enumerate(matrix.feature_names) if name != "entropy"]],
            feature_names=tuple(name for name in matrix.feature_names if name != "entropy"),
            blocks=matrix.blocks,
            image_metadata=matrix.image_metadata,
        )
        with self.assertRaisesRegex(ValueError, "entropy"):
            map_block_features_to_parameters(no_entropy, quantization_levels=8)
        for levels in (1, MAX_QUANTIZATION_LEVELS + 1, True):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                map_block_features_to_parameters(matrix, quantization_levels=levels)
        for strength in (-0.1, MAX_SMOOTHING_STRENGTH + 0.1, float("nan"), True):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                map_block_features_to_parameters(matrix, smoothing_strength=strength)

    def test_invalid_or_incomplete_optimizer_weights_are_rejected(self) -> None:
        matrix = self.matrix(((0.1, 0.2, 1.0, 0.3),))
        for weights in (
            {"variance": 0.7, "gradient": 0.1, "entropy": 0.2},
            {"variance": 0.7, "gradient": 0.1, "entropy": 0.1, "roi": float("nan")},
            {
                "variance": 0.7,
                "gradient": 0.1,
                "entropy": 0.1,
                "roi": 0.1,
                "roi_score": 0.1,
            },
            {"optimized_weights": [0.25, 0.25, 0.25, 0.25]},
            [0.2, 0.3],
        ):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                map_block_features_to_parameters(
                    matrix, quantization_levels=8, feature_weights=weights
                )


if __name__ == "__main__":
    unittest.main()
