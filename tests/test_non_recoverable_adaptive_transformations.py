"""Tests for block-adaptive non-recoverable transformations."""
from __future__ import annotations

import unittest

import numpy as np

from src.non_recoverable.adaptive_transformations import (
    adaptive_combined_transformation,
    adaptive_gaussian_smoothing,
    adaptive_quantization,
)
from src.non_recoverable.baseline_transformations import (
    uniform_gaussian_smoothing,
    uniform_quantization,
)
from src.non_recoverable.image_input import prepare_image


class AdaptiveTransformationsTest(unittest.TestCase):
    def assert_valid_output(self, source: np.ndarray, result: np.ndarray) -> None:
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, source.dtype)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

    def test_row_major_quantization_parameters_apply_to_expected_rgb_blocks(self) -> None:
        source = np.empty((8, 8, 3), dtype=np.float32)
        source[:4, :4] = 0.30
        source[:4, 4:] = 0.30
        source[4:, :4] = 0.30
        source[4:, 4:] = 0.30
        levels = [2, 3, 4, 5]

        result = adaptive_quantization(source, levels, block_size=4)

        for index, (top, left, block_levels) in enumerate(
            ((0, 0, 2), (0, 4, 3), (4, 0, 4), (4, 4, 5))
        ):
            expected = uniform_quantization(source[top:top + 4, left:left + 4], block_levels)
            np.testing.assert_array_equal(
                result[top:top + 4, left:left + 4], expected,
                err_msg=f"row-major block {index} did not receive its configured level count",
            )
        self.assert_valid_output(source, result)

    def test_partial_grayscale_blocks_and_prepared_image_geometry(self) -> None:
        source = np.linspace(0, 1, 5 * 7, dtype=np.float32).reshape(5, 7)
        prepared = prepare_image((source * 255).astype(np.uint8), block_size=4)
        result = adaptive_quantization(prepared, [2, 3, 4, 5])

        self.assert_valid_output(prepared.image, result)
        self.assertEqual(result.shape, (5, 7))
        self.assertTrue(np.all(result[4:, :] == uniform_quantization(prepared.image[4:, :], 4)))
        with self.assertRaisesRegex(ValueError, "match"):
            adaptive_quantization(prepared, [2, 3, 4, 5], block_size=2)

    def test_quantization_rejects_wrong_parameter_count_and_invalid_values(self) -> None:
        source = np.zeros((5, 7), dtype=np.float32)
        for levels in ([2], [2, 3, 4], [2, 3, 4, 5, 6]):
            with self.subTest(levels=levels), self.assertRaisesRegex(ValueError, "exactly 4"):
                adaptive_quantization(source, levels, block_size=4)
        for levels in ([1, 2, 2, 2], [2, 3.5, 2, 2], [2, True, 2, 2], [2, np.nan, 2, 2]):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                adaptive_quantization(source, levels, block_size=4)

    def test_adaptive_smoothing_uses_each_row_major_sigma(self) -> None:
        source = np.zeros((8, 8), dtype=np.float32)
        source[2, 2] = 1.0
        source[2, 6] = 1.0
        source[6, 2] = 1.0
        source[6, 6] = 1.0
        strengths = [0.0, 0.8, 1.6, 2.4]

        result = adaptive_gaussian_smoothing(source, strengths, block_size=4)

        np.testing.assert_array_equal(result[:4, :4], source[:4, :4])
        self.assertFalse(np.array_equal(result[:4, 4:], source[:4, 4:]))
        self.assertFalse(np.array_equal(result[4:, :4], source[4:, :4]))
        self.assertFalse(np.array_equal(result[4:, 4:], source[4:, 4:]))
        self.assert_valid_output(source, result)

    def test_uniform_adaptive_smoothing_matches_full_image_baseline_without_tile_seams(self) -> None:
        source = np.random.default_rng(31).random((13, 19, 3), dtype=np.float32)
        expected = uniform_gaussian_smoothing(source, 1.75)
        actual = adaptive_gaussian_smoothing(source, [1.75] * 20, block_size=4)

        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        self.assert_valid_output(source, actual)

    def test_adjacent_strengths_use_full_neighbor_context_at_the_shared_edge(self) -> None:
        source = np.random.default_rng(12).random((4, 8), dtype=np.float32)
        actual = adaptive_gaussian_smoothing(source, [0.0, 2.0], block_size=4)
        right_block_from_full_context = uniform_gaussian_smoothing(source, 2.0)[:, 4:]

        np.testing.assert_array_equal(actual[:, :4], source[:, :4])
        np.testing.assert_allclose(
            actual[:, 4:], right_block_from_full_context, rtol=0, atol=1e-7
        )

    def test_adaptive_smoothing_handles_tiny_non_square_image_and_zero_sigma(self) -> None:
        source = np.array([[0.0, 0.25, 0.75]], dtype=np.float64)
        result = adaptive_gaussian_smoothing(source, [0.0], block_size=4)
        np.testing.assert_array_equal(result, source)
        self.assert_valid_output(source, result)

    def test_smoothing_rejects_wrong_count_and_invalid_strengths(self) -> None:
        source = np.zeros((5, 7, 3), dtype=np.float32)
        for strengths in ([0], [0] * 3, [0] * 5):
            with self.subTest(strengths=strengths), self.assertRaisesRegex(ValueError, "exactly 4"):
                adaptive_gaussian_smoothing(source, strengths, block_size=4)
        for strengths in ([-0.1, 0, 0, 0], [101, 0, 0, 0], [np.inf, 0, 0, 0], [True, 0, 0, 0]):
            with self.subTest(strengths=strengths), self.assertRaises(ValueError):
                adaptive_gaussian_smoothing(source, strengths, block_size=4)

    def test_combined_adaptive_matches_quantize_then_adaptively_smooth(self) -> None:
        source = np.random.default_rng(44).random((9, 11, 3), dtype=np.float32)
        levels = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        strengths = [0, 0.5, 1, 1.5, 2, 2.5, 1.25, 0.75, 0.25]
        combined = adaptive_combined_transformation(
            source, levels, strengths, block_size=4
        )
        expected = adaptive_gaussian_smoothing(
            adaptive_quantization(source, levels, block_size=4),
            strengths,
            block_size=4,
        )
        np.testing.assert_array_equal(combined, expected)
        self.assert_valid_output(source, combined)

    def test_combined_validates_both_parameter_vectors_before_transform(self) -> None:
        source = np.zeros((8, 8), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "strengths_per_block"):
            adaptive_combined_transformation(source, [2] * 4, [0] * 3, block_size=4)
        with self.assertRaisesRegex(ValueError, "levels_per_block"):
            adaptive_combined_transformation(source, [2] * 3, [0] * 4, block_size=4)

    def test_adaptive_transforms_are_deterministic(self) -> None:
        source = np.random.default_rng(7).random((7, 10), dtype=np.float32)
        levels = [2, 3, 4, 5, 6, 7]
        strengths = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7]
        np.testing.assert_array_equal(
            adaptive_quantization(source, levels, block_size=4),
            adaptive_quantization(source, levels, block_size=4),
        )
        np.testing.assert_array_equal(
            adaptive_gaussian_smoothing(source, strengths, block_size=4),
            adaptive_gaussian_smoothing(source, strengths, block_size=4),
        )
        np.testing.assert_array_equal(
            adaptive_combined_transformation(source, levels, strengths, block_size=4),
            adaptive_combined_transformation(source, levels, strengths, block_size=4),
        )


if __name__ == "__main__":
    unittest.main()
