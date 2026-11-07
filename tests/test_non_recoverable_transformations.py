"""Tests for non-recoverable baseline image transformations."""
from __future__ import annotations

import unittest

import numpy as np

from src.non_recoverable.baseline_transformations import (
    MAX_QUANTIZATION_LEVELS,
    MAX_SMOOTHING_STRENGTH,
    MIN_QUANTIZATION_LEVELS,
    combined_transformation,
    uniform_gaussian_smoothing,
    uniform_quantization,
)
from src.non_recoverable.image_input import prepare_image


class BaselineTransformationsTest(unittest.TestCase):
    def assert_valid_output(self, source: np.ndarray, result: np.ndarray) -> None:
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, source.dtype)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

    def test_quantization_gray_and_rgb_at_minimum_and_maximum_levels(self) -> None:
        gray = np.array([[0.0, 0.2, 0.5, 0.8, 1.0]], dtype=np.float32)
        rgb = np.stack((gray, gray / 2, 1 - gray), axis=-1)

        minimum = uniform_quantization(gray, MIN_QUANTIZATION_LEVELS)
        maximum = uniform_quantization(rgb, MAX_QUANTIZATION_LEVELS)

        np.testing.assert_array_equal(minimum, np.array([[0, 0, 0, 1, 1]], np.float32))
        expected_maximum = (
            np.rint(rgb.astype(np.float64) * (MAX_QUANTIZATION_LEVELS - 1))
            / (MAX_QUANTIZATION_LEVELS - 1)
        ).astype(rgb.dtype)
        np.testing.assert_array_equal(maximum, expected_maximum)
        self.assert_valid_output(gray, minimum)
        self.assert_valid_output(rgb, maximum)

    def test_quantization_preserves_non_square_shape_dtype_and_level_grid(self) -> None:
        source = np.random.default_rng(23).random((3, 7, 3), dtype=np.float32)
        result = uniform_quantization(source, 5)
        self.assert_valid_output(source, result)
        scaled = result * 4
        np.testing.assert_allclose(scaled, np.rint(scaled), atol=2e-7)

    def test_quantization_accepts_prepared_image_and_preserves_its_dtype(self) -> None:
        prepared = prepare_image(np.arange(15, dtype=np.uint8).reshape(3, 5), block_size=4)
        result = uniform_quantization(prepared, 8)
        self.assert_valid_output(prepared.image, result)

    def test_invalid_quantization_levels_and_inputs_are_rejected(self) -> None:
        valid = np.zeros((2, 3), dtype=np.float32)
        for levels in (1, MAX_QUANTIZATION_LEVELS + 1, 2.5, True):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                uniform_quantization(valid, levels)
        invalid_inputs = (
            np.empty((0, 3), dtype=np.float32),
            np.zeros((2, 3, 4), dtype=np.float32),
            np.zeros((2, 3), dtype=np.uint8),
            np.array([[np.nan]], dtype=np.float32),
            np.array([[np.inf]], dtype=np.float32),
            np.array([[-0.1]], dtype=np.float32),
            np.array([[1.1]], dtype=np.float32),
        )
        for image in invalid_inputs:
            with self.subTest(shape=image.shape, dtype=image.dtype), self.assertRaises(ValueError):
                uniform_quantization(image)

    def test_zero_smoothing_returns_exact_unchanged_copy_for_gray_and_rgb(self) -> None:
        for source in (
            np.random.default_rng(5).random((1, 4), dtype=np.float32),
            np.random.default_rng(6).random((3, 7, 3), dtype=np.float64),
        ):
            result = uniform_gaussian_smoothing(source, strength=0)
            self.assert_valid_output(source, result)
            np.testing.assert_array_equal(result, source)
            self.assertIsNot(result, source)

    def test_increasing_gaussian_strength_smooths_impulse_more(self) -> None:
        source = np.zeros((21, 31), dtype=np.float32)
        source[10, 15] = 1.0
        mild = uniform_gaussian_smoothing(source, strength=1.0)
        stronger = uniform_gaussian_smoothing(source, strength=3.0)
        self.assert_valid_output(source, mild)
        self.assert_valid_output(source, stronger)
        self.assertLess(float(stronger.max()), float(mild.max()))
        self.assertLess(float(np.var(stronger)), float(np.var(mild)))

    def test_smoothing_handles_small_non_square_rgb_and_preserves_channels(self) -> None:
        source = np.zeros((1, 5, 3), dtype=np.float16)
        source[0, 2, 1] = 1
        result = uniform_gaussian_smoothing(source, strength=2.0)
        self.assert_valid_output(source, result)
        self.assertEqual(result.shape, (1, 5, 3))
        self.assertTrue(np.all(result[..., 0] == 0))
        self.assertTrue(np.all(result[..., 2] == 0))

    def test_invalid_smoothing_strengths_are_rejected(self) -> None:
        image = np.zeros((2, 3), dtype=np.float32)
        for strength in (-0.1, np.nan, np.inf, MAX_SMOOTHING_STRENGTH + 0.1, True, "1"):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                uniform_gaussian_smoothing(image, strength)

    def test_combined_operation_is_quantization_followed_by_smoothing(self) -> None:
        source = np.zeros((5, 9, 3), dtype=np.float32)
        source[2, 4] = (0.25, 0.75, 1.0)
        actual = combined_transformation(
            source, quantization_levels=5, smoothing_strength=1.25
        )
        expected = uniform_gaussian_smoothing(
            uniform_quantization(source, 5), 1.25
        )
        np.testing.assert_array_equal(actual, expected)
        self.assert_valid_output(source, actual)

    def test_combined_zero_smoothing_matches_quantization_and_invalid_configs_fail_early(self) -> None:
        source = np.random.default_rng(2).random((2, 6), dtype=np.float32)
        np.testing.assert_array_equal(
            combined_transformation(source, quantization_levels=4, smoothing_strength=0),
            uniform_quantization(source, 4),
        )
        for kwargs in (
            {"quantization_levels": 1, "smoothing_strength": 1},
            {"quantization_levels": 4, "smoothing_strength": np.inf},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                combined_transformation(source, **kwargs)

    def test_transformations_are_deterministic(self) -> None:
        source = np.random.default_rng(14).random((4, 8, 3), dtype=np.float32)
        np.testing.assert_array_equal(
            uniform_quantization(source, 17), uniform_quantization(source, 17)
        )
        np.testing.assert_array_equal(
            uniform_gaussian_smoothing(source, 2.5),
            uniform_gaussian_smoothing(source, 2.5),
        )
        np.testing.assert_array_equal(
            combined_transformation(source, quantization_levels=17, smoothing_strength=2.5),
            combined_transformation(source, quantization_levels=17, smoothing_strength=2.5),
        )


if __name__ == "__main__":
    unittest.main()
