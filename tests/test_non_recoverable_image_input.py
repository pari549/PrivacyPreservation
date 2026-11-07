"""Temporary-image tests for universal non-recoverable input preparation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.image_processing.gradients import gradient_maps
from src.non_recoverable import (
    ImageInputError,
    ImageSizeLimitError,
    extract_image_block_features,
    iter_image_blocks,
    load_image,
    prepare_image,
    restore_original_dimensions,
)


class NonRecoverableImageInputTest(unittest.TestCase):
    def test_rgb_png_loads_and_normalizes(self) -> None:
        pixels = np.zeros((12, 16, 3), dtype=np.uint8)
        pixels[2, 3] = (255, 128, 64)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rgb.png"
            Image.fromarray(pixels, mode="RGB").save(path)
            prepared = load_image(path, block_size=4)
        self.assertEqual(prepared.image.shape, (12, 16, 3))
        self.assertEqual(prepared.image.dtype, np.float32)
        np.testing.assert_allclose(prepared.image[2, 3], [1.0, 128 / 255, 64 / 255])
        self.assertEqual(prepared.metadata["source_format"], "PNG")
        self.assertEqual(prepared.metadata["original_channels"], 3)
        self.assertEqual(prepared.metadata["max_image_pixels"], 25_000_000)

    def test_rgba_is_composited_over_configured_background(self) -> None:
        pixels = np.array([[[20, 40, 200, 0], [200, 0, 0, 128]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "transparent.png"
            Image.fromarray(pixels).save(path)
            prepared = load_image(path, background_color=(10, 20, 30))
        np.testing.assert_allclose(prepared.image[0, 0], np.array([10, 20, 30]) / 255)
        np.testing.assert_allclose(prepared.image[0, 1], np.array([105, 10, 15]) / 255, atol=1 / 255)
        self.assertEqual(
            prepared.metadata["alpha_handling_policy"],
            "alpha-composite over configured opaque RGB background before channel conversion",
        )
        self.assertEqual(prepared.metadata["background_color_rgb"], [10, 20, 30])

    def test_palette_transparency_is_composited_to_white(self) -> None:
        palette = Image.new("P", (2, 1))
        palette.putpalette([255, 0, 0, 0, 0, 255] + [0, 0, 0] * 254)
        palette.putdata([0, 1])
        palette.info["transparency"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "palette.png"
            palette.save(path)
            prepared = load_image(path)
        np.testing.assert_array_equal(prepared.image[0, 1], np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(prepared.image[0, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
        self.assertIn("alpha-composite", prepared.metadata["alpha_handling_policy"])
        self.assertEqual(prepared.metadata["background_color_rgb"], [255, 255, 255])

    def test_grayscale_jpeg_and_non_square_dimensions(self) -> None:
        pixels = np.tile(np.arange(23, dtype=np.uint8), (11, 1))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gray.jpg"
            Image.fromarray(pixels, mode="L").save(path, quality=100)
            prepared = load_image(path, block_size=4)
        self.assertEqual(prepared.image.shape, (11, 23))
        self.assertEqual(prepared.metadata["original_channels"], 1)
        self.assertEqual(prepared.metadata["source_format"], "JPEG")

    def test_exif_rotated_jpeg_is_oriented_before_processing(self) -> None:
        pixels = np.zeros((8, 14, 3), dtype=np.uint8)
        pixels[:4, :7] = (255, 0, 0)
        image = Image.fromarray(pixels, mode="RGB")
        exif = image.getexif()
        exif[274] = 6  # Rotate 90 degrees clockwise for display.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rotated.jpg"
            image.save(path, quality=100, exif=exif)
            prepared = load_image(path)
        self.assertEqual((prepared.metadata["encoded_height"], prepared.metadata["encoded_width"]), (8, 14))
        self.assertEqual((prepared.metadata["original_height"], prepared.metadata["original_width"]), (14, 8))
        self.assertTrue(prepared.metadata["exif_orientation_applied"])
        self.assertEqual(prepared.metadata["exif_orientation_value"], 6)
        self.assertEqual(prepared.metadata["exif_orientation_policy"], "apply EXIF transpose before channel conversion")

    def test_remainder_dimensions_are_edge_padded_and_all_source_pixels_are_covered(self) -> None:
        source = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        prepared = prepare_image(source, block_size=4)
        self.assertEqual(prepared.padded_image.shape, (8, 8, 3))
        self.assertEqual(prepared.metadata["pad_bottom"], 3)
        self.assertEqual(prepared.metadata["pad_right"], 1)
        self.assertEqual(prepared.metadata["edge_policy"], "replicate bottom and right edge pixels")
        np.testing.assert_array_equal(
            restore_original_dimensions(prepared.padded_image, prepared.metadata), prepared.image
        )
        modified = prepared.padded_image.copy()
        modified[0, 0] = 0.75
        restored = restore_original_dimensions(modified, prepared.metadata)
        self.assertEqual(restored.shape, prepared.image.shape)
        np.testing.assert_array_equal(restored[0, 0], np.full(3, 0.75, dtype=np.float32))

        coverage = np.zeros(prepared.padded_image.shape[:2], dtype=np.uint8)
        blocks = list(iter_image_blocks(prepared))
        for _, row, column, block in blocks:
            self.assertEqual(block.shape, (4, 4, 3))
            coverage[row:row + 4, column:column + 4] += 1
        np.testing.assert_array_equal(coverage, np.ones((8, 8), dtype=np.uint8))
        np.testing.assert_array_equal(coverage[:5, :7], np.ones((5, 7), dtype=np.uint8))
        self.assertEqual(sum(record["valid_pixel_count"] for record in extract_image_block_features(prepared)), 5 * 7)

    def test_feature_primitives_work_for_grayscale_and_variable_block_size(self) -> None:
        prepared = prepare_image(np.arange(6 * 10, dtype=np.uint8).reshape(6, 10), block_size=3)
        features = extract_image_block_features(prepared)
        self.assertEqual(len(features), 8)
        self.assertEqual(sum(feature["valid_pixel_count"] for feature in features), 60)
        self.assertTrue(all(0 <= feature["roi_score"] <= 1 for feature in features))

    def test_rgb_features_and_partial_blocks_use_valid_source_pixels(self) -> None:
        source = np.zeros((5, 7, 3), dtype=np.uint8)
        source[:, 6, :] = 255
        prepared = prepare_image(source, block_size=4)
        features = extract_image_block_features(prepared)
        right_edge_block = next(feature for feature in features if feature["row"] == 0 and feature["column"] == 4)
        valid_crop = prepared.image[:4, 4:7]
        self.assertEqual(right_edge_block["valid_height"], 4)
        self.assertEqual(right_edge_block["valid_width"], 3)
        self.assertEqual(right_edge_block["valid_pixel_count"], 12)
        self.assertAlmostEqual(right_edge_block["mean"], float(valid_crop.mean()))
        self.assertNotAlmostEqual(right_edge_block["mean"], float(prepared.padded_image[:4, 4:8].mean()))

        _, _, full_image_magnitude = gradient_maps(prepared.image)
        self.assertAlmostEqual(
            right_edge_block["gradient"], float(full_image_magnitude[:4, 4:7].mean())
        )

    def test_integer_float_and_explicit_ranges_normalize_consistently(self) -> None:
        byte_image = np.array([[0, 255]], dtype=np.uint8)
        word_image = np.array([[0, 65535]], dtype=np.uint16)
        float_image = np.array([[0.0, 255.0]], dtype=np.float32)
        signed_image = np.array([[-10, 10]], dtype=np.int16)
        for image, value_range in (
            (byte_image, None),
            (word_image, None),
            (float_image, None),
            (signed_image, (-10, 10)),
        ):
            with self.subTest(dtype=image.dtype, value_range=value_range):
                prepared = prepare_image(image, block_size=1, value_range=value_range)
                np.testing.assert_array_equal(prepared.image, np.array([[0.0, 1.0]], dtype=np.float32))

    def test_invalid_empty_channels_ranges_and_block_sizes_are_rejected(self) -> None:
        invalid_images = (
            np.empty((0, 4), dtype=np.uint8),
            np.zeros((4, 4, 4), dtype=np.uint8),
            np.array([[np.nan]], dtype=np.float32),
            np.array([[-1.0, 2.0]], dtype=np.float32),
            np.array([[1, 2]], dtype=np.int16),
            np.array([[1, 2]], dtype=object),
        )
        for image in invalid_images:
            with self.subTest(shape=image.shape, dtype=image.dtype), self.assertRaises(ValueError):
                prepare_image(image)
        with self.assertRaises(ValueError):
            prepare_image(np.zeros((2, 2), dtype=np.uint8), block_size=0)
        with self.assertRaises(ValueError):
            prepare_image(np.array([[0, 5]], dtype=np.uint8), value_range=(0, 4))

    def test_malformed_and_empty_file_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            empty = Path(temporary) / "empty.png"
            empty.write_bytes(b"")
            malformed = Path(temporary) / "malformed.jpg"
            malformed.write_bytes(b"not an image")
            with self.assertRaises(ImageInputError):
                load_image(empty)
            with self.assertRaises(ImageInputError):
                load_image(malformed)

    def test_oversized_array_and_file_are_rejected_before_processing(self) -> None:
        # Strides make this logically large input without allocating a large image buffer.
        one_pixel = np.zeros((1,), dtype=np.uint8)
        oversized_array = np.lib.stride_tricks.as_strided(
            one_pixel, shape=(11, 10), strides=(0, 0), writeable=False
        )
        with self.assertRaisesRegex(ImageSizeLimitError, "110 pixels"):
            prepare_image(oversized_array, max_image_pixels=100)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oversized.png"
            Image.new("RGB", (20, 20)).save(path)
            with self.assertRaisesRegex(ImageSizeLimitError, "400 pixels"):
                load_image(path, max_image_pixels=100)


if __name__ == "__main__":
    unittest.main()
