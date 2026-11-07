"""Tests for the uploaded-image non-recoverable application service."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.application.non_recoverable_service import (
    NonRecoverableServiceError,
    process_non_recoverable_image,
)
from src.non_recoverable.image_input import ImageInputError


class NonRecoverableServiceTest(unittest.TestCase):
    def encode_png(self, pixels: np.ndarray) -> bytes:
        output = BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        return output.getvalue()

    def process(self, pixels: np.ndarray, **overrides):
        arguments = {
            "quantization_levels": 8,
            "smoothing_strength": 0.0,
            "block_size": 4,
        }
        arguments.update(overrides)
        return process_non_recoverable_image(self.encode_png(pixels), **arguments)

    def test_rgb_upload_returns_png_previews_features_metrics_and_settings(self) -> None:
        pixels = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
        result = self.process(pixels)

        self.assertEqual(result.dimensions, (7, 9))
        self.assertEqual(result.channels, 3)
        self.assertEqual(result.original_image.shape, (7, 9, 3))
        self.assertEqual(result.transformed_image.shape, (7, 9, 3))
        self.assertEqual(result.original_image.dtype, np.float32)
        self.assertEqual(result.transformed_image.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(result.transformed_image)))
        self.assertGreaterEqual(float(result.transformed_image.min()), 0.0)
        self.assertLessEqual(float(result.transformed_image.max()), 1.0)
        self.assertEqual(result.mime_type, "image/png")
        self.assertEqual(result.image_format, "PNG")
        with Image.open(BytesIO(result.transformed_image_bytes)) as output:
            self.assertEqual(output.size, (9, 7))
            self.assertEqual(output.mode, "RGB")
        self.assertEqual(result.feature_summary["block_count"], 6)
        self.assertEqual(result.feature_summary["block_rows"], 2)
        self.assertEqual(result.feature_summary["block_columns"], 3)
        self.assertEqual(result.transformation_settings["quantization_levels"]["mode"], "global_uniform")
        self.assertEqual(result.transformation_settings["smoothing_strength"]["value"], 0.0)
        self.assertGreaterEqual(result.processing_time_seconds, 0.0)
        self.assertEqual(set(result.metrics), {"mae", "psnr_db", "ssim"})
        self.assertGreaterEqual(result.metrics["mae"], 0.0)
        self.assertGreater(result.metrics["psnr_db"], 0.0)
        self.assertIsNotNone(result.metrics["ssim"])
        self.assertTrue(result.warnings)

    def test_grayscale_and_partial_edge_blocks_are_supported(self) -> None:
        pixels = np.arange(5 * 6, dtype=np.uint8).reshape(5, 6)
        result = self.process(pixels, grayscale_image=True)

        self.assertEqual(result.dimensions, (5, 6))
        self.assertEqual(result.channels, 1)
        self.assertEqual(result.transformed_image.shape, (5, 6))
        self.assertEqual(result.feature_summary["block_count"], 4)
        self.assertEqual(result.feature_summary["valid_source_pixel_count"], pixels.size)
        with Image.open(BytesIO(result.transformed_image_bytes)) as output:
            self.assertEqual(output.size, (6, 5))
            self.assertEqual(output.mode, "L")

    def test_per_block_settings_are_accepted_and_recorded(self) -> None:
        pixels = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
        result = self.process(
            pixels,
            quantization_levels=[2, 4, 8, 16],
            smoothing_strength=np.asarray([0.0, 0.1, 0.2, 0.3]),
        )

        self.assertEqual(result.transformation_settings["quantization_levels"], {
            "mode": "per_block", "values": [2, 4, 8, 16]
        })
        self.assertEqual(result.transformation_settings["smoothing_strength"]["mode"], "per_block")
        self.assertIn("block-boundary", " ".join(result.warnings))

    def test_invalid_image_bytes_and_empty_inputs_are_rejected(self) -> None:
        for invalid in (b"not an image", b""):
            with self.subTest(invalid=invalid), self.assertRaises((ImageInputError, ValueError)):
                process_non_recoverable_image(
                    invalid, quantization_levels=8, smoothing_strength=0.0
                )

    def test_invalid_global_and_per_block_parameters_are_rejected(self) -> None:
        pixels = np.zeros((5, 6, 3), dtype=np.uint8)
        invalid_settings = (
            {"quantization_levels": 1},
            {"quantization_levels": [2]},
            {"smoothing_strength": float("nan")},
            {"smoothing_strength": 101.0},
            {"smoothing_strength": [0.0]},
        )
        for setting in invalid_settings:
            arguments = {"quantization_levels": 8, "smoothing_strength": 0.0}
            arguments.update(setting)
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                process_non_recoverable_image(self.encode_png(pixels), **arguments)

    def test_output_is_deterministic_for_identical_inputs_and_settings(self) -> None:
        pixels = np.random.default_rng(7).integers(0, 256, size=(9, 11, 3), dtype=np.uint8)
        first = self.process(pixels, quantization_levels=16, smoothing_strength=0.8)
        second = self.process(pixels, quantization_levels=16, smoothing_strength=0.8)

        np.testing.assert_array_equal(first.transformed_image, second.transformed_image)
        self.assertEqual(first.transformed_image_bytes, second.transformed_image_bytes)
        self.assertEqual(first.metrics, second.metrics)

    def test_safe_temporary_input_file_is_accepted(self) -> None:
        pixels = np.zeros((4, 4, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.png"
            path.write_bytes(self.encode_png(pixels))
            result = process_non_recoverable_image(
                path, quantization_levels=8, smoothing_strength=0.0
            )
        self.assertEqual(result.dimensions, (4, 4))

    def test_transform_processing_failure_is_reported(self) -> None:
        pixels = np.zeros((4, 4, 3), dtype=np.uint8)
        with patch(
            "src.application.non_recoverable_service.adaptive_combined_transformation",
            side_effect=RuntimeError("synthetic transform failure"),
        ), self.assertRaisesRegex(NonRecoverableServiceError, "synthetic transform failure"):
            self.process(pixels)


if __name__ == "__main__":
    unittest.main()
