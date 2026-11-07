"""End-to-end service tests with temporary images and generated test keys only."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import secrets
import tempfile
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from src.application.recoverable_service import RecoverableImageService
from src.recoverable.authentication import generate_metadata_hmac


def _png(array: np.ndarray, mode: str | None = None) -> bytes:
    stream = BytesIO()
    Image.fromarray(array, mode=mode).save(stream, format="PNG")
    return stream.getvalue()


def _close_project_logger() -> None:
    logger = logging.getLogger("privacy_project.dataset_download")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


class RecoverableServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="recoverable-service-test-")
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(_close_project_logger)
        self.root = Path(self.temporary.name)
        self.key_path = self.root / "generated-test-key.bin"
        self.key_path.write_bytes(secrets.token_bytes(16))
        self.service = RecoverableImageService(
            self.root / "private-app-storage",
            key_path=self.key_path,
            jaya_population_size=2,
            jaya_iterations=1,
            jaya_seed=17,
        )
        rows, columns = np.indices((28, 28))
        self.original = np.stack((rows * 7, columns * 7, (rows + columns) * 4), axis=2).astype(np.uint8)
        self.upload = _png(self.original, "RGB")

    def _transform(self):
        return self.service.transform(self.upload)

    def test_valid_rgb_transform_and_exact_recovery(self) -> None:
        transformed = self._transform()
        self.assertEqual(transformed.metadata["dimensions"], [28, 28])
        self.assertEqual(transformed.metadata["block_count"], 49)
        self.assertEqual(transformed.mime_type, "image/png")
        self.assertNotIn("metadata_hmac", transformed.metadata)
        self.assertNotIn("source_image", transformed.metadata)
        transformed_image = np.asarray(Image.open(BytesIO(transformed.transformed_image_bytes)).convert("RGB"))
        self.assertEqual(transformed_image.shape, (28, 28, 3))
        recovered = self.service.recover(transformed.run_id)
        self.assertTrue(recovered.pixel_perfect)
        restored = np.asarray(Image.open(BytesIO(recovered.recovered_image_bytes)).convert("RGB"))
        np.testing.assert_array_equal(restored, self.original)

    def test_rejects_wrong_dimensions_grayscale_and_invalid_bytes(self) -> None:
        with self.assertRaises(ValueError):
            self.service.transform(_png(np.zeros((27, 28, 3), dtype=np.uint8), "RGB"))
        with self.assertRaises(ValueError):
            self.service.transform(_png(np.zeros((28, 28), dtype=np.uint8), "L"))
        with self.assertRaises(ValueError):
            self.service.transform(b"not an image")

    def test_invalid_key_is_rejected_without_generating_or_replacing_it(self) -> None:
        invalid_key_path = self.root / "invalid-test-key.bin"
        invalid_key_path.write_bytes(secrets.token_bytes(15))
        before = invalid_key_path.read_bytes()
        service = RecoverableImageService(
            self.root / "invalid-key-storage",
            key_path=invalid_key_path,
            jaya_population_size=2,
            jaya_iterations=1,
        )
        with self.assertRaises(Exception):
            service.transform(self.upload)
        self.assertEqual(invalid_key_path.read_bytes(), before)

    def test_tampered_artifact_is_rejected(self) -> None:
        transformed = self._transform()
        artifact = self.root / "private-app-storage" / "runs" / transformed.run_id / "jaya_results.json"
        artifact.write_bytes(artifact.read_bytes() + b" ")
        with self.assertRaises(Exception):
            self.service.recover(transformed.run_id)

    def test_authenticated_metadata_with_wrong_run_id_is_rejected(self) -> None:
        transformed = self._transform()
        run_dir = self.root / "private-app-storage" / "runs" / transformed.run_id
        metadata_path = run_dir / "recovery_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["run_id"] = "0" * 32
        metadata["metadata_hmac"] = generate_metadata_hmac(metadata, self.key_path.read_bytes())
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(Exception):
            self.service.recover(transformed.run_id)

    def test_unknown_run_id_and_path_traversal_are_rejected(self) -> None:
        for value in ("f" * 32, "../" + "a" * 29, "..", ""):
            with self.subTest(run_id=value), self.assertRaises((ValueError, KeyError)):
                self.service.recover(value)

    def test_runs_have_isolated_directories_and_no_default_key_path(self) -> None:
        first = self._transform()
        second = self.service.transform(_png(255 - self.original, "RGB"))
        self.assertNotEqual(first.run_id, second.run_id)
        runs_root = self.root / "private-app-storage" / "runs"
        self.assertEqual({path.name for path in runs_root.iterdir()}, {first.run_id, second.run_id})
        self.assertEqual(self.service._key_path, self.key_path.resolve())


if __name__ == "__main__":
    unittest.main()
