"""Tests for the Streamlit UI's validation and service adapter helpers."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.application.ui_helpers import (
    RecoverableConfigurationError,
    UploadValidationError,
    call_non_recoverable_transform,
    call_recoverable_recovery,
    call_recoverable_transform,
    inspect_image_upload,
    recoverable_service_configuration,
    safe_service_error,
    validate_recoverable_upload,
)


def _encoded(array: np.ndarray, format_name: str, mode: str | None = None) -> bytes:
    stream = BytesIO()
    Image.fromarray(array, mode=mode).save(stream, format=format_name)
    return stream.getvalue()


class StreamlitUiHelperTest(unittest.TestCase):
    def test_upload_validation_for_rgb_png_and_grayscale_jpeg(self) -> None:
        rgb = _encoded(np.zeros((28, 28, 3), dtype=np.uint8), "PNG", "RGB")
        info = inspect_image_upload(rgb, "image.png")
        self.assertEqual((info.height, info.width), (28, 28))
        self.assertEqual(info.color_mode, "RGB")
        self.assertFalse(info.is_grayscale)
        validate_recoverable_upload(info)

        gray = _encoded(np.zeros((11, 17), dtype=np.uint8), "JPEG", "L")
        gray_info = inspect_image_upload(gray, "gray.jpeg")
        self.assertTrue(gray_info.is_grayscale)
        self.assertEqual((gray_info.height, gray_info.width), (11, 17))

    def test_malformed_empty_unsupported_and_mismatched_uploads_are_rejected(self) -> None:
        for data, name in ((b"", "empty.png"), (b"not an image", "broken.jpg"), (b"x", "file.gif")):
            with self.subTest(name=name), self.assertRaises(UploadValidationError):
                inspect_image_upload(data, name)
        png = _encoded(np.zeros((2, 2, 3), dtype=np.uint8), "PNG", "RGB")
        with self.assertRaises(UploadValidationError):
            inspect_image_upload(png, "mislabelled.jpg")

    def test_recoverable_geometry_and_grayscale_validation(self) -> None:
        wrong_size = inspect_image_upload(
            _encoded(np.zeros((28, 29, 3), dtype=np.uint8), "PNG", "RGB"), "x.png"
        )
        with self.assertRaisesRegex(UploadValidationError, "28"):
            validate_recoverable_upload(wrong_size)
        grayscale = inspect_image_upload(_encoded(np.zeros((28, 28), dtype=np.uint8), "PNG", "L"), "x.png")
        with self.assertRaisesRegex(UploadValidationError, "grayscale"):
            validate_recoverable_upload(grayscale)

        palette_image = Image.new("P", (28, 28), color=40)
        gray_palette = []
        for value in range(256):
            gray_palette.extend((value, value, value))
        palette_image.putpalette(gray_palette)
        palette_bytes = BytesIO()
        palette_image.save(palette_bytes, format="PNG")
        palette_info = inspect_image_upload(palette_bytes.getvalue(), "palette.png")
        self.assertTrue(palette_info.is_grayscale)
        with self.assertRaisesRegex(UploadValidationError, "grayscale"):
            validate_recoverable_upload(palette_info)

    def test_recovery_configuration_requires_explicit_server_paths(self) -> None:
        with self.assertRaises(RecoverableConfigurationError):
            recoverable_service_configuration({})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "temporary-key.bin"
            key.write_bytes(b"test-only-key")
            storage, configured_key = recoverable_service_configuration(
                {
                    "PRIVACYPRESERVATION_PRIVATE_STORAGE": str(root / "private-runs"),
                    "PRIVACYPRESERVATION_KEY_PATH": str(key),
                }
            )
            self.assertEqual(storage, (root / "private-runs").resolve())
            self.assertEqual(configured_key, key.resolve())
        with self.assertRaises(RecoverableConfigurationError):
            recoverable_service_configuration(
                {"PRIVACYPRESERVATION_PRIVATE_STORAGE": "relative", "PRIVACYPRESERVATION_KEY_PATH": "relative-key"}
            )
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "temporary-key.bin"
            key.write_bytes(b"test-only-key")
            project_storage = Path(__file__).resolve().parents[1] / ".private-test-storage"
            with self.assertRaises(RecoverableConfigurationError):
                recoverable_service_configuration(
                    {
                        "PRIVACYPRESERVATION_PRIVATE_STORAGE": str(project_storage),
                        "PRIVACYPRESERVATION_KEY_PATH": str(key),
                    }
                )

    def test_recoverable_service_invocation_and_run_id_validation(self) -> None:
        content = _encoded(np.zeros((28, 28, 3), dtype=np.uint8), "PNG", "RGB")
        info = inspect_image_upload(content, "image.png")
        service = Mock()
        service.transform.return_value = "result"
        self.assertEqual(call_recoverable_transform(service, content, info), "result")
        service.transform.assert_called_once_with(content)

        run_id = uuid.uuid4().hex
        service.recover.return_value = "recovered"
        self.assertEqual(call_recoverable_recovery(service, run_id), "recovered")
        service.recover.assert_called_once_with(run_id)
        with self.assertRaises(ValueError):
            call_recoverable_recovery(service, "../not-a-run")

    def test_non_recoverable_service_invocation_and_parameter_validation(self) -> None:
        content = _encoded(np.zeros((8, 9), dtype=np.uint8), "PNG", "L")
        sentinel = object()
        with patch("src.application.ui_helpers.process_non_recoverable_image", return_value=sentinel) as process:
            result = call_non_recoverable_transform(
                content, quantization_levels=16, smoothing_strength=0.5, grayscale=True
            )
        self.assertIs(result, sentinel)
        process.assert_called_once_with(
            content,
            quantization_levels=16,
            smoothing_strength=0.5,
            grayscale_image=True,
        )
        with patch("src.application.ui_helpers.process_non_recoverable_image") as process:
            with self.assertRaises(ValueError):
                call_non_recoverable_transform(
                    content, quantization_levels=1, smoothing_strength=0.5, grayscale=True
                )
            process.assert_not_called()

    def test_user_facing_service_error_does_not_disclose_exception_details(self) -> None:
        message = safe_service_error(
            RuntimeError("C:\\private\\secret_key.bin contents: NEVER-DISPLAY"), mode="recoverable"
        )
        self.assertNotIn("secret_key.bin", message)
        self.assertNotIn("NEVER-DISPLAY", message)

    def test_failed_service_call_is_sanitized_for_the_ui(self) -> None:
        content = _encoded(np.zeros((28, 28, 3), dtype=np.uint8), "PNG", "RGB")
        info = inspect_image_upload(content, "image.png")
        failing_service = Mock()
        failing_service.transform.side_effect = RuntimeError("private path and key details")
        with self.assertRaises(RuntimeError) as caught:
            call_recoverable_transform(failing_service, content, info)
        message = safe_service_error(caught.exception, mode="recoverable")
        self.assertNotIn("private path", message)
        self.assertNotIn("key details", message)

    def test_streamlit_app_loads_and_mode_selector_switches_sections(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        self.assertFalse(app.exception)
        self.assertIn("Privacy Preservation System", [item.value for item in app.title])
        self.assertEqual(len(app.radio), 1)
        app.radio[0].set_value("Non-Recoverable Privacy").run()
        self.assertFalse(app.exception)
        self.assertIn("Non-Recoverable Privacy", [item.value for item in app.title])


if __name__ == "__main__":
    unittest.main()
