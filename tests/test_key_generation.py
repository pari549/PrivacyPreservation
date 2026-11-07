"""Tests for secure recoverable-pipeline key generation and validation."""
import json
from pathlib import Path
import unittest

from src.recoverable.key_generator import KeyExistsError, generate_key
from src.recoverable.key_loader import load_key
from src.recoverable.key_validator import validate_key
from tests.recovery_test_support import temporary_test_directory


class KeyGenerationTest(unittest.TestCase):
    def test_generate_load_validate_and_refuse_unconfirmed_overwrite(self) -> None:
        root = temporary_test_directory(self)
        key_path = root / "test_secret_key.bin"
        metadata_path = root / "key_metadata.json"
        log_path = root / "key_test.log"
        generated = generate_key(key_path, metadata_path, log_path=log_path)
        self.assertTrue(key_path.is_file())
        self.assertEqual(len(generated), 16)
        self.assertEqual(load_key(key_path), generated)
        self.assertTrue(validate_key(key_path, log_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["algorithm"], "secrets")
        self.assertEqual(metadata["key_length"], 128)
        self.assertTrue(metadata["created_at"])
        with self.assertRaises(KeyExistsError):
            generate_key(key_path, metadata_path, log_path=log_path)
        self.assertEqual(load_key(key_path), generated)


if __name__ == "__main__":
    unittest.main()
