"""Tests for secure recoverable-pipeline key generation and validation."""
import json
from pathlib import Path
import tempfile
import unittest

from src.recoverable.key_generator import KeyExistsError, generate_key
from src.recoverable.key_loader import load_key
from src.recoverable.key_validator import validate_key


class KeyGenerationTest(unittest.TestCase):
    def test_generate_load_validate_and_refuse_unconfirmed_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key_path = root / "secret_key.bin"
            metadata_path = root / "key_metadata.json"
            generated = generate_key(key_path, metadata_path)
            self.assertTrue(key_path.is_file())
            self.assertEqual(len(generated), 16)
            self.assertEqual(load_key(key_path), generated)
            self.assertTrue(validate_key(key_path))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["algorithm"], "secrets")
            self.assertEqual(metadata["key_length"], 128)
            self.assertTrue(metadata["created_at"])
            with self.assertRaises(KeyExistsError):
                generate_key(key_path, metadata_path)
            self.assertEqual(load_key(key_path), generated)


if __name__ == "__main__":
    unittest.main()
