"""Authentication and artifact-binding checks using temporary v2 bundles."""
from __future__ import annotations

import hashlib
import io
import json
import math
import secrets
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.recoverable.authentication import (
    METADATA_HMAC_FIELD,
    calculate_bytes_sha256,
    calculate_file_sha256,
    canonicalize_metadata,
    derive_authentication_key,
    generate_metadata_hmac,
    verify_metadata_hmac,
)
from src.recoverable.metadata_verifier import MetadataVerificationError
from src.recoverable.metadata_verifier import source_image_sha256, verify_recovery_metadata
from src.recoverable.recovery_engine import recover_image, run_recovery
from tests.recovery_test_support import create_recovery_fixture


class RecoveryAuthenticationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.paths = create_recovery_fixture(cls.root / "bundle_a")
        cls.pristine = {
            name: cls.paths[name].read_bytes()
            for name in ("key", "permutation", "features", "jaya", "metadata", "perturbed", "reference")
        }

    @classmethod
    def tearDownClass(cls) -> None:
        from tests.recovery_test_support import close_test_logger

        close_test_logger()
        cls.temporary.cleanup()

    def tearDown(self) -> None:
        from tests.recovery_test_support import close_test_logger

        close_test_logger()

    def setUp(self) -> None:
        for name, content in self.pristine.items():
            self.paths[name].write_bytes(content)

    def run_recovery(self, **overrides):
        arguments = {
            "key_path": self.paths["key"],
            "permutation_path": self.paths["permutation"],
            "metadata_path": self.paths["metadata"],
            "perturbed_path": self.paths["perturbed"],
            "jaya_path": self.paths["jaya"],
            "feature_matrix_path": self.paths["features"],
            "reference_path": self.paths["reference"],
            "report_path": self.paths["report"],
            "output_dir": self.paths["recovery_images"],
            "log_path": self.paths["log"],
        }
        arguments.update(overrides)
        return run_recovery(**arguments)

    def test_identical_metadata_and_key_order_produce_identical_hmac(self) -> None:
        key = secrets.token_bytes(16)
        first = {"z": 1, "nested": {"b": 2, "a": 3}}
        second = {"nested": {"a": 3, "b": 2}, "z": 1}
        self.assertEqual(canonicalize_metadata(first), canonicalize_metadata(second))
        self.assertEqual(generate_metadata_hmac(first, key), generate_metadata_hmac(second, key))
        self.assertEqual(generate_metadata_hmac(first, key), generate_metadata_hmac(first, key))

    def test_hmac_field_is_excluded_from_canonical_metadata(self) -> None:
        unsigned = {"run_id": "run-1", "schema_version": 2}
        tagged = dict(unsigned, **{METADATA_HMAC_FIELD: "placeholder"})
        self.assertEqual(canonicalize_metadata(unsigned), canonicalize_metadata(tagged))

    def test_metadata_modification_and_wrong_key_are_rejected(self) -> None:
        metadata = json.loads(self.paths["metadata"].read_text(encoding="utf-8"))
        metadata["source_index"] = 1
        with self.assertRaises(ValueError):
            verify_metadata_hmac(metadata, self.paths["key"].read_bytes())
        pristine_metadata = json.loads(self.pristine["metadata"])
        with self.assertRaises(ValueError):
            verify_metadata_hmac(pristine_metadata, secrets.token_bytes(16))

    def test_invalid_metadata_nested_keys_and_nonfinite_values_are_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonicalize_metadata({"nested": {"value": value}})
        with self.assertRaisesRegex(ValueError, "Non-string dictionary key"):
            canonicalize_metadata({"nested": {1: "invalid"}})
        with self.assertRaises(ValueError):
            canonicalize_metadata({"unsupported": object()})

    def test_missing_and_malformed_hmac_are_rejected(self) -> None:
        key = self.paths["key"].read_bytes()
        with self.assertRaisesRegex(ValueError, "missing or malformed"):
            verify_metadata_hmac({"schema_version": 2}, key)
        for tag in (None, "", "x" * 64, "0" * 63):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, "missing or malformed"):
                verify_metadata_hmac({METADATA_HMAC_FIELD: tag}, key)

    def test_invalid_key_types_and_lengths_are_rejected(self) -> None:
        for key in (b"short", b"x" * 17, bytearray(16), "not-bytes"):
            with self.subTest(key_type=type(key).__name__), self.assertRaises(ValueError):
                derive_authentication_key(key)

    def test_file_hashes_support_empty_and_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            empty = Path(temporary) / "empty.bin"
            empty.write_bytes(b"")
            self.assertEqual(calculate_file_sha256(empty), hashlib.sha256(b"").hexdigest())
            self.assertEqual(calculate_bytes_sha256(b""), hashlib.sha256(b"").hexdigest())
            with self.assertRaises(FileNotFoundError):
                calculate_file_sha256(Path(temporary) / "missing.bin")

    def test_substituted_perturbed_image_is_rejected_before_recovery(self) -> None:
        self.paths["perturbed"].write_bytes(b"substituted image bytes")
        original_load = np.load
        with patch("src.recoverable.recovery_engine.np.load", wraps=original_load) as numpy_load:
            with self.assertRaises(MetadataVerificationError):
                self.run_recovery()
        perturbed_loads = [
            call for call in numpy_load.call_args_list
            if call.args and isinstance(call.args[0], io.BytesIO)
        ]
        self.assertEqual(perturbed_loads, [])

    def test_modified_permutation_map_is_rejected(self) -> None:
        self.paths["permutation"].write_bytes(self.pristine["permutation"] + b" ")
        with self.assertRaises(MetadataVerificationError):
            self.run_recovery()

    def test_modified_jaya_output_is_rejected(self) -> None:
        changed = json.loads(self.pristine["jaya"])
        changed["optimized_beta"][0] += .01
        self.paths["jaya"].write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaises(MetadataVerificationError):
            self.run_recovery()

    def test_modified_feature_matrix_is_rejected(self) -> None:
        changed = json.loads(self.pristine["features"])
        changed["samples"][0]["features"][0]["variance"] += .01
        self.paths["features"].write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaises(MetadataVerificationError):
            self.run_recovery()

    def test_source_image_mismatch_is_rejected(self) -> None:
        dataset = np.load(io.BytesIO(self.pristine["reference"]), allow_pickle=False)
        dataset[0, 0, 0, 0] = np.float32(1.0 - dataset[0, 0, 0, 0])
        buffer = io.BytesIO()
        np.save(buffer, dataset, allow_pickle=False)
        self.paths["reference"].write_bytes(buffer.getvalue())
        with self.assertRaises(MetadataVerificationError):
            self.run_recovery()

    def test_artifacts_from_different_runs_cannot_be_mixed(self) -> None:
        second = create_recovery_fixture(self.root / "bundle_b", run_id="second-run-v2")
        self.paths["permutation"].write_bytes(second["permutation"].read_bytes())
        with self.assertRaises(MetadataVerificationError):
            self.run_recovery()

    def test_expected_run_id_mismatch_is_rejected(self) -> None:
        with self.assertRaises(MetadataVerificationError):
            self.run_recovery(expected_run_id="trusted-other-run")

    def test_recomputed_hmac_does_not_hide_inconsistent_alpha_snapshot(self) -> None:
        key = self.paths["key"].read_bytes()
        metadata = json.loads(self.pristine["metadata"])
        metadata["alpha_snapshot"][0] += 0.01
        metadata["metadata_hmac"] = generate_metadata_hmac(metadata, key)

        permutation_bytes = self.pristine["permutation"]
        feature_bytes = self.pristine["features"]
        jaya_bytes = self.pristine["jaya"]
        perturbed_bytes = self.pristine["perturbed"]
        mappings = json.loads(permutation_bytes)["mappings"]
        feature_matrix = json.loads(feature_bytes)
        jaya = json.loads(jaya_bytes)
        source = np.load(io.BytesIO(self.pristine["reference"]), allow_pickle=False)[0]
        artifact_hashes = {
            "perturbed_image": calculate_bytes_sha256(perturbed_bytes),
            "permutation_map": calculate_bytes_sha256(permutation_bytes),
            "jaya_output": calculate_bytes_sha256(jaya_bytes),
            "feature_matrix": calculate_bytes_sha256(feature_bytes),
            "source_image": source_image_sha256(source),
        }
        with self.assertRaisesRegex(MetadataVerificationError, "Alpha snapshot"):
            verify_recovery_metadata(
                metadata, key, mappings, jaya, feature_matrix, artifact_hashes,
                source, expected_run_id="test-run-v2", log_path=self.paths["log"],
            )

    def test_legacy_metadata_is_rejected(self) -> None:
        legacy = json.loads(self.pristine["metadata"])
        legacy.pop("schema_version")
        legacy.pop("metadata_hmac")
        legacy["metadata_signature"] = "0" * 64
        self.paths["metadata"].write_text(json.dumps(legacy), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Legacy unauthenticated"):
            self.run_recovery()

    def test_authenticated_recovery_is_pixel_perfect(self) -> None:
        key = self.paths["key"].read_bytes()
        metadata = json.loads(self.paths["metadata"].read_bytes())
        permutation_bytes = self.paths["permutation"].read_bytes()
        feature_bytes = self.paths["features"].read_bytes()
        jaya_bytes = self.paths["jaya"].read_bytes()
        perturbed_bytes = self.paths["perturbed"].read_bytes()
        source_dataset_bytes = self.paths["reference"].read_bytes()
        mapping = json.loads(permutation_bytes)["mappings"]
        feature_matrix = json.loads(feature_bytes)
        jaya = json.loads(jaya_bytes)
        source = np.load(io.BytesIO(source_dataset_bytes), allow_pickle=False)[0]
        artifact_hashes = {
            "perturbed_image": calculate_bytes_sha256(perturbed_bytes),
            "permutation_map": calculate_bytes_sha256(permutation_bytes),
            "jaya_output": calculate_bytes_sha256(jaya_bytes),
            "feature_matrix": calculate_bytes_sha256(feature_bytes),
            "source_image": source_image_sha256(source),
        }
        self.assertTrue(verify_recovery_metadata(
            metadata, key, mapping, jaya, feature_matrix, artifact_hashes, source,
            expected_run_id="test-run-v2", log_path=self.paths["log"],
        ))
        perturbed = np.load(io.BytesIO(perturbed_bytes), allow_pickle=False)
        recovered_bytes = recover_image(perturbed, key, metadata["alpha_snapshot"], mapping)
        recovered = recovered_bytes.astype(np.float32) / np.float32(255)
        np.testing.assert_array_equal(recovered, source)

    def test_complete_recovery_runner_generates_outputs_and_is_pixel_perfect(self) -> None:
        result = self.run_recovery(expected_run_id="test-run-v2")
        source = np.load(self.paths["reference"], allow_pickle=False)[0]
        recovered = result["recovered_image"].astype(np.float32) / np.float32(255)
        np.testing.assert_array_equal(recovered, source)
        self.assertTrue(result["metadata_verified"])
        self.assertEqual(result["recovery_rate"], 100.0)
        self.assertTrue(self.paths["report"].is_file())
        report = json.loads(self.paths["report"].read_text(encoding="utf-8"))
        self.assertEqual(report["recovery_rate"], 100.0)
        self.assertTrue(all(Path(path).is_file() for path in result["image_paths"]))


if __name__ == "__main__":
    unittest.main()
