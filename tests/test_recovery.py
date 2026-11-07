"""Integration checks for v2 authenticated pixel-perfect recovery."""
import io
import json
import unittest

import numpy as np

from src.recoverable.authentication import calculate_bytes_sha256
from src.recoverable.metadata_verifier import source_image_sha256, verify_recovery_metadata
from src.recoverable.recovery_engine import recover_image
from tests.recovery_test_support import create_recovery_fixture, temporary_test_directory


class RecoveryTest(unittest.TestCase):
    def test_authenticated_pixel_perfect_recovery(self) -> None:
        paths = create_recovery_fixture(temporary_test_directory(self))
        key = paths["key"].read_bytes()
        metadata = json.loads(paths["metadata"].read_bytes())
        permutation_bytes = paths["permutation"].read_bytes()
        feature_bytes = paths["features"].read_bytes()
        jaya_bytes = paths["jaya"].read_bytes()
        perturbed_bytes = paths["perturbed"].read_bytes()
        source_dataset_bytes = paths["reference"].read_bytes()
        mapping = json.loads(permutation_bytes)["mappings"]
        features = json.loads(feature_bytes)
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
            metadata, key, mapping, jaya, features, artifact_hashes, source,
            expected_run_id="test-run-v2", log_path=paths["log"],
        ))
        perturbed = np.load(io.BytesIO(perturbed_bytes), allow_pickle=False)
        recovered = recover_image(perturbed, key, metadata["alpha_snapshot"], mapping)
        recovered_float = recovered.astype(np.float32) / np.float32(255)
        np.testing.assert_array_equal(recovered_float, source)


if __name__ == "__main__":
    unittest.main()
