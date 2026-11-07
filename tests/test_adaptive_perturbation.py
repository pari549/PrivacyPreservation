"""Integration checks for the v2 adaptive perturbation producer."""
import json
import unittest

import numpy as np

from src.recoverable.authentication import verify_metadata_hmac
from src.recoverable.perturbation_validator import validate_perturbation
from tests.recovery_test_support import create_recovery_fixture, temporary_test_directory


class AdaptivePerturbationTest(unittest.TestCase):
    def test_v2_bundle_and_deterministic_perturbation(self) -> None:
        paths = create_recovery_fixture(temporary_test_directory(self))
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        self.assertEqual(metadata["schema_version"], 2)
        self.assertTrue(metadata["run_id"])
        self.assertTrue(verify_metadata_hmac(metadata, paths["key"].read_bytes()))
        alphas = metadata["alpha_snapshot"]
        self.assertEqual(len(alphas), 49)
        self.assertGreaterEqual(min(alphas), .10)
        self.assertLessEqual(max(alphas), .70)
        perturbed = np.load(paths["perturbed"], allow_pickle=False)
        original = np.load(paths["reference"], allow_pickle=False)[0]
        mapping = json.loads(paths["permutation"].read_text(encoding="utf-8"))["mappings"]
        self.assertTrue(validate_perturbation(
            original, perturbed, paths["key"].read_bytes(), alphas, mapping, metadata,
            log_path=paths["log"],
        ))


if __name__ == "__main__":
    unittest.main()
