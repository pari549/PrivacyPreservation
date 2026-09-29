"""Integration tests for deterministic adaptive reversible perturbation."""
import json
import unittest
from pathlib import Path
import numpy as np

from src.recoverable.adaptive_perturbation import run_adaptive_perturbation
from src.recoverable.key_loader import load_key
from src.recoverable.perturbation_validator import validate_perturbation
from src.recoverable.perturbation_validator import (
    PerturbationValidationError,
    verify_metadata_signature,
)


class AdaptivePerturbationTest(unittest.TestCase):
    def test_determinism_alpha_metadata_and_validation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        summary = run_adaptive_perturbation()
        metadata_path = root / "metadata" / "recovery_metadata.json"
        output_path = root / "data" / "recoverable" / "perturbed_image.npy"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertTrue(verify_metadata_signature(metadata))
        self.assertEqual(len(metadata["metadata_signature"]), 64)
        alphas = np.asarray(metadata["alpha_snapshot"], dtype=np.float64)
        self.assertEqual(alphas.shape, (49,))
        self.assertGreaterEqual(float(alphas.min()), .10)
        self.assertLessEqual(float(alphas.max()), .70)
        self.assertEqual(len(metadata["beta_snapshot"]), 49)
        self.assertEqual(set(metadata["jaya_weights"]), {"variance", "gradient", "entropy", "roi"})
        self.assertTrue(metadata["noise_seed"])
        self.assertTrue(metadata["permutation_checksum"])
        self.assertTrue(output_path.is_file())
        perturbed = np.load(output_path, allow_pickle=False)
        original = np.load(root / "data" / "processed" / "train" / "images.npy", mmap_mode="r", allow_pickle=False)[0]
        key = load_key(root / "keys" / "secret_key.bin")
        mapping = json.loads((root / "metadata" / "permutation_map.json").read_text(encoding="utf-8"))["mappings"]
        self.assertTrue(validate_perturbation(original, perturbed, key, alphas.tolist(), mapping, metadata))
        tampered_metadata = dict(metadata)
        tampered_metadata["beta_snapshot"] = list(metadata["beta_snapshot"])
        tampered_metadata["beta_snapshot"][0] += .01
        with self.assertRaises(PerturbationValidationError):
            validate_perturbation(original, perturbed, key, alphas.tolist(), mapping, tampered_metadata)
        self.assertEqual(summary["perturbed_blocks"], 49)
        self.assertTrue(summary["deterministic_replay"])
        self.assertTrue(all(Path(path).is_file() for path in summary["visualization_paths"]))


if __name__ == "__main__":
    unittest.main()
