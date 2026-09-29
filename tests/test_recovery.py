"""Integration tests for signed, deterministic Method 1 image recovery."""
import json
import unittest
from pathlib import Path
import numpy as np

from src.recoverable.key_loader import load_key
from src.recoverable.metadata_verifier import MetadataVerificationError, verify_recovery_metadata
from src.recoverable.recovery_engine import run_recovery


class RecoveryTest(unittest.TestCase):
    def test_pixel_perfect_recovery_metadata_and_report(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = run_recovery()
        recovered = result["recovered_image"]
        reference = np.load(root / "data" / "processed" / "train" / "images.npy", mmap_mode="r", allow_pickle=False)[0]
        recovered_float = recovered.astype(np.float32) / np.float32(255)
        np.testing.assert_array_equal(recovered_float, reference)
        metadata = json.loads((root / "metadata" / "recovery_metadata.json").read_text(encoding="utf-8"))
        mapping = json.loads((root / "metadata" / "permutation_map.json").read_text(encoding="utf-8"))["mappings"]
        jaya = json.loads((root / "results" / "metrics" / "jaya_results.json").read_text(encoding="utf-8"))
        self.assertTrue(verify_recovery_metadata(metadata, load_key(root / "keys" / "secret_key.bin"), mapping, jaya))
        self.assertTrue(result["metadata_verified"])
        self.assertEqual(result["mse"], 0.0)
        self.assertEqual(result["recovery_rate"], 100.0)
        self.assertGreaterEqual(result["ssim"], .999)
        report_path = root / "results" / "metrics" / "recovery_report.json"
        self.assertTrue(report_path.is_file())
        self.assertTrue(json.loads(report_path.read_text(encoding="utf-8"))["metadata_verified"])
        self.assertEqual(len(result["image_paths"]), 4)
        self.assertTrue(all(Path(path).is_file() for path in result["image_paths"]))

        tampered = dict(metadata)
        tampered["alpha_snapshot"] = list(metadata["alpha_snapshot"])
        tampered["alpha_snapshot"][0] += .001
        with self.assertRaises(MetadataVerificationError):
            verify_recovery_metadata(tampered, load_key(root / "keys" / "secret_key.bin"), mapping, jaya)


if __name__ == "__main__":
    unittest.main()
