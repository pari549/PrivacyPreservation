"""Integration checks for normalized PathMNIST output."""

import unittest
from pathlib import Path

import numpy as np

from src.preprocessing.normalize import prepare_processed_dataset
from src.preprocessing.processed_dataset import load_processed_split


class NormalizationTest(unittest.TestCase):
    def test_normalized_splits_and_metadata(self) -> None:
        root = Path(__file__).resolve().parents[1]
        prepare_processed_dataset(
            processed_dir=root / "data" / "processed",
            report_path=root / "results" / "metrics" / "normalization_report.json",
            log_path=root / "results" / "logs" / "dataset_download.log",
        )
        for name, count in (("train", 89996), ("validation", 10004), ("test", 7180)):
            images, labels, metadata = load_processed_split(name, root / "data" / "processed")
            self.assertEqual(images.dtype, np.float32)
            self.assertEqual(images.shape, (count, 28, 28, 3))
            self.assertEqual(float(images.min()), 0.0)
            self.assertEqual(float(images.max()), 1.0)
            self.assertEqual(labels.shape[0], count)
            self.assertEqual(metadata["split"], name)
            self.assertEqual(metadata["images"], count)
            self.assertEqual(metadata["shape"], "28x28x3")
            self.assertEqual(metadata["dtype"], "float32")
            self.assertEqual(metadata["normalization"], "pixel/255.0")
            self.assertTrue((root / "data" / "processed" / name / "metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
