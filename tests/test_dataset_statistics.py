"""Integration checks for PathMNIST visualizations and summary statistics."""

import json
import unittest
from pathlib import Path

from src.preprocessing.dataset_statistics import generate_dataset_statistics


class DatasetStatisticsTest(unittest.TestCase):
    def test_statistics_and_visualizations(self) -> None:
        root = Path(__file__).resolve().parents[1]
        preview = root / "results" / "images" / "dataset_preview"
        output = root / "results" / "metrics" / "dataset_statistics.json"
        stats = generate_dataset_statistics(
            processed_dir=root / "data" / "processed",
            preview_dir=preview,
            statistics_path=output,
            log_path=root / "results" / "logs" / "dataset_download.log",
        )
        self.assertTrue(output.is_file())
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), stats)
        for filename in ("sample_grid.png", "rgb_histogram.png", "class_distribution.png", "pixel_distribution.png"):
            self.assertTrue((preview / filename).is_file())
        pixel = stats["pixel_statistics"]
        self.assertEqual(pixel["min"], 0.0)
        self.assertEqual(pixel["max"], 1.0)
        self.assertTrue(0.0 < pixel["mean"] < 1.0)
        self.assertTrue(0.0 <= pixel["median"] <= 1.0)
        self.assertTrue(pixel["std"] > 0.0)
        self.assertEqual(sum(stats["class_distribution"].values()), 89996)


if __name__ == "__main__":
    unittest.main()
