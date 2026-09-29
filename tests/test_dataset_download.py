"""Integration test for PathMNIST download, loading, validation, and reporting."""

from pathlib import Path
import unittest

from src.preprocessing.data_loader import PathMNISTSplits, load_pathmnist
from src.preprocessing.dataset_validator import validate_pathmnist


class PathMNISTDownloadIntegrationTest(unittest.TestCase):
    """Exercise official PathMNIST download and integrity verification."""

    def test_download_load_validate_and_report(self) -> None:
        """Download to the canonical raw folder and check generated metadata."""
        project_root = Path(__file__).resolve().parents[1]
        raw_data_dir = project_root / "data" / "raw" / "PathMNIST"
        report_path = project_root / "results" / "metrics" / "dataset_report.json"
        log_path = project_root / "results" / "logs" / "dataset_download.log"

        splits = load_pathmnist(data_dir=raw_data_dir, log_path=log_path)
        report = validate_pathmnist(
            splits,
            report_path=report_path,
            log_path=log_path,
        )

        self.assertIsInstance(splits, PathMNISTSplits)
        self.assertEqual(splits.data_dir, raw_data_dir)
        self.assertTrue(splits.archive_path.is_file())
        self.assertEqual(len(splits.train), 89996)
        self.assertEqual(len(splits.validation), 10004)
        self.assertEqual(len(splits.test), 7180)
        self.assertEqual(report["status"], "verified")
        self.assertTrue(report_path.is_file())
        self.assertTrue(log_path.is_file())
        self.assertTrue(report["package_version"])
        self.assertTrue(report["download_timestamp"])
        self.assertEqual(len(str(report["sample_hash"])), 64)
