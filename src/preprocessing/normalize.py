"""Normalize PathMNIST images and validate them before saving."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.preprocessing.data_loader import DEFAULT_LOG_PATH, PROJECT_ROOT, get_dataset_logger

EXPECTED_COUNTS = {"train": 89996, "validation": 10004, "test": 7180}
SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}
IMAGE_SHAPE = (28, 28, 3)
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "results" / "metrics" / "normalization_report.json"


class NormalizationError(ValueError):
    """Raised when normalized data fails a preservation check."""


def prepare_processed_dataset(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict[str, object]:
    """Load official raw splits, normalize and save each verified split."""
    from src.preprocessing.data_loader import load_pathmnist

    started = time.perf_counter()
    logger = get_dataset_logger(log_path)
    logger.info("Starting PathMNIST normalization.")
    splits = load_pathmnist(log_path=log_path)
    split_data = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    for name, dataset in split_data.items():
        raw_images = np.asarray(dataset.imgs)
        raw_labels = np.asarray(dataset.labels)
        if raw_images.dtype != np.uint8 or raw_images.shape != (EXPECTED_COUNTS[name], *IMAGE_SHAPE):
            raise NormalizationError(f"Unexpected raw {name} image dtype/shape: {raw_images.dtype} {raw_images.shape}")
        if raw_labels.shape[0] != EXPECTED_COUNTS[name]:
            raise NormalizationError(f"Unexpected raw {name} label count: {raw_labels.shape[0]}")

        images = raw_images.astype(np.float32) / np.float32(255.0)
        if images.dtype != np.float32:
            raise NormalizationError(f"{name} dtype is {images.dtype}, expected float32")
        if images.shape != raw_images.shape:
            raise NormalizationError(f"{name} image shape changed from {raw_images.shape} to {images.shape}")
        if not np.array_equal(raw_labels, np.asarray(dataset.labels)):
            raise NormalizationError(f"{name} labels changed during normalization")
        minimum, maximum = float(images.min()), float(images.max())
        if minimum != 0.0 or maximum != 1.0:
            raise NormalizationError(f"{name} pixel range is [{minimum}, {maximum}], expected [0.0, 1.0]")
        destination = Path(processed_dir) / name
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "images.npy", images)
        np.save(destination / "labels.npy", raw_labels)
        metadata = {
            "split": name,
            "images": EXPECTED_COUNTS[name],
            "shape": "28x28x3",
            "dtype": "float32",
            "normalization": "pixel/255.0",
        }
        (destination / "metadata.json").write_text(
            __import__("json").dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Normalized %s split to %s", name, destination)

    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "train_count": EXPECTED_COUNTS["train"],
        "validation_count": EXPECTED_COUNTS["validation"],
        "test_count": EXPECTED_COUNTS["test"],
        "dtype": "float32",
        "pixel_range": [0.0, 1.0],
        "shape": "28x28x3",
        "execution_time_seconds": round(elapsed, 3),
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(__import__("json").dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("PathMNIST normalization completed in %.3f seconds.", elapsed)
    return report
