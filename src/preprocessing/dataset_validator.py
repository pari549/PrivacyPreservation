"""Validate PathMNIST integrity and write its verification report."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from medmnist import INFO

from src.preprocessing.data_loader import (
    DEFAULT_LOG_PATH,
    PROJECT_ROOT,
    PathMNISTSplits,
    get_dataset_logger,
)

EXPECTED_COUNTS: dict[str, int] = {
    "train": 89996,
    "validation": 10004,
    "test": 7180,
}
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
EXPECTED_CLASSES = 9
DEFAULT_REPORT_PATH = PROJECT_ROOT / "results" / "metrics" / "dataset_report.json"


class DatasetValidationError(ValueError):
    """Raised when a PathMNIST integrity requirement is not satisfied."""


class DatasetReportError(OSError):
    """Raised when the validated dataset report cannot be written."""


def _verify_official_checksum(archive_path: Path) -> None:
    """Compare the raw archive against the checksum supplied by MedMNIST.

    Args:
        archive_path: Path to the official MedMNIST NPZ archive.

    Raises:
        DatasetValidationError: If the archive is missing or its checksum fails.
    """
    if not archive_path.is_file():
        raise DatasetValidationError(f"Dataset archive does not exist: {archive_path}")

    expected_md5 = INFO["pathmnist"].get("MD5")
    if not expected_md5:
        raise DatasetValidationError("MedMNIST provides no PathMNIST MD5 value.")

    digest = hashlib.md5()
    try:
        with archive_path.open("rb") as archive_file:
            for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DatasetValidationError(
            f"Could not read PathMNIST archive {archive_path}: {error}"
        ) from error

    actual_md5 = digest.hexdigest()
    if actual_md5.lower() != str(expected_md5).lower():
        raise DatasetValidationError(
            f"PathMNIST checksum mismatch for {archive_path}: expected "
            f"{expected_md5}, received {actual_md5}."
        )


def _validate_split(
    split_name: str,
    dataset: Any,
    expected_count: int,
) -> np.ndarray:
    """Validate one split's sample count, image shape, and labels.

    Args:
        split_name: Human-readable split name.
        dataset: MedMNIST split exposing ``imgs`` and ``labels`` arrays.
        expected_count: Official sample count for the split.

    Returns:
        Flattened labels for the split.

    Raises:
        DatasetValidationError: If a split does not match its specification.
    """
    images = np.asarray(dataset.imgs)
    labels = np.asarray(dataset.labels)
    expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    if images.ndim != 4 or images.shape[1:] != expected_shape:
        raise DatasetValidationError(
            f"{split_name} images must have shape (N, 28, 28, 3); "
            f"received {images.shape}."
        )
    if images.shape[0] != expected_count:
        raise DatasetValidationError(
            f"{split_name} image count must be {expected_count}; "
            f"received {images.shape[0]}."
        )
    if labels.shape[0] != expected_count:
        raise DatasetValidationError(
            f"{split_name} label count must be {expected_count}; "
            f"received {labels.shape[0]}."
        )
    return labels.reshape(-1)


def _make_sample_hash(splits: PathMNISTSplits) -> str:
    """Hash deterministic representative image-label samples from each split.

    Args:
        splits: The three loaded PathMNIST splits.

    Returns:
        A SHA-256 hex digest based on first, middle, and last samples per split.
    """
    digest = hashlib.sha256()
    for split_name, dataset in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        images = np.asarray(dataset.imgs)
        labels = np.asarray(dataset.labels)
        indices = sorted({0, len(images) // 2, len(images) - 1})
        for index in indices:
            digest.update(split_name.encode("utf-8"))
            digest.update(index.to_bytes(length=8, byteorder="big", signed=False))
            digest.update(np.ascontiguousarray(images[index]).tobytes())
            digest.update(np.ascontiguousarray(labels[index]).tobytes())
    return digest.hexdigest()


def validate_pathmnist(
    splits: PathMNISTSplits,
    report_path: Path = DEFAULT_REPORT_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict[str, str | int]:
    """Verify PathMNIST and write a report only after all checks pass.

    Args:
        splits: Train, validation, and test objects returned by the loader.
        report_path: Destination JSON file for the verified summary.
        log_path: Destination for verification and error messages.

    Returns:
        Verified dataset information and its reproducibility fingerprint.

    Raises:
        DatasetValidationError: If archive or dataset properties fail.
        DatasetReportError: If the validated report cannot be saved.
    """
    logger = get_dataset_logger(log_path)
    logger.info("Starting PathMNIST integrity verification.")
    try:
        _verify_official_checksum(splits.archive_path)
        labels = np.concatenate(
            [
                _validate_split("train", splits.train, EXPECTED_COUNTS["train"]),
                _validate_split(
                    "validation", splits.validation, EXPECTED_COUNTS["validation"]
                ),
                _validate_split("test", splits.test, EXPECTED_COUNTS["test"]),
            ]
        )
        classes = np.unique(labels)
        if classes.size != EXPECTED_CLASSES:
            raise DatasetValidationError(
                f"PathMNIST must contain {EXPECTED_CLASSES} classes; "
                f"found {classes.size}."
            )
        if not np.array_equal(classes, np.arange(EXPECTED_CLASSES)):
            raise DatasetValidationError(
                "PathMNIST labels must cover class IDs 0 through 8."
            )
        sample_hash = _make_sample_hash(splits)
        medmnist_version = version("medmnist")
        download_timestamp = datetime.fromtimestamp(
            splits.archive_path.stat().st_mtime, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z")
    except DatasetValidationError:
        logger.exception("PathMNIST integrity verification failed.")
        raise
    except Exception as error:
        logger.exception("Unexpected PathMNIST verification error.")
        raise DatasetValidationError(
            f"PathMNIST verification failed: {error}"
        ) from error

    report: dict[str, str | int] = {
        "dataset": "PathMNIST",
        "status": "verified",
        "train": EXPECTED_COUNTS["train"],
        "validation": EXPECTED_COUNTS["validation"],
        "test": EXPECTED_COUNTS["test"],
        "image_size": "28x28",
        "channels": "RGB",
        "classes": EXPECTED_CLASSES,
        "package_version": medmnist_version,
        "download_timestamp": download_timestamp,
        "sample_hash": sample_hash,
    }
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as error:
        logger.exception("Could not write dataset report to %s", report_path)
        raise DatasetReportError(
            f"Could not write PathMNIST report to {report_path}: {error}"
        ) from error

    logger.info("Verification successful.")
    logger.info("Dataset report saved to %s", report_path)
    logger.info("PathMNIST ingestion completed.")
    return report
