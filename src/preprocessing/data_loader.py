"""Load PathMNIST through the official MedMNIST package."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from medmnist import PathMNIST

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "PathMNIST"
DEFAULT_LOG_PATH = PROJECT_ROOT / "results" / "logs" / "dataset_download.log"
DATASET_FILENAME = "pathmnist.npz"


class DatasetDownloadError(RuntimeError):
    """Raised when PathMNIST cannot be downloaded or loaded."""


@dataclass(frozen=True)
class PathMNISTSplits:
    """Container for the official PathMNIST splits and raw archive path."""

    train: PathMNIST
    validation: PathMNIST
    test: PathMNIST
    data_dir: Path
    archive_path: Path


def get_dataset_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Configure the dedicated dataset log and return its logger.

    Args:
        log_path: Destination for dataset download and verification messages.

    Returns:
        A logger writing to the dedicated dataset log.

    Raises:
        OSError: If the log directory or file cannot be created.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("privacy_project.dataset_download")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_pathmnist(
    data_dir: Path = DEFAULT_DATA_DIR,
    log_path: Path = DEFAULT_LOG_PATH,
) -> PathMNISTSplits:
    """Load PathMNIST, downloading into ``data_dir`` only when absent.

    MedMNIST stores all three splits in one NPZ archive. The train object
    performs the one-time download when the archive is absent; the other split
    objects read from that archive. No raw data is copied elsewhere.

    Args:
        data_dir: Directory in which MedMNIST stores ``pathmnist.npz``.
        log_path: Destination for the dedicated dataset log.

    Returns:
        Dataset objects for train, validation, and test.

    Raises:
        DatasetDownloadError: If download or dataset loading fails.
        OSError: If the data or log directories cannot be created.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / DATASET_FILENAME
    logger = get_dataset_logger(log_path)
    logger.info("Checking PathMNIST dataset at %s", archive_path)
    should_download = not archive_path.is_file()
    if should_download:
        logger.info("Downloading PathMNIST with the official MedMNIST package.")
    else:
        logger.info("Existing PathMNIST archive detected; reusing it.")

    try:
        train = PathMNIST(split="train", download=should_download, root=data_dir)
        logger.info("Loading validation and test splits from the same archive.")
        validation = PathMNIST(split="val", download=False, root=data_dir)
        test = PathMNIST(split="test", download=False, root=data_dir)
    except Exception as error:
        logger.exception("PathMNIST download or loading failed.")
        raise DatasetDownloadError(
            f"Could not download or load PathMNIST at {archive_path}: {error}"
        ) from error

    logger.info("PathMNIST archive is available at %s", archive_path)
    return PathMNISTSplits(
        train=train,
        validation=validation,
        test=test,
        data_dir=data_dir,
        archive_path=archive_path,
    )
