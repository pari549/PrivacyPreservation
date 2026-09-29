"""Common logging configuration for the medical image privacy project."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    log_dir: Path | str = Path("results/logs"),
    level: int = logging.INFO,
    log_filename: str = "privacy_project.log",
) -> logging.Logger:
    """Configure console and file logging and return the project logger.

    Args:
        log_dir: Directory where the persistent log file is stored.
        level: Minimum logging level for both handlers.
        log_filename: Name of the persistent log file.

    Returns:
        The configured project logger.

    Raises:
        OSError: If the log directory or file cannot be created.
        ValueError: If ``log_filename`` is not a plain filename.
    """
    filename = Path(log_filename)
    if filename.name != log_filename or filename.suffix.lower() != ".log":
        raise ValueError("log_filename must be a plain .log filename")

    destination = Path(log_dir)
    destination.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("privacy_project")
    logger.setLevel(level)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(destination / filename, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Logging initialized; log file: %s", destination / filename)
    return logger
