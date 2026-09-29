"""Read processed PathMNIST splits from the canonical output folders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.preprocessing.normalize import DEFAULT_PROCESSED_DIR


def load_processed_split(split: str, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load images, labels and metadata for one processed split."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown PathMNIST split: {split}")
    directory = Path(processed_dir) / split
    images = np.load(directory / "images.npy", allow_pickle=False)
    labels = np.load(directory / "labels.npy", allow_pickle=False)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return images, labels, metadata
