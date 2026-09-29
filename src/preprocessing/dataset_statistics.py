"""Compute reproducible PathMNIST pixel and class analytics."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.preprocessing.data_loader import DEFAULT_LOG_PATH, PROJECT_ROOT, get_dataset_logger

MATPLOTLIB_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.normalize import DEFAULT_PROCESSED_DIR, EXPECTED_COUNTS

PREVIEW_DIR = PROJECT_ROOT / "results" / "images" / "dataset_preview"
DEFAULT_STATISTICS_PATH = PROJECT_ROOT / "results" / "metrics" / "dataset_statistics.json"


def generate_dataset_statistics(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    preview_dir: Path = PREVIEW_DIR,
    statistics_path: Path = DEFAULT_STATISTICS_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict[str, object]:
    """Generate sample grid, histograms, class chart, and summary JSON."""
    logger = get_dataset_logger(log_path)
    logger.info("Starting PathMNIST visualization and analytics.")
    processed_dir, preview_dir = Path(processed_dir), Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    train_dir = processed_dir / "train"
    images = np.load(train_dir / "images.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(train_dir / "labels.npy", mmap_mode="r", allow_pickle=False).reshape(-1)
    if images.shape != (EXPECTED_COUNTS["train"], 28, 28, 3) or images.dtype != np.float32:
        raise ValueError(f"Unexpected processed training image array: {images.shape}, {images.dtype}")

    rng = np.random.default_rng(42)
    selected = rng.choice(len(images), size=20, replace=False)
    figure, axes = plt.subplots(4, 5, figsize=(12, 10))
    for axis, index in zip(axes.flat, selected):
        axis.imshow(images[index])
        axis.set_title(f"Class {int(labels[index])}", fontsize=9)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(preview_dir / "sample_grid.png", dpi=160)
    plt.close(figure)

    channel_counts = [np.zeros(256, dtype=np.int64) for _ in range(3)]
    all_counts = np.zeros(256, dtype=np.int64)
    total = 0
    total_sq = 0.0
    pixels = 0
    class_counts = np.bincount(labels.astype(np.int64), minlength=9)
    for start in range(0, len(images), 256):
        chunk = np.asarray(images[start : start + 256])
        quantized = np.rint(chunk * np.float32(255.0)).astype(np.uint8)
        for channel in range(3):
            channel_counts[channel] += np.bincount(quantized[..., channel].ravel(), minlength=256)
        counts = np.bincount(quantized.ravel(), minlength=256)
        all_counts += counts
        total += int(np.dot(np.arange(256, dtype=np.int64), counts))
        total_sq += float(np.dot(np.arange(256, dtype=np.float64) ** 2, counts))
        pixels += quantized.size

    values = np.arange(256, dtype=np.float64) / 255.0
    mean = total / (pixels * 255.0)
    variance = max(0.0, total_sq / (pixels * 255.0**2) - mean**2)
    midpoint = (pixels - 1) / 2
    cumulative = np.cumsum(all_counts)
    lower = int(np.searchsorted(cumulative, np.floor(midpoint) + 1))
    upper = int(np.searchsorted(cumulative, np.ceil(midpoint) + 1))
    median = float((lower + upper) / (2 * 255.0))
    pixel_stats = {
        "min": float(values[np.flatnonzero(all_counts)[0]]),
        "max": float(values[np.flatnonzero(all_counts)[-1]]),
        "mean": float(mean),
        "median": median,
        "std": float(np.sqrt(variance)),
    }

    figure, axis = plt.subplots(figsize=(10, 6))
    for channel, name, color in zip(range(3), ("Red", "Green", "Blue"), ("red", "green", "blue")):
        axis.plot(values, channel_counts[channel] / channel_counts[channel].sum(), color=color, label=name)
    axis.set(xlabel="Normalized pixel value", ylabel="Proportion", title="PathMNIST RGB channel histograms")
    axis.legend()
    figure.tight_layout()
    figure.savefig(preview_dir / "rgb_histogram.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(np.arange(9), class_counts, color="steelblue")
    axis.set(xticks=np.arange(9), xlabel="Class", ylabel="Image count", title="PathMNIST training class distribution")
    figure.tight_layout()
    figure.savefig(preview_dir / "class_distribution.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(values, all_counts / pixels, width=1 / 255, color="slateblue")
    axis.set(xlabel="Normalized pixel value", ylabel="Proportion", title="PathMNIST pixel distribution")
    figure.tight_layout()
    figure.savefig(preview_dir / "pixel_distribution.png", dpi=160)
    plt.close(figure)

    result: dict[str, object] = {
        "dataset": "PathMNIST",
        "train": EXPECTED_COUNTS["train"],
        "validation": EXPECTED_COUNTS["validation"],
        "test": EXPECTED_COUNTS["test"],
        "channels": "RGB",
        "shape": "28x28x3",
        "pixel_statistics": pixel_stats,
        "class_distribution": {str(index): int(count) for index, count in enumerate(class_counts)},
    }
    statistics_path = Path(statistics_path)
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("PathMNIST visualizations and statistics saved under %s", preview_dir)
    logger.info("Dataset statistics saved to %s", statistics_path)
    return result
