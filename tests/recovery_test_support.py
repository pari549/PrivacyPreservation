"""Temporary, production-key-free recovery bundle fixture for tests."""
from __future__ import annotations

import json
import logging
import secrets
import tempfile
from pathlib import Path

import numpy as np

from src.recoverable.adaptive_perturbation import run_adaptive_perturbation
from src.recoverable.block_permutation import generate_permutation


def close_test_logger() -> None:
    """Close the shared project logger's test-path file handle on Windows."""
    logger = logging.getLogger("privacy_project.dataset_download")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def temporary_test_directory(testcase) -> Path:
    """Register cleanup so the shared logger closes before Windows removes files."""
    temporary = tempfile.TemporaryDirectory()
    testcase.addCleanup(temporary.cleanup)
    testcase.addCleanup(close_test_logger)
    return Path(temporary.name)


def create_recovery_fixture(root: Path, run_id: str = "test-run-v2") -> dict[str, Path]:
    """Create a complete v2 test bundle under ``root`` and return its paths."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "key": root / "test_key_v2.bin",
        "processed": root / "processed_v2",
        "permutation": root / "permutation_map_v2.json",
        "features": root / "feature_matrix_v2.json",
        "jaya": root / "jaya_results_v2.json",
        "metadata": root / "recovery_metadata_v2.json",
        "perturbed": root / "perturbed_image_v2.npy",
        "visualizations": root / "visualizations_v2",
        "log": root / "test_v2.log",
        "report": root / "recovery_report_v2.json",
        "recovery_images": root / "recovery_images_v2",
    }
    paths["key"].write_bytes(secrets.token_bytes(16))
    train_dir = paths["processed"] / "train"
    train_dir.mkdir(parents=True)

    image = (
        np.random.default_rng(2718).integers(0, 256, (28, 28, 3), dtype=np.uint8)
        .astype(np.float32)
        / np.float32(255)
    )
    np.save(train_dir / "images.npy", image[np.newaxis, ...], allow_pickle=False)
    feature_records = []
    for block_id in range(49):
        row, col = divmod(block_id, 7)
        feature_records.append({
            "block_id": block_id,
            "row": row * 4,
            "column": col * 4,
            "mean": 0.5,
            "variance": 0.1 + (block_id / 1000),
            "std": 0.2,
            "gradient": 0.15 + (block_id / 2000),
            "edge_density": 0.25,
            "entropy": 2.0 + (block_id / 100),
            "roi_score": 0.3 + (block_id / 500),
        })
    feature_matrix = {"samples": [{"source_index": 0, "features": feature_records}]}
    paths["features"].write_text(json.dumps(feature_matrix, indent=2) + "\n", encoding="utf-8")

    weights = {"variance": 0.25, "gradient": 0.25, "entropy": 0.25, "roi": 0.25}
    jaya = {
        "optimized_alpha": np.linspace(0.1, 0.9, 49).tolist(),
        "optimized_beta": np.linspace(0.2, 0.8, 49).tolist(),
        "optimized_weights": weights,
    }
    paths["jaya"].write_text(json.dumps(jaya, indent=2) + "\n", encoding="utf-8")
    priorities = [
        weights["variance"] * row["variance"]
        + weights["gradient"] * row["gradient"]
        + weights["entropy"] * row["entropy"]
        + weights["roi"] * row["roi_score"]
        for row in feature_records
    ]
    permutation = generate_permutation(paths["key"].read_bytes(), priorities)
    permutation_document = {
        "algorithm": "HMAC-SHA256 priority-weighted deterministic permutation",
        "block_size": 4,
        "grid_size": 7,
        "total_blocks": 49,
        "key_bits": 128,
        "mappings": permutation,
    }
    paths["permutation"].write_text(
        json.dumps(permutation_document, indent=2) + "\n", encoding="utf-8"
    )

    run_adaptive_perturbation(
        processed_dir=paths["processed"],
        key_path=paths["key"],
        permutation_path=paths["permutation"],
        feature_matrix_path=paths["features"],
        jaya_results_path=paths["jaya"],
        metadata_path=paths["metadata"],
        output_path=paths["perturbed"],
        visualization_dir=paths["visualizations"],
        log_path=paths["log"],
        run_id=run_id,
    )
    paths["reference"] = train_dir / "images.npy"
    return paths
