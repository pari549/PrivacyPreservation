"""Deterministic, priority-aware permutation of a 7x7 block image."""
from __future__ import annotations
import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Sequence
import numpy as np
from src.recoverable.permutation_map import seal_mapping
from src.recoverable.key_loader import load_key
from src.preprocessing.data_loader import DEFAULT_LOG_PATH, PROJECT_ROOT, get_dataset_logger

IMAGE_SIZE = 28
BLOCK_SIZE = 4
GRID_SIZE = 7
BLOCK_COUNT = GRID_SIZE * GRID_SIZE

def _keyed_uniform(key: bytes, block_id: int) -> float:
    message = b"privacy-preservation:block-permutation:v1:" + block_id.to_bytes(4, "big")
    value = int.from_bytes(hmac.new(key, message, hashlib.sha256).digest(), "big")
    return (value + 1) / ((1 << 256) + 1)

def generate_permutation(key: bytes, priorities: Sequence[float]) -> list[dict]:
    """Generate source-to-destination mappings using HMAC-derived ordering.

    Exponential-race ranking makes higher-priority blocks more likely to
    appear earlier while HMAC-derived values make the result key deterministic.
    """
    key = bytes(key)
    if len(key) != 16: raise ValueError("Permutation key must be exactly 128 bits (16 bytes)")
    scores = np.asarray(priorities, dtype=np.float64)
    if scores.shape != (BLOCK_COUNT,) or not np.all(np.isfinite(scores)):
        raise ValueError("Exactly 49 finite priority scores are required")
    low, high = float(scores.min()), float(scores.max())
    normalized = (scores - low) / (high - low) if high > low else np.zeros_like(scores)
    race_times = [-math.log(_keyed_uniform(key, block_id)) / (0.05 + float(normalized[block_id]))
                  for block_id in range(BLOCK_COUNT)]
    source_order = sorted(range(BLOCK_COUNT), key=lambda block_id: (race_times[block_id], block_id))
    mappings = []
    for destination_id, block_id in enumerate(source_order):
        src = [block_id // GRID_SIZE, block_id % GRID_SIZE]
        dst = [destination_id // GRID_SIZE, destination_id % GRID_SIZE]
        mappings.append(seal_mapping({"block_id": block_id, "from": src, "to": dst, "priority": float(scores[block_id])}))
    return sorted(mappings, key=lambda item: item["block_id"])

def permute_blocks(image: np.ndarray, mappings: Sequence[dict]) -> np.ndarray:
    """Place every 4x4 source tile at its mapped destination coordinate."""
    image = np.asarray(image)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3): raise ValueError("Expected a 28x28 RGB image")
    output = np.empty_like(image)
    for item in mappings:
        sr, sc = item["from"]; dr, dc = item["to"]
        output[dr*BLOCK_SIZE:(dr+1)*BLOCK_SIZE, dc*BLOCK_SIZE:(dc+1)*BLOCK_SIZE] = image[sr*BLOCK_SIZE:(sr+1)*BLOCK_SIZE, sc*BLOCK_SIZE:(sc+1)*BLOCK_SIZE]
    return output

def invert_permutation(mappings: Sequence[dict]) -> list[dict]:
    """Return inverse source/destination coordinates for reversible tiling."""
    inverse = []
    for item in mappings:
        src, dst = item["to"], item["from"]
        block_id = src[0] * GRID_SIZE + src[1]
        inverse.append(seal_mapping({"block_id": block_id, "from": src, "to": dst, "priority": float(item["priority"])}))
    return sorted(inverse, key=lambda item: item["block_id"])

def save_permutation_visualizations(image: np.ndarray, mappings: Sequence[dict], output_dir) -> list:
    """Save labeled before/after, destination grid and block-ID overlay figures."""
    import os
    from src.preprocessing.data_loader import PROJECT_ROOT
    cache = PROJECT_ROOT / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    permuted = permute_blocks(image, mappings)
    id_at_destination = {tuple(item["to"]): item["block_id"] for item in mappings}
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, view, title, destination in (
        (axes[0], image, "Original image · source block IDs", False),
        (axes[1], permuted, "Permuted image · carried source IDs", True),
    ):
        axis.imshow(view)
        axis.set_title(title); axis.set_xlabel("Pixel column"); axis.set_ylabel("Pixel row")
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                block_id = id_at_destination[(row, col)] if destination else row * GRID_SIZE + col
                axis.text(col * BLOCK_SIZE + 2, row * BLOCK_SIZE + 2, str(block_id), color="white", fontsize=6,
                          ha="center", va="center", bbox={"facecolor": "black", "alpha": .55, "pad": 1})
    fig.suptitle("Deterministic 4×4 block permutation")
    fig.tight_layout(); path = output_dir / "permutation_before_after.png"
    fig.savefig(path, dpi=200); plt.close(fig); paths.append(path)

    grid = np.asarray([[id_at_destination[(r, c)] for c in range(GRID_SIZE)] for r in range(GRID_SIZE)])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(np.zeros((GRID_SIZE, GRID_SIZE, 3)), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-.5, GRID_SIZE, 1), minor=True); ax.set_yticks(np.arange(-.5, GRID_SIZE, 1), minor=True)
    ax.grid(which="minor", color="steelblue", linewidth=1.5); ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(range(GRID_SIZE)); ax.set_yticks(range(GRID_SIZE))
    ax.set_xlabel("Destination block column"); ax.set_ylabel("Destination block row")
    ax.set_title("Source block ID at each destination")
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE): ax.text(col, row, f"{grid[row,col]:02d}", color="white", ha="center", va="center", fontsize=11)
    fig.tight_layout(); path = output_dir / "permutation_grid.png"
    fig.savefig(path, dpi=200); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 7)); ax.imshow(image)
    ax.set_title("Original image with row-major block IDs"); ax.set_xlabel("Pixel column"); ax.set_ylabel("Pixel row")
    for point in range(0, IMAGE_SIZE + 1, BLOCK_SIZE):
        ax.axhline(point - .5, color="cyan", linewidth=.5); ax.axvline(point - .5, color="cyan", linewidth=.5)
    for block_id in range(BLOCK_COUNT):
        row, col = divmod(block_id, GRID_SIZE)
        ax.text(col*BLOCK_SIZE+2, row*BLOCK_SIZE+2, str(block_id), color="white", fontsize=6,
                ha="center", va="center", bbox={"facecolor": "black", "alpha": .55, "pad": 1})
    fig.tight_layout(); path = output_dir / "block_id_overlay.png"
    fig.savefig(path, dpi=200); plt.close(fig); paths.append(path)
    return paths

def run_block_permutation(
    processed_dir=PROJECT_ROOT / "data" / "processed",
    key_path=PROJECT_ROOT / "keys" / "secret_key.bin",
    key_metadata_path=PROJECT_ROOT / "keys" / "key_metadata.json",
    feature_matrix_path=PROJECT_ROOT / "results" / "metrics" / "block_feature_matrix.json",
    jaya_results_path=PROJECT_ROOT / "results" / "metrics" / "jaya_results.json",
    map_path=PROJECT_ROOT / "metadata" / "permutation_map.json",
    visualization_dir=PROJECT_ROOT / "results" / "images" / "permutation",
    log_path=DEFAULT_LOG_PATH,
) -> dict:
    """Generate, validate, and visualize a map from existing project inputs."""
    from src.recoverable.permutation_validator import validate_permutation
    logger = get_dataset_logger(log_path)
    logger.info("Starting deterministic block-wise permutation.")
    with open(key_metadata_path, "r", encoding="utf-8") as metadata_file:
        key_metadata = json.load(metadata_file)
    if key_metadata.get("algorithm") != "secrets" or key_metadata.get("key_length") != 128:
        raise ValueError("Secret key metadata must identify a 128-bit secrets key")
    key = load_key(key_path)
    with open(feature_matrix_path, "r", encoding="utf-8") as matrix_file:
        matrix = json.load(matrix_file)
    with open(jaya_results_path, "r", encoding="utf-8") as result_file:
        jaya = json.load(result_file)
    samples = matrix.get("samples", [])
    if not samples or int(samples[0].get("source_index", -1)) != 0:
        raise ValueError("Feature matrix has no representative feature set for training image 0")
    features = samples[0].get("features", [])
    if len(features) != BLOCK_COUNT:
        raise ValueError("The representative image must provide exactly 49 block feature records")
    weights = jaya.get("optimized_weights")
    if not isinstance(weights, dict) or not all(name in weights for name in ("variance", "gradient", "entropy", "roi")):
        raise ValueError("Jaya results do not contain the four optimized priority weights")
    # Preserve the saved Jaya weights exactly; compute the requested priority score.
    priorities = [float(weights["variance"]) * float(item["variance"])
                  + float(weights["gradient"]) * float(item["gradient"])
                  + float(weights["entropy"]) * float(item["entropy"])
                  + float(weights["roi"]) * float(item["roi_score"])
                  for item in features]
    mappings = generate_permutation(key, priorities)
    if not validate_permutation(mappings, key, priorities, Path(log_path)):
        raise RuntimeError("Permutation validation failed")
    # Verify each stored score and all mapping checksums before writing outputs.
    map_document = {"algorithm": "HMAC-SHA256 priority-weighted deterministic permutation",
                    "block_size": BLOCK_SIZE, "grid_size": GRID_SIZE, "total_blocks": BLOCK_COUNT,
                    "key_bits": 128, "mappings": mappings}
    map_path = Path(map_path)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(map_document, indent=2) + "\n", encoding="utf-8")
    images = np.load(Path(processed_dir) / "train" / "images.npy", mmap_mode="r", allow_pickle=False)
    image = np.asarray(images[0])
    paths = save_permutation_visualizations(image, mappings, visualization_dir)
    logger.info("Permutation map saved to %s", map_path)
    logger.info("Permutation visualizations saved under %s", visualization_dir)
    return {"total_blocks": BLOCK_COUNT, "unique_mappings": len({tuple(item["to"]) for item in mappings}),
            "deterministic_replay": True, "checksum_verified": True,
            "map_path": str(map_path), "visualization_paths": [str(path) for path in paths]}

if __name__ == "__main__":
    run_block_permutation()
