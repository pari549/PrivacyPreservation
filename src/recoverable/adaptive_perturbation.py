"""Adaptive, deterministic modulo-256 block perturbation engine."""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
import uuid
from pathlib import Path
import numpy as np

from src.recoverable.block_permutation import permute_blocks
from src.recoverable.authentication import (
    calculate_bytes_sha256,
    generate_metadata_hmac,
)
from src.recoverable.noise_generator import derive_noise_seed, generate_block_noise
from src.preprocessing.data_loader import DEFAULT_LOG_PATH, PROJECT_ROOT, get_dataset_logger

BLOCK_SIZE = 4
GRID_SIZE = 7
BLOCK_COUNT = 49
ALPHA_BOUNDS = (0.10, 0.70)
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_KEY_PATH = PROJECT_ROOT / "keys" / "secret_key.bin"
DEFAULT_PERMUTATION_PATH = PROJECT_ROOT / "metadata" / "permutation_map.json"
DEFAULT_MATRIX_PATH = PROJECT_ROOT / "results" / "metrics" / "block_feature_matrix.json"
DEFAULT_JAYA_PATH = PROJECT_ROOT / "results" / "metrics" / "jaya_results.json"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "metadata" / "recovery_metadata.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "recoverable" / "perturbed_image.npy"
DEFAULT_VISUALIZATION_DIR = PROJECT_ROOT / "results" / "images" / "perturbation"

def permutation_checksum(mappings: list[dict]) -> str:
    """Hash the ordered permutation records to bind recovery metadata to a map."""
    data = [{key: item[key] for key in ("block_id", "from", "to", "priority", "checksum")} for item in mappings]
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

def calculate_alpha_map(features: list[dict], alpha_base: list[float], jaya_weights: dict[str, float]) -> list[float]:
    """Apply the requested priority-aware alpha formula and min-max bounds."""
    if len(features) != BLOCK_COUNT or len(alpha_base) != BLOCK_COUNT:
        raise ValueError("Alpha calculation requires exactly 49 blocks")
    raw = []
    for feature, base in zip(features, alpha_base):
        priority = (float(jaya_weights["variance"]) * float(feature["variance"])
                    + float(jaya_weights["gradient"]) * float(feature["gradient"])
                    + float(jaya_weights["entropy"]) * float(feature["entropy"])
                    + float(jaya_weights["roi"]) * float(feature["roi_score"]))
        adaptive_factor = (.4 * priority + .3 * float(feature["roi_score"])
                           + .2 * float(feature["gradient"]) + .1 * float(feature["entropy"]))
        raw.append(float(base) * adaptive_factor)
    values = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)): raise ValueError("Alpha source values must be finite")
    minimum, maximum = float(values.min()), float(values.max())
    if maximum == minimum:
        normalized = np.full(BLOCK_COUNT, (ALPHA_BOUNDS[0] + ALPHA_BOUNDS[1]) / 2)
    else:
        normalized = ALPHA_BOUNDS[0] + (values - minimum) * (ALPHA_BOUNDS[1] - ALPHA_BOUNDS[0]) / (maximum - minimum)
    return normalized.tolist()

def apply_adaptive_perturbation(image: np.ndarray, key: bytes, alpha_snapshot: list[float],
                                mappings: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perturb each source block reversibly, then apply the saved bijection.

    Pixel bytes are converted to uint8 and updated modulo 256. The signed noise
    field is deterministically regenerated from key and source block index.
    """
    image = np.asarray(image)
    if image.shape != (28, 28, 3): raise ValueError("Expected a 28x28 RGB image")
    if image.dtype == np.float32:
        if image.min() < 0 or image.max() > 1: raise ValueError("Float image must be normalized to [0, 1]")
        source = np.rint(image * np.float32(255)).astype(np.uint8)
    elif image.dtype == np.uint8:
        source = image.copy()
    else:
        raise ValueError("Image must be normalized float32 or uint8")
    if len(alpha_snapshot) != BLOCK_COUNT: raise ValueError("Alpha snapshot must contain 49 values")
    perturbed_source = np.empty_like(source)
    noise_field = np.empty(source.shape, dtype=np.int16)
    for block_id, alpha in enumerate(alpha_snapshot):
        if not ALPHA_BOUNDS[0] <= float(alpha) <= ALPHA_BOUNDS[1]:
            raise ValueError(f"Alpha at block {block_id} is outside [0.10, 0.70]")
        row, col = divmod(block_id, GRID_SIZE)
        r0, c0 = row * BLOCK_SIZE, col * BLOCK_SIZE
        noise = generate_block_noise(key, block_id, float(alpha))
        source_block = source[r0:r0+BLOCK_SIZE, c0:c0+BLOCK_SIZE].astype(np.int16)
        perturbed_source[r0:r0+BLOCK_SIZE, c0:c0+BLOCK_SIZE] = ((source_block + noise) % 256).astype(np.uint8)
        noise_field[r0:r0+BLOCK_SIZE, c0:c0+BLOCK_SIZE] = noise
    return permute_blocks(perturbed_source, mappings), permute_blocks(noise_field, mappings), source

def save_perturbation_visualizations(original: np.ndarray, perturbed: np.ndarray, noise_field: np.ndarray,
                                     alpha_snapshot: list[float], output_dir: Path) -> list[Path]:
    """Write labeled comparison and consistent-scale perturbation figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    originals = original.astype(np.float32) / 255.0
    changed = perturbed.astype(np.float32) / 255.0
    magnitude = np.abs(noise_field.astype(np.float32)).mean(axis=2)
    signed_noise = noise_field.astype(np.float32).mean(axis=2)
    alpha_grid = np.asarray(alpha_snapshot, dtype=np.float32).reshape(GRID_SIZE, GRID_SIZE)
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(originals); axes[0].set_title("Original image"); axes[0].set_xlabel("Pixel column"); axes[0].set_ylabel("Pixel row")
    axes[1].imshow(changed); axes[1].set_title("Permuted and reversibly perturbed"); axes[1].set_xlabel("Pixel column"); axes[1].set_ylabel("Pixel row")
    fig.suptitle("Adaptive reversible perturbation · 4×4 blocks"); fig.tight_layout()
    path = output_dir / "perturbed_image.png"; fig.savefig(path, dpi=200); plt.close(fig); paths.append(path)
    fig, ax = plt.subplots(figsize=(6, 5)); plot = ax.imshow(alpha_grid, cmap="viridis", vmin=.10, vmax=.70)
    ax.set(title="Block-specific alpha", xlabel="Block column", ylabel="Block row"); fig.colorbar(plot, ax=ax, label="Alpha")
    fig.tight_layout(); path = output_dir / "alpha_heatmap.png"; fig.savefig(path, dpi=200); plt.close(fig); paths.append(path)
    fig, ax = plt.subplots(figsize=(7, 5)); plot = ax.imshow(magnitude, cmap="magma", vmin=0, vmax=math.ceil(.70*255))
    ax.set(title="Absolute perturbation magnitude", xlabel="Permuted pixel column", ylabel="Permuted pixel row")
    fig.colorbar(plot, ax=ax, label="Mean absolute signed noise (uint8 levels)")
    fig.tight_layout(); path = output_dir / "perturbation_heatmap.png"; fig.savefig(path, dpi=200); plt.close(fig); paths.append(path)
    fig, ax = plt.subplots(figsize=(7, 5)); plot = ax.imshow(signed_noise, cmap="coolwarm", vmin=-math.ceil(.70*255), vmax=math.ceil(.70*255))
    ax.set(title="Deterministic signed noise pattern", xlabel="Permuted pixel column", ylabel="Permuted pixel row")
    fig.colorbar(plot, ax=ax, label="Signed noise (uint8 levels)")
    fig.tight_layout(); path = output_dir / "noise_pattern.png"; fig.savefig(path, dpi=200); plt.close(fig); paths.append(path)
    return paths

def run_adaptive_perturbation(
    processed_dir: Path = DEFAULT_PROCESSED_DIR, key_path: Path = DEFAULT_KEY_PATH,
    permutation_path: Path = DEFAULT_PERMUTATION_PATH, feature_matrix_path: Path = DEFAULT_MATRIX_PATH,
    jaya_results_path: Path = DEFAULT_JAYA_PATH, metadata_path: Path = DEFAULT_METADATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH, visualization_dir: Path = DEFAULT_VISUALIZATION_DIR,
    log_path: Path = DEFAULT_LOG_PATH,
    run_id: str | None = None,
) -> dict:
    """Create a v2 authenticated bundle for training image zero."""
    from src.recoverable.key_loader import load_key
    from src.recoverable.permutation_validator import validate_permutation
    from src.recoverable.metadata_verifier import source_image_sha256, verify_recovery_metadata
    from src.recoverable.perturbation_validator import validate_perturbation
    logger = get_dataset_logger(Path(log_path)); logger.info("Starting adaptive reversible perturbation.")
    key = load_key(Path(key_path))
    permutation_bytes = Path(permutation_path).read_bytes()
    feature_bytes = Path(feature_matrix_path).read_bytes()
    jaya_bytes = Path(jaya_results_path).read_bytes()
    source_dataset_path = Path(processed_dir) / "train" / "images.npy"
    permutation_document = json.loads(permutation_bytes)
    mappings = permutation_document["mappings"]
    feature_data = json.loads(feature_bytes)
    jaya = json.loads(jaya_bytes)
    features = feature_data["samples"][0]["features"]
    weights = jaya["optimized_weights"]
    alpha_base = jaya["optimized_alpha"][:BLOCK_COUNT]
    beta_snapshot = [float(value) for value in jaya["optimized_beta"][:BLOCK_COUNT]]
    if len(beta_snapshot) != BLOCK_COUNT: raise ValueError("Jaya output must contain 49 beta values")
    alpha_snapshot = calculate_alpha_map(features, alpha_base, weights)
    source_dataset = np.load(source_dataset_path, mmap_mode="r", allow_pickle=False)
    original = np.array(source_dataset[0], copy=True)
    priorities = [
        float(weights["variance"]) * float(item["variance"])
        + float(weights["gradient"]) * float(item["gradient"])
        + float(weights["entropy"]) * float(item["entropy"])
        + float(weights["roi"]) * float(item["roi_score"])
        for item in features
    ]
    if not validate_permutation(mappings, key, priorities, Path(log_path)):
        raise ValueError("Existing permutation map failed validation")
    perturbed, noise_field, original_bytes = apply_adaptive_perturbation(original, key, alpha_snapshot, mappings)
    perturbed_buffer = io.BytesIO()
    np.save(perturbed_buffer, perturbed, allow_pickle=False)
    perturbed_bytes = perturbed_buffer.getvalue()
    current_run_id = run_id or uuid.uuid4().hex
    artifact_hashes = {
        "perturbed_image": calculate_bytes_sha256(perturbed_bytes),
        "permutation_map": calculate_bytes_sha256(permutation_bytes),
        "jaya_output": calculate_bytes_sha256(jaya_bytes),
        "feature_matrix": calculate_bytes_sha256(feature_bytes),
        "source_image": source_image_sha256(original),
    }
    metadata = {
        "schema_version": 2,
        "run_id": current_run_id,
        "noise_seed": derive_noise_seed(key),
        "alpha_snapshot": alpha_snapshot,
        "beta_snapshot": beta_snapshot,
        "jaya_weights": {name: float(value) for name, value in weights.items()},
        "permutation_checksum": permutation_checksum(mappings),
        "algorithm": "HMAC-SHA256-derived PCG64 signed integer noise; uint8 addition modulo 256",
        "block_size": BLOCK_SIZE,
        "block_count": BLOCK_COUNT,
        "source_split": "train",
        "source_index": 0,
        "artifacts": artifact_hashes,
    }
    metadata["metadata_hmac"] = generate_metadata_hmac(metadata, key)
    if not validate_perturbation(original, perturbed, key, alpha_snapshot, mappings, metadata, log_path=Path(log_path)):
        raise RuntimeError("Adaptive perturbation validation failed")
    if not verify_recovery_metadata(
        metadata, key, mappings, jaya, feature_data, artifact_hashes, original,
        expected_run_id=current_run_id, log_path=Path(log_path),
    ):
        raise RuntimeError("Authenticated recovery bundle validation failed")
    metadata_path, output_path = Path(metadata_path), Path(output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True); output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_metadata = json.dumps(metadata, indent=2) + "\n"
    output_path.write_bytes(perturbed_bytes)
    metadata_path.write_text(serialized_metadata, encoding="utf-8")
    visualization_paths = save_perturbation_visualizations(original_bytes, perturbed, noise_field, alpha_snapshot, visualization_dir)
    logger.info("Adaptive perturbation saved to %s; recovery metadata saved to %s", output_path, metadata_path)
    return {"alpha_minimum": min(alpha_snapshot), "alpha_maximum": max(alpha_snapshot),
            "perturbed_blocks": BLOCK_COUNT, "deterministic_replay": True,
            "metadata_size": metadata_path.stat().st_size, "metadata_path": str(metadata_path),
            "visualization_paths": [str(path) for path in visualization_paths]}

if __name__ == "__main__":
    run_adaptive_perturbation()
