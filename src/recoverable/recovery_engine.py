"""Exact recovery for the saved deterministic Method 1 transform."""
from __future__ import annotations
import json
import io
import os
from pathlib import Path
import numpy as np
from src.recoverable.noise_generator import generate_block_noise
from src.recoverable.authentication import calculate_bytes_sha256, verify_metadata_hmac
from src.preprocessing.data_loader import DEFAULT_LOG_PATH, PROJECT_ROOT, get_dataset_logger

GRID_SIZE = 7
BLOCK_SIZE = 4

def recover_image(perturbed: np.ndarray, key: bytes, alpha_snapshot: list[float],
                  mappings: list[dict]) -> np.ndarray:
    """Undo permutation, then subtract regenerated block noise modulo 256."""
    perturbed = np.asarray(perturbed)
    if perturbed.shape != (28, 28, 3) or perturbed.dtype != np.uint8:
        raise ValueError("Perturbed input must be a uint8 image of shape 28x28x3")
    if len(alpha_snapshot) != 49 or len(mappings) != 49:
        raise ValueError("Recovery requires 49 alpha values and 49 mappings")
    unpermuted = np.empty_like(perturbed)
    for item in mappings:
        source_row, source_col = item["from"]
        destination_row, destination_col = item["to"]
        sr, sc = source_row * BLOCK_SIZE, source_col * BLOCK_SIZE
        dr, dc = destination_row * BLOCK_SIZE, destination_col * BLOCK_SIZE
        unpermuted[sr:sr+BLOCK_SIZE, sc:sc+BLOCK_SIZE] = perturbed[dr:dr+BLOCK_SIZE, dc:dc+BLOCK_SIZE]
    recovered = np.empty_like(unpermuted)
    for block_id, alpha in enumerate(alpha_snapshot):
        row, col = divmod(block_id, GRID_SIZE)
        r0, c0 = row * BLOCK_SIZE, col * BLOCK_SIZE
        noise = generate_block_noise(key, block_id, float(alpha))
        block = unpermuted[r0:r0+BLOCK_SIZE, c0:c0+BLOCK_SIZE].astype(np.int16)
        recovered[r0:r0+BLOCK_SIZE, c0:c0+BLOCK_SIZE] = ((block - noise) % 256).astype(np.uint8)
    return recovered

def run_recovery(
    key_path: Path = PROJECT_ROOT / "keys" / "secret_key.bin",
    permutation_path: Path = PROJECT_ROOT / "metadata" / "permutation_map.json",
    metadata_path: Path = PROJECT_ROOT / "metadata" / "recovery_metadata.json",
    perturbed_path: Path = PROJECT_ROOT / "data" / "recoverable" / "perturbed_image.npy",
    jaya_path: Path = PROJECT_ROOT / "results" / "metrics" / "jaya_results.json",
    feature_matrix_path: Path = PROJECT_ROOT / "results" / "metrics" / "block_feature_matrix.json",
    reference_path: Path = PROJECT_ROOT / "data" / "processed" / "train" / "images.npy",
    report_path: Path = PROJECT_ROOT / "results" / "metrics" / "recovery_report.json",
    output_dir: Path = PROJECT_ROOT / "results" / "images" / "recovery",
    log_path: Path = DEFAULT_LOG_PATH,
    expected_run_id: str | None = None,
) -> dict:
    """Authenticate one v2 artifact bundle before recovery, then evaluate it.

    Whole-bundle rollback cannot be detected unless ``expected_run_id`` comes
    from trusted external state; an older complete authenticated bundle remains
    cryptographically valid on its own.
    """
    from src.recoverable.key_loader import load_key
    from src.recoverable.metadata_verifier import (
        MetadataVerificationError,
        source_image_sha256,
        verify_recovery_metadata,
    )
    from src.recoverable.recovery_validator import validate_recovery
    logger = get_dataset_logger(Path(log_path)); logger.info("Starting Method 1 image recovery.")

    key = load_key(Path(key_path))
    metadata_bytes = Path(metadata_path).read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata.get("schema_version") != 2:
        raise ValueError("Legacy unauthenticated metadata is not accepted by secure recovery")
    verify_metadata_hmac(metadata, key)

    permutation_bytes = Path(permutation_path).read_bytes()
    jaya_bytes = Path(jaya_path).read_bytes()
    feature_bytes = Path(feature_matrix_path).read_bytes()
    perturbed_bytes = Path(perturbed_path).read_bytes()

    # Reject substituted artifacts before parsing or loading their contents.
    expected_hashes = metadata.get("artifacts")
    if not isinstance(expected_hashes, dict):
        raise MetadataVerificationError("Recovery metadata has no artifact hashes")
    input_bytes = {
        "permutation_map": permutation_bytes,
        "jaya_output": jaya_bytes,
        "feature_matrix": feature_bytes,
        "perturbed_image": perturbed_bytes,
    }
    input_hashes = {}
    for name, content in input_bytes.items():
        actual_hash = calculate_bytes_sha256(content)
        if expected_hashes.get(name) != actual_hash:
            raise MetadataVerificationError(f"Recovery {name} artifact hash mismatch")
        input_hashes[name] = actual_hash

    permutation_document = json.loads(permutation_bytes)
    mappings = permutation_document["mappings"]
    jaya = json.loads(jaya_bytes)
    feature_matrix = json.loads(feature_bytes)
    source_dataset = np.load(Path(reference_path), mmap_mode="r", allow_pickle=False)
    source_index = metadata.get("source_index")
    if not isinstance(source_index, int) or isinstance(source_index, bool) or not 0 <= source_index < len(source_dataset):
        raise ValueError("Recovery metadata has an invalid source image index")
    reference = np.array(source_dataset[source_index], copy=True)
    source_mapping = getattr(source_dataset, "_mmap", None)
    if source_mapping is not None:
        source_mapping.close()
    artifact_hashes = {
        **input_hashes,
        "source_image": source_image_sha256(reference),
    }
    if not verify_recovery_metadata(
        metadata, key, mappings, jaya, feature_matrix, artifact_hashes, reference,
        expected_run_id=expected_run_id, log_path=Path(log_path),
    ):
        raise RuntimeError("Recovery metadata verification failed")
    perturbed = np.load(io.BytesIO(perturbed_bytes), allow_pickle=False)
    recovered = recover_image(perturbed, key, metadata["alpha_snapshot"], mappings)

    validation = validate_recovery(
        perturbed, recovered, reference, key, metadata, mappings, jaya,
        feature_matrix, artifact_hashes, expected_run_id, Path(log_path),
    )
    reference_float = np.asarray(reference, dtype=np.float32)
    recovered_float = recovered.astype(np.float32) / np.float32(255.0)
    difference = np.abs(reference_float - recovered_float)
    mse = float(np.mean(np.square(difference, dtype=np.float64)))
    from skimage.metrics import structural_similarity
    ssim = float(structural_similarity(reference_float, recovered_float, channel_axis=2, data_range=1.0))
    psnr: float | str
    psnr = "Infinity" if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))
    recovery_rate = float(np.mean(np.all(reference_float == recovered_float, axis=2)) * 100.0)
    report = {"ssim": ssim, "psnr": psnr, "mse": mse, "recovery_rate": recovery_rate,
              "metadata_verified": validation["metadata_verified"]}
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / ".matplotlib_cache"; cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    original_display = reference_float
    perturbed_display = perturbed.astype(np.float32) / np.float32(255.0)
    fig, axis = plt.subplots(figsize=(5, 5)); axis.imshow(recovered_float)
    axis.set(title=f"Recovered image · SSIM {ssim:.6f}", xlabel="Pixel column", ylabel="Pixel row")
    fig.tight_layout(); fig.savefig(output_dir / "recovered_image.png", dpi=200); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title in zip(axes, (original_display, perturbed_display, recovered_float),
                                  ("Original", "Permuted + perturbed", "Recovered")):
        axis.imshow(image); axis.set_title(title); axis.set_xlabel("Pixel column"); axis.set_ylabel("Pixel row")
    fig.suptitle("Method 1 recovery comparison"); fig.tight_layout()
    fig.savefig(output_dir / "comparison_grid.png", dpi=200); plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 5)); plot = axis.imshow(difference.max(axis=2), cmap="inferno", vmin=0, vmax=1/255)
    axis.set(title=f"Absolute pixel difference · MSE {mse:.3g}", xlabel="Pixel column", ylabel="Pixel row")
    fig.colorbar(plot, ax=axis, label="Absolute normalized difference"); fig.tight_layout()
    fig.savefig(output_dir / "difference_map.png", dpi=200); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, image, title in zip(axes[0], (original_display, perturbed_display, recovered_float),
                                  ("Original", "Perturbed", "Recovered")):
        axis.imshow(image); axis.set_title(title); axis.axis("off")
    axes[1, 0].axis("off"); axes[1, 2].axis("off"); mid = axes[1, 1]; mid.axis("off")
    mid.text(.5, .8, "Recovery pipeline", ha="center", fontsize=14, weight="bold")
    mid.text(.5, .56, "verify metadata HMAC", ha="center", bbox={"boxstyle":"round", "facecolor":"#dceeff"})
    mid.text(.5, .34, "inverse block permutation", ha="center", bbox={"boxstyle":"round", "facecolor":"#e5f5e0"})
    mid.text(.5, .12, "regenerate noise · subtract mod 256", ha="center", bbox={"boxstyle":"round", "facecolor":"#fff2cc"})
    mid.add_patch(FancyArrowPatch((.5,.53),(.5,.44), arrowstyle="->", mutation_scale=12))
    mid.add_patch(FancyArrowPatch((.5,.31),(.5,.22), arrowstyle="->", mutation_scale=12))
    fig.suptitle("Exact deterministic recovery steps"); fig.tight_layout()
    fig.savefig(output_dir / "recovery_pipeline.png", dpi=200); plt.close(fig)
    report_path = Path(report_path); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Recovery completed: SSIM %.8f, MSE %.8g, rate %.3f%%", ssim, mse, recovery_rate)
    return {**report, "recovered_image": recovered,
            "image_paths": [str(output_dir / name) for name in ("recovered_image.png", "comparison_grid.png", "difference_map.png", "recovery_pipeline.png")],
            "report_path": str(report_path)}

if __name__ == "__main__":
    run_recovery()
