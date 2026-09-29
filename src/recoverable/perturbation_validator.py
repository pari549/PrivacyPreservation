"""Validation for adaptive perturbation dimensions, metadata and replay."""
from __future__ import annotations
import hashlib
import hmac
import json
from pathlib import Path
import numpy as np

from src.recoverable.noise_generator import derive_noise_seed
from src.preprocessing.data_loader import DEFAULT_LOG_PATH, get_dataset_logger

class PerturbationValidationError(ValueError):
    """Raised when a perturbed image or recovery metadata is invalid."""

def compute_metadata_signature(metadata: dict) -> str:
    """SHA-256 the canonical complete metadata, excluding its signature field."""
    unsigned = {key: value for key, value in metadata.items() if key != "metadata_signature"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

def add_metadata_signature(metadata: dict) -> dict:
    """Return a signed copy of metadata using its canonical SHA-256 digest."""
    signed = {key: value for key, value in metadata.items() if key != "metadata_signature"}
    signed["metadata_signature"] = compute_metadata_signature(signed)
    return signed

def verify_metadata_signature(metadata: dict) -> bool:
    """Raise when metadata has no signature or its content has changed."""
    stored = metadata.get("metadata_signature")
    if not isinstance(stored, str) or len(stored) != 64:
        raise PerturbationValidationError("Metadata signature is missing or malformed; stop before recovery")
    expected = compute_metadata_signature(metadata)
    if not hmac.compare_digest(stored, expected):
        raise PerturbationValidationError("Metadata signature mismatch; stop before recovery")
    return True

def validate_perturbation(original: np.ndarray, perturbed: np.ndarray, key: bytes,
                           alpha_snapshot: list[float], mappings: list[dict], metadata: dict,
                           log_path: Path = DEFAULT_LOG_PATH) -> bool:
    """Check shape, snapshots, permutation binding and deterministic regeneration."""
    from src.recoverable.adaptive_perturbation import (
        ALPHA_BOUNDS, BLOCK_COUNT, apply_adaptive_perturbation, permutation_checksum,
    )
    logger = get_dataset_logger(Path(log_path))
    try:
        verify_metadata_signature(metadata)
        original, perturbed = np.asarray(original), np.asarray(perturbed)
        if original.shape != perturbed.shape or original.shape != (28, 28, 3):
            raise PerturbationValidationError("Original and perturbed image dimensions must both be 28x28x3")
        if len(alpha_snapshot) != BLOCK_COUNT or len(metadata.get("alpha_snapshot", [])) != BLOCK_COUNT:
            raise PerturbationValidationError("Metadata must contain exactly 49 alpha values")
        alphas = np.asarray(alpha_snapshot, dtype=np.float64)
        if not np.all(np.isfinite(alphas)) or np.any(alphas < ALPHA_BOUNDS[0]) or np.any(alphas > ALPHA_BOUNDS[1]):
            raise PerturbationValidationError("Alpha values are outside [0.10, 0.70]")
        if not np.array_equal(alphas, np.asarray(metadata["alpha_snapshot"], dtype=np.float64)):
            raise PerturbationValidationError("Metadata alpha snapshot differs from applied alpha values")
        betas = np.asarray(metadata.get("beta_snapshot", []), dtype=np.float64)
        if betas.shape != (BLOCK_COUNT,) or not np.all(np.isfinite(betas)) or np.any(betas < 0) or np.any(betas > 1):
            raise PerturbationValidationError("Metadata must contain 49 beta values in [0, 1]")
        weights = metadata.get("jaya_weights")
        if not isinstance(weights, dict) or not all(name in weights for name in ("variance", "gradient", "entropy", "roi")):
            raise PerturbationValidationError("Metadata is missing optimized Jaya weights")
        if metadata.get("noise_seed") != derive_noise_seed(key):
            raise PerturbationValidationError("Metadata noise seed does not match the supplied key")
        if metadata.get("permutation_checksum") != permutation_checksum(mappings):
            raise PerturbationValidationError("Metadata does not bind to the supplied permutation map")
        if metadata.get("block_size") != 4 or metadata.get("block_count") != BLOCK_COUNT:
            raise PerturbationValidationError("Metadata lacks required block geometry")
        if not metadata.get("algorithm"):
            raise PerturbationValidationError("Metadata lacks deterministic noise algorithm details")
        first, _, _ = apply_adaptive_perturbation(original, key, list(alphas), mappings)
        second, _, _ = apply_adaptive_perturbation(original, key, list(alphas), mappings)
        if not np.array_equal(first, perturbed) or not np.array_equal(first, second):
            raise PerturbationValidationError("Deterministic perturbation regeneration failed")
    except (KeyError, TypeError, ValueError) as error:
        logger.exception("Adaptive perturbation validation failed")
        if isinstance(error, PerturbationValidationError):
            raise
        raise PerturbationValidationError(f"Invalid perturbation metadata or image: {error}") from error
    logger.info("Adaptive perturbation dimensions, metadata, and deterministic replay validated")
    return True
