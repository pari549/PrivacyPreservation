"""Validation for authenticated adaptive perturbation outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.recoverable.authentication import verify_metadata_hmac
from src.preprocessing.data_loader import DEFAULT_LOG_PATH, get_dataset_logger


class PerturbationValidationError(ValueError):
    """Raised when a perturbed image or authenticated metadata is invalid."""


def validate_perturbation(
    original: np.ndarray,
    perturbed: np.ndarray,
    key: bytes,
    alpha_snapshot: list[float],
    mappings: list[dict],
    metadata: dict,
    log_path: Path = DEFAULT_LOG_PATH,
) -> bool:
    """Check authenticated metadata and deterministic perturbation replay."""
    from src.recoverable.adaptive_perturbation import (
        ALPHA_BOUNDS,
        BLOCK_COUNT,
        apply_adaptive_perturbation,
        permutation_checksum,
    )

    logger = get_dataset_logger(Path(log_path))
    try:
        if metadata.get("schema_version") != 2:
            raise PerturbationValidationError("Authenticated recovery metadata schema v2 is required")
        verify_metadata_hmac(metadata, key)
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
        if metadata.get("permutation_checksum") != permutation_checksum(mappings):
            raise PerturbationValidationError("Metadata does not bind to the supplied permutation map")
        first, _, _ = apply_adaptive_perturbation(original, key, list(alphas), mappings)
        if not np.array_equal(first, perturbed):
            raise PerturbationValidationError("Deterministic perturbation regeneration failed")
    except (KeyError, TypeError, ValueError) as error:
        logger.exception("Adaptive perturbation validation failed")
        if isinstance(error, PerturbationValidationError):
            raise
        raise PerturbationValidationError(f"Invalid perturbation metadata or image: {error}") from error
    logger.info("Authenticated perturbation and deterministic replay validated")
    return True
