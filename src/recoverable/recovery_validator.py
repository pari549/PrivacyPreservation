"""Validate deterministic, dimension-preserving, pixel-perfect recovery."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from src.recoverable.metadata_verifier import verify_recovery_metadata
from src.recoverable.recovery_engine import recover_image
from src.preprocessing.data_loader import DEFAULT_LOG_PATH, get_dataset_logger

class RecoveryValidationError(ValueError):
    """Raised when recovery inputs, replay, dimensions or pixels fail checks."""

def validate_recovery(perturbed: np.ndarray, recovered: np.ndarray, reference: np.ndarray,
                      key: bytes, metadata: dict, mappings: list[dict], jaya: dict,
                      log_path: Path = DEFAULT_LOG_PATH) -> dict[str, object]:
    """Verify metadata, replay twice, shape preservation and exact source match."""
    logger = get_dataset_logger(Path(log_path))
    try:
        if not verify_recovery_metadata(metadata, key, mappings, jaya, Path(log_path)):
            raise RecoveryValidationError("Metadata integrity validation failed")
        perturbed, recovered, reference = np.asarray(perturbed), np.asarray(recovered), np.asarray(reference)
        if perturbed.shape != (28, 28, 3) or recovered.shape != perturbed.shape or reference.shape != perturbed.shape:
            raise RecoveryValidationError("Original, perturbed and recovered dimensions must all be 28x28x3")
        if perturbed.dtype != np.uint8 or recovered.dtype != np.uint8:
            raise RecoveryValidationError("Perturbed and recovered images must use uint8 pixels")
        replay = recover_image(perturbed, key, metadata["alpha_snapshot"], mappings)
        repeated = recover_image(perturbed, key, metadata["alpha_snapshot"], mappings)
        if not np.array_equal(replay, recovered) or not np.array_equal(replay, repeated):
            raise RecoveryValidationError("Deterministic recovery replay did not reproduce the recovered pixels")
        if reference.dtype == np.float32:
            reference_bytes = np.rint(reference * np.float32(255)).astype(np.uint8)
        elif reference.dtype == np.uint8:
            reference_bytes = reference
        else:
            raise RecoveryValidationError(f"Unsupported reference dtype: {reference.dtype}")
        if not np.array_equal(recovered, reference_bytes):
            raise RecoveryValidationError("Recovered image differs from the reference pixels")
    except (KeyError, TypeError, ValueError) as error:
        logger.exception("Recovery validation failed")
        if isinstance(error, RecoveryValidationError):
            raise
        raise RecoveryValidationError(f"Recovery validation failed: {error}") from error
    logger.info("Recovery passed metadata, deterministic replay, dimensions, and pixel checks")
    return {"metadata_verified": True, "deterministic_replay": True,
            "dimensions_unchanged": True, "pixel_perfect": True}
