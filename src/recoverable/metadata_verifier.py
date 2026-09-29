"""Verify signed recovery metadata and its bindings to existing inputs."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from src.recoverable.noise_generator import derive_noise_seed
from src.recoverable.permutation_validator import validate_permutation
from src.recoverable.perturbation_validator import (
    PerturbationValidationError,
    verify_metadata_signature,
)
from src.preprocessing.data_loader import DEFAULT_LOG_PATH, get_dataset_logger

class MetadataVerificationError(ValueError):
    """Raised when signed metadata or any bound recovery input is invalid."""

def verify_recovery_metadata(metadata: dict, key: bytes, mappings: list[dict], jaya: dict,
                             log_path: Path = DEFAULT_LOG_PATH) -> bool:
    """Check signature first, then key seed, snapshots, Jaya values and map."""
    logger = get_dataset_logger(Path(log_path))
    try:
        # This check deliberately precedes all other metadata use.
        verify_metadata_signature(metadata)
        if metadata.get("noise_seed") != derive_noise_seed(key):
            raise MetadataVerificationError("Stored noise seed does not match the supplied secret key")
        if metadata.get("block_size") != 4 or metadata.get("block_count") != 49:
            raise MetadataVerificationError("Recovery metadata has invalid block geometry")
        alpha = np.asarray(metadata.get("alpha_snapshot", []), dtype=np.float64)
        beta = np.asarray(metadata.get("beta_snapshot", []), dtype=np.float64)
        if alpha.shape != (49,) or not np.all(np.isfinite(alpha)) or np.any(alpha < .10) or np.any(alpha > .70):
            raise MetadataVerificationError("Alpha snapshot must contain 49 values in [0.10, 0.70]")
        if beta.shape != (49,) or not np.all(np.isfinite(beta)) or np.any(beta < 0) or np.any(beta > 1):
            raise MetadataVerificationError("Beta snapshot must contain 49 values in [0, 1]")
        jaya_beta = np.asarray(jaya.get("optimized_beta", [])[:49], dtype=np.float64)
        if jaya_beta.shape != (49,) or not np.array_equal(beta, jaya_beta):
            raise MetadataVerificationError("Beta snapshot differs from the saved Jaya outputs")
        weights = metadata.get("jaya_weights", {})
        saved_weights = jaya.get("optimized_weights", {})
        weight_names = ("variance", "gradient", "entropy", "roi")
        if set(weights) != set(weight_names) or any(float(weights[name]) != float(saved_weights[name]) for name in weight_names):
            raise MetadataVerificationError("Jaya weights differ from the recovery metadata snapshot")
        priorities = [float(item["priority"]) for item in sorted(mappings, key=lambda item: item["block_id"])]
        if not validate_permutation(mappings, key, priorities, Path(log_path)):
            raise MetadataVerificationError("Permutation map validation failed")
        from src.recoverable.adaptive_perturbation import permutation_checksum
        if metadata.get("permutation_checksum") != permutation_checksum(mappings):
            raise MetadataVerificationError("Permutation checksum differs from signed recovery metadata")
        if not metadata.get("algorithm") or metadata.get("source_split") != "train" or metadata.get("source_index") != 0:
            raise MetadataVerificationError("Recovery metadata is missing its transform or source description")
    except (KeyError, TypeError, ValueError, PerturbationValidationError) as error:
        logger.exception("Recovery metadata verification failed")
        if isinstance(error, MetadataVerificationError):
            raise
        raise MetadataVerificationError(f"Recovery metadata verification failed: {error}") from error
    logger.info("Recovery metadata signature and input bindings verified")
    return True

def load_recovery_inputs(metadata_path: Path, permutation_path: Path, jaya_path: Path) -> tuple[dict, list[dict], dict]:
    """Read existing metadata, permutation map, and Jaya result files."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    mappings = json.loads(Path(permutation_path).read_text(encoding="utf-8"))["mappings"]
    jaya = json.loads(Path(jaya_path).read_text(encoding="utf-8"))
    return metadata, mappings, jaya
