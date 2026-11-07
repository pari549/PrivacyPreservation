"""Authenticate and validate all inputs required by the v2 recovery bundle."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.recoverable.authentication import verify_metadata_hmac
from src.recoverable.noise_generator import derive_noise_seed
from src.recoverable.permutation_validator import validate_permutation
from src.preprocessing.data_loader import DEFAULT_LOG_PATH, get_dataset_logger

ARTIFACT_NAMES = (
    "perturbed_image",
    "permutation_map",
    "jaya_output",
    "feature_matrix",
    "source_image",
)


class MetadataVerificationError(ValueError):
    """Raised when authenticated metadata or a bound recovery input is invalid."""


def source_image_sha256(image: np.ndarray) -> str:
    """Hash the selected image's dtype, shape, and contiguous pixel bytes."""
    image = np.asarray(image)
    digest = hashlib.sha256()
    digest.update(image.dtype.str.encode("ascii"))
    digest.update(repr(tuple(image.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def verify_recovery_metadata(
    metadata: dict,
    key: bytes,
    mappings: list[dict],
    jaya: dict,
    feature_matrix: dict,
    artifact_hashes: dict[str, str],
    source_image: np.ndarray,
    *,
    expected_run_id: str | None = None,
    log_path: Path = DEFAULT_LOG_PATH,
) -> bool:
    """Authenticate metadata and bind it to the exact artifacts being consumed.

    Artifact bytes must be read once by the caller, hashed from those bytes,
    and parsed/loaded from the same bytes before this function is called.
    """
    logger = get_dataset_logger(Path(log_path))
    try:
        if metadata.get("schema_version") != 2:
            raise MetadataVerificationError(
                "Unsupported or legacy recovery metadata; authenticated schema v2 is required"
            )
        verify_metadata_hmac(metadata, key)
        run_id = metadata.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise MetadataVerificationError("Recovery metadata has no valid run_id")
        if expected_run_id is not None and run_id != expected_run_id:
            raise MetadataVerificationError("Recovery run_id does not match the trusted expected_run_id")

        recorded_hashes = metadata.get("artifacts")
        if not isinstance(recorded_hashes, dict) or set(recorded_hashes) != set(ARTIFACT_NAMES):
            raise MetadataVerificationError("Recovery metadata has incomplete artifact hashes")
        if set(artifact_hashes) != set(ARTIFACT_NAMES):
            raise MetadataVerificationError("Recovery inputs have incomplete artifact hashes")
        for name in ARTIFACT_NAMES:
            if recorded_hashes[name] != artifact_hashes[name]:
                raise MetadataVerificationError(f"Recovery {name} artifact hash mismatch")
        actual_source_hash = source_image_sha256(source_image)
        if recorded_hashes["source_image"] != actual_source_hash:
            raise MetadataVerificationError("Selected processed source image hash mismatch")

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

        source_index = metadata.get("source_index")
        samples = feature_matrix.get("samples", [])
        sample = next(
            (item for item in samples if item.get("source_index") == source_index), None
        )
        if metadata.get("source_split") != "train" or sample is None:
            raise MetadataVerificationError("Feature matrix does not contain the selected training source")
        features = sample.get("features", [])
        if len(features) != 49:
            raise MetadataVerificationError("Selected feature sample must contain 49 block records")
        from src.recoverable.adaptive_perturbation import calculate_alpha_map

        jaya_alpha = jaya.get("optimized_alpha", [])[:49]
        jaya_beta = np.asarray(jaya.get("optimized_beta", [])[:49], dtype=np.float64)
        if len(jaya_alpha) != 49 or jaya_beta.shape != (49,) or not np.array_equal(
            beta, jaya_beta
        ):
            raise MetadataVerificationError("Beta snapshot differs from the saved Jaya output")
        weights = metadata.get("jaya_weights", {})
        saved_weights = jaya.get("optimized_weights", {})
        weight_names = ("variance", "gradient", "entropy", "roi")
        if set(weights) != set(weight_names) or any(
            float(weights[name]) != float(saved_weights[name]) for name in weight_names
        ):
            raise MetadataVerificationError("Jaya weights differ from the metadata snapshot")
        recomputed_alpha = np.asarray(
            calculate_alpha_map(features, jaya_alpha, saved_weights), dtype=np.float64
        )
        if recomputed_alpha.shape != (49,) or not np.array_equal(alpha, recomputed_alpha):
            raise MetadataVerificationError("Alpha snapshot does not match authenticated feature/Jaya inputs")

        priorities = [
            float(saved_weights["variance"]) * float(item["variance"])
            + float(saved_weights["gradient"]) * float(item["gradient"])
            + float(saved_weights["entropy"]) * float(item["entropy"])
            + float(saved_weights["roi"]) * float(item["roi_score"])
            for item in features
        ]
        if not validate_permutation(mappings, key, priorities, Path(log_path)):
            raise MetadataVerificationError("Permutation map validation failed")
        from src.recoverable.adaptive_perturbation import permutation_checksum

        if metadata.get("permutation_checksum") != permutation_checksum(mappings):
            raise MetadataVerificationError("Permutation checksum differs from authenticated metadata")
        if not metadata.get("algorithm"):
            raise MetadataVerificationError("Recovery metadata is missing transform details")
    except (KeyError, TypeError, ValueError) as error:
        logger.exception("Recovery metadata verification failed")
        if isinstance(error, MetadataVerificationError):
            raise
        raise MetadataVerificationError(f"Recovery metadata verification failed: {error}") from error
    logger.info("Recovery metadata HMAC and artifact bindings verified")
    return True
