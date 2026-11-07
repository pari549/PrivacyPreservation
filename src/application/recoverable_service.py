"""Application service for isolated authenticated recoverable image runs.

This adapter supports only decoded 28x28 RGB images because the existing
recoverable transform requires exactly that geometry. Recovery restores those
decoded pixels; it does not restore the uploaded file's original encoding.
The configured storage root must be private application storage.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import threading
import uuid

import numpy as np
from PIL import Image

from src.non_recoverable.image_input import load_image


class RecoverableServiceError(RuntimeError):
    """Raised when a recoverable service operation cannot safely complete."""


@dataclass(frozen=True)
class RecoverableTransformResult:
    """Public output of transform; internal recovery artifacts are not exposed."""

    transformed_image_bytes: bytes
    run_id: str
    metadata: dict[str, object]
    mime_type: str = "image/png"


@dataclass(frozen=True)
class RecoverableImageResult:
    """Public output of exact recovery; reports and source files stay private."""

    recovered_image_bytes: bytes
    run_id: str
    pixel_perfect: bool
    dimensions: tuple[int, int]
    mime_type: str = "image/png"


class RecoverableImageService:
    """Run the existing v2 transformation and recovery within private storage.

    ``key_path`` is mandatory and is never inferred from project defaults.
    The expected run ID is retained in this service instance's private registry;
    callers should keep the service in trusted application state. Registry state
    is intentionally process-local, so a restart requires a separately designed
    protected registry before prior runs can be recovered.
    """

    def __init__(
        self,
        storage_root: str | Path,
        *,
        key_path: str | Path,
        jaya_population_size: int = 20,
        jaya_iterations: int = 100,
        jaya_seed: int = 42,
        max_image_pixels: int = 784,
    ) -> None:
        if key_path is None:
            raise ValueError("An explicitly configured key_path is required")
        if isinstance(jaya_population_size, bool) or not isinstance(jaya_population_size, int) or jaya_population_size < 2:
            raise ValueError("jaya_population_size must be an integer of at least 2")
        if isinstance(jaya_iterations, bool) or not isinstance(jaya_iterations, int) or jaya_iterations < 1:
            raise ValueError("jaya_iterations must be a positive integer")
        if isinstance(jaya_seed, bool) or not isinstance(jaya_seed, int) or jaya_seed < 0:
            raise ValueError("jaya_seed must be a non-negative integer")
        if isinstance(max_image_pixels, bool) or not isinstance(max_image_pixels, int) or max_image_pixels < 784:
            raise ValueError("max_image_pixels must be an integer of at least 784")
        self._storage_root = Path(storage_root).expanduser().resolve()
        self._runs_root = self._storage_root / "runs"
        self._key_path = Path(key_path).expanduser().resolve()
        self._jaya_population_size = jaya_population_size
        self._jaya_iterations = jaya_iterations
        self._jaya_seed = jaya_seed
        self._max_image_pixels = max_image_pixels
        self._runs: dict[str, dict[str, Path | str]] = {}
        self._lock = threading.RLock()

    def transform(self, uploaded_image: bytes | bytearray | memoryview) -> RecoverableTransformResult:
        """Create a new authenticated run from an uploaded image byte string."""
        if not isinstance(uploaded_image, (bytes, bytearray, memoryview)):
            raise TypeError("uploaded_image must be bytes-like")
        content = bytes(uploaded_image)
        if not content:
            raise ValueError("uploaded_image is empty")
        self._ensure_storage()
        run_id = uuid.uuid4().hex
        run_dir = self._runs_root / run_id
        run_dir.mkdir(exist_ok=False)
        try:
            return self._create_run(content, run_id, run_dir)
        except Exception as error:
            self._remove_run_directory(run_dir)
            if isinstance(error, (ValueError, TypeError)):
                raise
            raise RecoverableServiceError(
                f"Recoverable transformation failed: {type(error).__name__}: {error}"
            ) from error

    def recover(self, run_id: str) -> RecoverableImageResult:
        """Recover a registered run using trusted service state and the set key."""
        run_dir, expected_run_id = self._resolve_run(run_id)
        attempt_id = uuid.uuid4().hex
        attempt_dir = run_dir / "recovery_attempts" / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=False)
        try:
            from src.recoverable.recovery_engine import run_recovery

            paths = self._artifact_paths(run_dir)
            source_dataset = np.load(paths["reference"], allow_pickle=False)
            if source_dataset.shape != (1, 28, 28, 3) or source_dataset.dtype != np.float32:
                raise RecoverableServiceError("Retained source image has invalid dimensions or dtype")
            source = np.array(source_dataset[0], copy=True)
            report_path = attempt_dir / "recovery_report.json"
            output_dir = attempt_dir / "outputs"
            recovery = run_recovery(
                key_path=self._key_path,
                permutation_path=paths["permutation"],
                metadata_path=paths["metadata"],
                perturbed_path=paths["perturbed"],
                jaya_path=paths["jaya"],
                feature_matrix_path=paths["features"],
                reference_path=paths["reference"],
                report_path=report_path,
                output_dir=output_dir,
                log_path=attempt_dir / "recovery.log",
                expected_run_id=expected_run_id,
            )
            recovered = np.asarray(recovery["recovered_image"])
            expected_pixels = np.rint(source * np.float32(255.0)).astype(np.uint8)
            if recovered.shape != (28, 28, 3) or recovered.dtype != np.uint8 or not np.array_equal(recovered, expected_pixels):
                raise RecoverableServiceError("Recovered pixels do not exactly match the retained decoded source")
            encoded = _encode_png(recovered)
            return RecoverableImageResult(
                recovered_image_bytes=encoded,
                run_id=run_id,
                pixel_perfect=True,
                dimensions=(28, 28),
            )
        except Exception as error:
            shutil.rmtree(attempt_dir, ignore_errors=True)
            if isinstance(error, (ValueError, TypeError, RecoverableServiceError)):
                raise
            raise RecoverableServiceError(
                f"Authenticated recovery failed: {type(error).__name__}: {error}"
            ) from error

    def _create_run(self, content: bytes, run_id: str, run_dir: Path) -> RecoverableTransformResult:
        upload_path = run_dir / "upload.tmp"
        upload_path.write_bytes(content)
        try:
            prepared = load_image(
                upload_path,
                block_size=4,
                grayscale_image=False,
                max_image_pixels=self._max_image_pixels,
            )
        finally:
            upload_path.unlink(missing_ok=True)
        if prepared.image.shape != (28, 28, 3) or prepared.image.dtype != np.float32:
            raise ValueError("Recoverable mode accepts only decoded 28x28 RGB images; uploads are not resized")
        if prepared.metadata.get("source_mode") in {"1", "L", "LA", "I", "I;16", "I;16B", "I;16L", "F"}:
            raise ValueError("Recoverable mode rejects grayscale source images")
        source_u8 = np.rint(prepared.image * np.float32(255.0)).astype(np.uint8)
        if not np.all(np.isfinite(prepared.image)) or np.any(prepared.image < 0) or np.any(prepared.image > 1):
            raise ValueError("Decoded source must contain finite normalized RGB pixels")

        paths = self._artifact_paths(run_dir)
        paths["processed"].mkdir(parents=True)
        paths["reference"].parent.mkdir(parents=True)
        np.save(paths["reference"], prepared.image[np.newaxis, ...], allow_pickle=False)

        from src.non_recoverable.image_input import extract_image_block_features

        feature_records = extract_image_block_features(prepared)
        if len(feature_records) != 49:
            raise ValueError("The recoverable transform requires exactly 49 feature blocks")
        feature_document = {"samples": [{"source_index": 0, "features": feature_records}]}
        _write_json(paths["features"], feature_document)
        _write_json(paths["key_metadata"], {"algorithm": "secrets", "key_length": 128})

        from src.optimization.run_jaya import run_jaya

        run_jaya(
            input_path=paths["features"],
            result_path=paths["jaya"],
            image_dir=paths["jaya_plots"],
            log_path=paths["transform_log"],
            population_size=self._jaya_population_size,
            iterations=self._jaya_iterations,
            seed=self._jaya_seed,
        )
        from src.recoverable.block_permutation import run_block_permutation

        run_block_permutation(
            processed_dir=paths["processed"],
            key_path=self._key_path,
            key_metadata_path=paths["key_metadata"],
            feature_matrix_path=paths["features"],
            jaya_results_path=paths["jaya"],
            map_path=paths["permutation"],
            visualization_dir=paths["permutation_plots"],
            log_path=paths["transform_log"],
        )
        from src.recoverable.adaptive_perturbation import run_adaptive_perturbation

        run_adaptive_perturbation(
            processed_dir=paths["processed"],
            key_path=self._key_path,
            permutation_path=paths["permutation"],
            feature_matrix_path=paths["features"],
            jaya_results_path=paths["jaya"],
            metadata_path=paths["metadata"],
            output_path=paths["perturbed"],
            visualization_dir=paths["perturbation_plots"],
            log_path=paths["transform_log"],
            run_id=run_id,
        )
        perturbed = np.load(paths["perturbed"], allow_pickle=False)
        if perturbed.shape != (28, 28, 3) or perturbed.dtype != np.uint8:
            raise RecoverableServiceError("Existing transform returned an invalid perturbed image")
        with self._lock:
            self._runs[run_id] = {"directory": run_dir, "expected_run_id": run_id}
        safe_metadata: dict[str, object] = {
            "run_id": run_id,
            "schema_version": 2,
            "dimensions": [28, 28],
            "channels": 3,
            "block_size": 4,
            "block_count": 49,
            "source_format": prepared.metadata.get("source_format"),
            "source_mode": prepared.metadata.get("source_mode"),
            "preprocessing": {
                "orientation_applied": prepared.metadata.get("exif_orientation_applied"),
                "channel_conversion": prepared.metadata.get("channel_conversion_policy"),
                "alpha_handling": prepared.metadata.get("alpha_handling_policy"),
                "background_color_rgb": prepared.metadata.get("background_color_rgb"),
                "normalization": prepared.metadata.get("normalization"),
                "normalized_value_range": prepared.metadata.get("normalized_value_range"),
            },
            "transformation": "authenticated deterministic modulo-256 perturbation and 4x4 block permutation",
            "limitations": [
                "Recovery restores the decoded 28x28 RGB pixel array, not original upload bytes.",
                "HMAC authenticates metadata and artifact bindings; it does not encrypt stored artifacts.",
                "Jaya uses existing illustrative proxy objectives, not measured privacy attack results.",
            ],
        }
        return RecoverableTransformResult(_encode_png(perturbed), run_id, safe_metadata)

    def _ensure_storage(self) -> None:
        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._runs_root.mkdir(parents=True, exist_ok=True)
        if self._storage_root.is_symlink() or self._runs_root.is_symlink():
            raise RecoverableServiceError("Application storage directories must not be symlinks")

    def _resolve_run(self, run_id: str) -> tuple[Path, str]:
        if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
            raise ValueError("run_id must be a service-issued identifier")
        with self._lock:
            record = self._runs.get(run_id)
        if record is None or record.get("expected_run_id") != run_id:
            raise KeyError("Unknown or unregistered run_id")
        run_dir = Path(record["directory"])
        resolved = run_dir.resolve(strict=True)
        if resolved.parent != self._runs_root.resolve(strict=True) or run_dir.is_symlink():
            raise RecoverableServiceError("Registered run directory failed storage isolation checks")
        return resolved, str(record["expected_run_id"])

    @staticmethod
    def _remove_run_directory(run_dir: Path) -> None:
        try:
            resolved = run_dir.resolve(strict=False)
            expected_parent = run_dir.parent.resolve(strict=True)
            if resolved.parent == expected_parent and not run_dir.is_symlink():
                shutil.rmtree(resolved, ignore_errors=True)
        except OSError:
            pass

    @staticmethod
    def _artifact_paths(run_dir: Path) -> dict[str, Path]:
        processed = run_dir / "processed"
        return {
            "processed": processed,
            "reference": processed / "train" / "images.npy",
            "features": run_dir / "feature_matrix.json",
            "jaya": run_dir / "jaya_results.json",
            "permutation": run_dir / "permutation_map.json",
            "key_metadata": run_dir / "key_metadata.json",
            "metadata": run_dir / "recovery_metadata.json",
            "perturbed": run_dir / "perturbed_image.npy",
            "jaya_plots": run_dir / "jaya_plots",
            "permutation_plots": run_dir / "permutation_plots",
            "perturbation_plots": run_dir / "perturbation_plots",
            "transform_log": run_dir / "transform.log",
        }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _encode_png(image: np.ndarray) -> bytes:
    pixels = np.asarray(image)
    if pixels.shape != (28, 28, 3) or pixels.dtype != np.uint8:
        raise ValueError("PNG output requires a uint8 28x28 RGB array")
    output = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(output, format="PNG")
    return output.getvalue()
