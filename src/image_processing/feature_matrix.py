"""Block-level feature extraction and representative sample export."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from src.image_processing.block_splitter import split_into_blocks, validate_block_coverage
from src.image_processing.entropy import block_entropy
from src.image_processing.gradients import edge_density, gradient_maps, grayscale
from src.image_processing.roi_detection import roi_score
from src.image_processing.variance import block_statistics
from src.image_processing.visualization import save_block_figures
from src.preprocessing.data_loader import DEFAULT_LOG_PATH, PROJECT_ROOT, get_dataset_logger
from src.preprocessing.normalize import DEFAULT_PROCESSED_DIR

OUTPUT_DIR = PROJECT_ROOT / "results" / "images" / "block_analysis"
MATRIX_PATH = PROJECT_ROOT / "results" / "metrics" / "block_feature_matrix.json"

def extract_block_features(image: np.ndarray) -> list[dict[str, float | int]]:
    image = np.asarray(image)
    if image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3 or image.min() < 0 or image.max() > 1:
        raise ValueError("Expected normalized float32 RGB image in [0, 1]")
    height, width = image.shape[:2]
    validate_block_coverage(height, width, 4)
    gray = grayscale(image)
    _, _, magnitude = gradient_maps(image)
    magnitude = np.clip(magnitude, 0, 1)
    byte_gray = np.clip(np.rint(gray * 255), 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(byte_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(otsu / 255)
    records = []
    for block_id, (row, col, block) in enumerate(split_into_blocks(image, 4)):
        mean, variance, std = block_statistics(block)
        gradient = float(magnitude[row:row+4, col:col+4].mean())
        records.append({"block_id": block_id, "row": row, "column": col, "mean": mean,
                        "variance": variance, "std": std, "gradient": gradient,
                        "edge_density": edge_density(block), "entropy": block_entropy(block),
                        "roi_score": roi_score(block, gradient, threshold)})
    if len(records) != (height // 4) * (width // 4): raise ValueError("Incomplete block map")
    return records

def generate_feature_matrix(processed_dir: Path = DEFAULT_PROCESSED_DIR, output_dir: Path = OUTPUT_DIR,
                            matrix_path: Path = MATRIX_PATH, log_path: Path = DEFAULT_LOG_PATH,
                            sample_count: int = 5) -> dict:
    logger = get_dataset_logger(log_path)
    logger.info("Starting block feature analysis.")
    processed_dir, output_dir, matrix_path = Path(processed_dir), Path(output_dir), Path(matrix_path)
    images = np.load(processed_dir / "train" / "images.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(processed_dir / "train" / "labels.npy", mmap_mode="r", allow_pickle=False).reshape(-1)
    if len(images) != len(labels) or not 0 < sample_count <= len(images): raise ValueError("Invalid train data/sample count")
    indices = np.linspace(0, len(images)-1, sample_count, dtype=np.int64)
    samples = []; output_dir.mkdir(parents=True, exist_ok=True)
    for number, index in enumerate(indices):
        image = np.asarray(images[index]); features = extract_block_features(image)
        samples.append({"sample_id": number, "source_index": int(index), "label": int(labels[index]), "features": features})
        save_block_figures(image, features, output_dir, f"sample_{number:02d}")
        logger.info("Extracted %d blocks for training image %d", len(features), int(index))
    sizes = {len(sample["features"]) for sample in samples}
    if len(sizes) != 1: raise ValueError("Feature count differs between sample images")
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"block_size": 4, "feature_names": ["mean", "variance", "std", "gradient", "edge_density", "entropy", "roi_score"], "samples": samples}
    matrix_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("Block feature matrix saved to %s", matrix_path)
    return {**result, "blocks_per_image": sizes.pop()}

if __name__ == "__main__": generate_feature_matrix()
