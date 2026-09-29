"""Shannon entropy for normalized image blocks."""
import numpy as np
from src.image_processing.gradients import grayscale

def block_entropy(block: np.ndarray) -> float:
    gray = grayscale(block)
    values = np.clip(np.rint(gray * 255), 0, 255).astype(np.uint8)
    counts = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / values.size
    return float(-np.sum(probabilities * np.log2(probabilities)))
