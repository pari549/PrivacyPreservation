"""Block image statistics."""
import numpy as np

def block_statistics(block: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(block, dtype=np.float32)
    return float(values.mean()), float(values.var()), float(values.std())
