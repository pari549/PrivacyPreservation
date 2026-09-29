"""Non-overlapping full-coverage image tiling."""
from collections.abc import Iterator
import numpy as np

def split_into_blocks(image: np.ndarray, block_size: int = 4) -> Iterator[tuple[int, int, np.ndarray]]:
    image = np.asarray(image)
    if image.ndim not in (2, 3) or block_size <= 0:
        raise ValueError("Expected 2D/3D image and positive block size")
    height, width = image.shape[:2]
    if height % block_size or width % block_size:
        raise ValueError("Image dimensions must be divisible by block size")
    for row in range(0, height, block_size):
        for column in range(0, width, block_size):
            yield row, column, image[row:row+block_size, column:column+block_size]

def validate_block_coverage(height: int, width: int, block_size: int = 4) -> bool:
    if height % block_size or width % block_size:
        raise ValueError("Image dimensions must be divisible by block size")
    coverage = np.zeros((height, width), dtype=np.uint8)
    for row in range(0, height, block_size):
        for col in range(0, width, block_size): coverage[row:row+block_size, col:col+block_size] += 1
    if not np.all(coverage == 1): raise ValueError("Missing or overlapping pixels")
    return True
