"""Sobel gradients and edge density."""
import cv2
import numpy as np

def grayscale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gradient_maps(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = grayscale(image)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=.25)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=.25)
    return gx, gy, cv2.magnitude(gx, gy)

def edge_density(block: np.ndarray) -> float:
    gray = np.clip(np.rint(grayscale(block) * 255), 0, 255).astype(np.uint8)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges) / edges.size)
