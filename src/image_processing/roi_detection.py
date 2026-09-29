"""ROI score from Sobel, Canny and Otsu cues."""
import cv2
import numpy as np
from src.image_processing.gradients import grayscale

def roi_score(block: np.ndarray, sobel_mean: float, global_otsu: float) -> float:
    gray = grayscale(block)
    byte_gray = np.clip(np.rint(gray * 255), 0, 255).astype(np.uint8)
    canny = float(np.count_nonzero(cv2.Canny(byte_gray, 50, 150)) / byte_gray.size)
    foreground = float(np.mean(gray >= global_otsu))
    return float(np.clip((np.clip(sobel_mean, 0, 1) + canny + foreground) / 3, 0, 1))
