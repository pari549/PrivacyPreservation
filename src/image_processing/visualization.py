"""Visualizations pairing source images with their block maps."""
from pathlib import Path
import os
from src.preprocessing.data_loader import PROJECT_ROOT

MATPLOTLIB_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def save_block_figures(image: np.ndarray, features: list[dict], output_dir: Path, sample_id: str) -> list[Path]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fields = {"variance_heatmap.png": ("variance", "Block variance", 0, .25), "gradient_heatmap.png": ("gradient", "Gradient magnitude", 0, 1), "entropy_heatmap.png": ("entropy", "Entropy (bits)", 0, 4), "roi_heatmap.png": ("roi_score", "ROI score", 0, 1)}
    paths = []
    for filename, (field, title, low, high) in fields.items():
        heat = np.zeros((image.shape[0] // 4, image.shape[1] // 4), dtype=np.float32)
        for record in features: heat[int(record["row"]) // 4, int(record["column"]) // 4] = float(record[field])
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(image); axes[0].set_title(f"{sample_id}: original image"); axes[0].axis("off")
        plot = axes[1].imshow(heat, cmap="viridis", vmin=low, vmax=high, interpolation="nearest")
        axes[1].set_title(f"{sample_id}: {title}"); axes[1].set_xlabel("Block column"); axes[1].set_ylabel("Block row")
        fig.colorbar(plot, ax=axes[1], label=title); fig.tight_layout()
        path = output_dir / filename; fig.savefig(path, dpi=200); plt.close(fig); paths.append(path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(image); axes[0].set_title(f"{sample_id}: original image"); axes[0].axis("off")
    axes[1].imshow(image)
    for p in range(0, image.shape[0], 4): axes[1].axhline(p - .5, color="yellow", linewidth=.5)
    for p in range(0, image.shape[1], 4): axes[1].axvline(p - .5, color="yellow", linewidth=.5)
    axes[1].set_title(f"{sample_id}: 4×4 block grid"); axes[1].set_xlabel("Pixel column"); axes[1].set_ylabel("Pixel row")
    fig.tight_layout(); path = output_dir / "block_grid.png"; fig.savefig(path, dpi=200); plt.close(fig); paths.append(path)
    return paths
