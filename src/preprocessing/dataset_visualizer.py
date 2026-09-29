"""Public entry point for PathMNIST visualization and analytics."""

from src.preprocessing.dataset_statistics import (
    DEFAULT_STATISTICS_PATH,
    PREVIEW_DIR,
    generate_dataset_statistics,
)

__all__ = ["DEFAULT_STATISTICS_PATH", "PREVIEW_DIR", "generate_dataset_statistics"]


if __name__ == "__main__":
    generate_dataset_statistics()
