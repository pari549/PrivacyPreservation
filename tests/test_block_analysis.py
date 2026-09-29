"""Validation tests for block splitting and feature extraction."""
import unittest
import numpy as np
from src.image_processing.block_splitter import split_into_blocks, validate_block_coverage
from src.image_processing.feature_matrix import extract_block_features

class BlockAnalysisTest(unittest.TestCase):
    def test_complete_block_feature_map(self) -> None:
        image = np.random.default_rng(5).random((28, 28, 3), dtype=np.float32)
        features = extract_block_features(image)
        self.assertEqual(len(list(split_into_blocks(image, 4))), 49)
        self.assertTrue(validate_block_coverage(28, 28, 4))
        self.assertEqual(len(features), 49)
        self.assertEqual(len(features[0]), 10)
        self.assertTrue(all(0 <= item["roi_score"] <= 1 for item in features))
        self.assertTrue(all(0 <= item["entropy"] <= 4 for item in features))

if __name__ == "__main__": unittest.main()
