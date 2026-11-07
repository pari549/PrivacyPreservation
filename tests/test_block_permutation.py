"""Tests for deterministic reversible block permutation and map validation."""
import unittest
from pathlib import Path
import numpy as np
from tests.recovery_test_support import temporary_test_directory

from src.recoverable.block_permutation import (
    generate_permutation,
    invert_permutation,
    permute_blocks,
)
from src.recoverable.permutation_validator import (
    PermutationValidationError,
    validate_permutation,
)


class BlockPermutationTest(unittest.TestCase):
    def test_keyed_bijection_checksum_replay_and_outputs(self) -> None:
        log_path = temporary_test_directory(self) / "permutation.log"
        self._check_permutation(log_path)

    def _check_permutation(self, log_path: Path) -> None:
        priorities = np.linspace(0.0, 1.0, 49).tolist()
        key_a = bytes(range(16))
        key_b = bytes(reversed(range(16)))
        first = generate_permutation(key_a, priorities)
        self.assertEqual(first, generate_permutation(key_a, priorities))
        second = generate_permutation(key_b, priorities)
        self.assertNotEqual([item["to"] for item in first], [item["to"] for item in second])
        self.assertEqual(len(first), 49)
        self.assertEqual(len({tuple(item["from"]) for item in first}), 49)
        self.assertEqual(len({tuple(item["to"]) for item in first}), 49)
        self.assertTrue(validate_permutation(first, key_a, priorities, log_path))
        corrupted = [dict(item) for item in first]
        corrupted[0]["to"] = list(corrupted[1]["to"])
        with self.assertRaises(PermutationValidationError):
            validate_permutation(corrupted, key_a, priorities, log_path)

        image = np.arange(28 * 28 * 3, dtype=np.uint8).reshape(28, 28, 3)
        restored = permute_blocks(permute_blocks(image, first), invert_permutation(first))
        np.testing.assert_array_equal(restored, image)



if __name__ == "__main__":
    unittest.main()
