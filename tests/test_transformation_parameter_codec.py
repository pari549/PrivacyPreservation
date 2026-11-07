import unittest

import numpy as np

from src.non_recoverable.transformation_parameter_codec import (
    INITIAL_8BIT_QUANTIZATION_MENU,
    TransformationParameterCodec,
)


class TransformationParameterCodecTest(unittest.TestCase):
    def setUp(self):
        self.codec = TransformationParameterCodec(
            max_sigma=2.0, experiment_bit_depth=8
        )

    def test_menu_endpoints_and_encode_decode_round_trip(self):
        for index, level in enumerate(INITIAL_8BIT_QUANTIZATION_MENU):
            sigma = index / 7 * 2.0
            vector = self.codec.encode(level, sigma)
            decoded = self.codec.decode(vector)
            self.assertEqual(decoded.quantization_levels, level)
            self.assertAlmostEqual(decoded.smoothing_strength, sigma)
        self.assertEqual(self.codec.decode([0.0, 0.0]).quantization_levels, 2)
        self.assertEqual(self.codec.decode([1.0, 1.0]).quantization_levels, 256)

    def test_round_half_to_even_mapping_is_deterministic(self):
        # q * 7 = .5 rounds to even zero; 1.5 rounds to even index two.
        self.assertEqual(self.codec.decode([0.5 / 7, 0]).quantization_levels, 2)
        self.assertEqual(self.codec.decode([1.5 / 7, 0]).quantization_levels, 8)
        self.assertEqual(self.codec.decode([0.4, 0.5]), self.codec.decode([0.4, 0.5]))

    def test_duplicate_phenotypes_share_canonical_key(self):
        first = [0.0, 0.25]
        second = [0.01, 0.25]
        self.assertTrue(self.codec.are_duplicate_configurations(first, second))
        self.assertEqual(self.codec.configuration_key(first), self.codec.configuration_key(second))

    def test_global_values_expand_in_row_major_image_geometry(self):
        expanded = self.codec.expand_for_image(
            [0.5, 0.25], height=5, width=9, block_size=4
        )
        self.assertEqual((expanded.block_rows, expanded.block_columns), (2, 3))
        self.assertEqual(
            expanded.block_coordinates,
            ((0, 0), (0, 4), (0, 8), (4, 0), (4, 4), (4, 8)),
        )
        self.assertEqual(expanded.levels_per_block, (32,) * 6)
        self.assertEqual(expanded.strengths_per_block, (0.5,) * 6)

    def test_invalid_configuration_and_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            TransformationParameterCodec(max_sigma=2.0)
        with self.assertRaises(ValueError):
            TransformationParameterCodec(max_sigma=2.0, experiment_bit_depth=8,
                                         quantization_levels=(2, 2))
        for vector in ([0.5], [0.5, np.nan], [-0.1, 0], [0, 1.1]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                self.codec.decode(vector)
        with self.assertRaises(ValueError):
            self.codec.encode(3, 0.5)
        with self.assertRaises(ValueError):
            self.codec.encode(2, 2.1)
        with self.assertRaises(ValueError):
            self.codec.expand_to_blocks([0.5, 0.5], 0)
        with self.assertRaises(ValueError):
            self.codec.expand_for_image([0.5, 0.5], height=4, width=4, block_size=0)


if __name__ == "__main__":
    unittest.main()
