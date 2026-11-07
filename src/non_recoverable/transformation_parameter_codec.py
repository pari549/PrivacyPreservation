"""Deterministic codec for global quantization and smoothing parameters.

The Stage 3F-1 pilot vector is ``theta = [q_encoded, sigma_encoded]`` in
``[0, 1]^2``. The eight-level menu is intended only for experiments that
explicitly declare an 8-bit source profile; it is not a universal search
recommendation. Sigma bounds are supplied by the experiment configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np

from src.non_recoverable.baseline_transformations import (
    MAX_QUANTIZATION_LEVELS,
    MAX_SMOOTHING_STRENGTH,
    MIN_QUANTIZATION_LEVELS,
)


INITIAL_8BIT_QUANTIZATION_MENU = (2, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class DecodedTransformationConfiguration:
    """Canonical global configuration decoded from a normalized vector."""

    quantization_levels: int
    smoothing_strength: float

    @property
    def canonical_key(self) -> tuple[int, str]:
        """Exact stable key for decoded phenotype deduplication."""
        return self.quantization_levels, self.smoothing_strength.hex()


@dataclass(frozen=True)
class ExpandedBlockParameters:
    """Global settings replicated in Stage 3A row-major block order."""

    block_rows: int
    block_columns: int
    block_coordinates: tuple[tuple[int, int], ...]
    levels_per_block: tuple[int, ...]
    strengths_per_block: tuple[float, ...]


class TransformationParameterCodec:
    """Encode/decode the global quantization and Gaussian-sigma pilot vector.

    Quantization code is mapped to the nearest menu index with NumPy
    round-half-to-even semantics: ``index = rint(q_encoded * (K - 1))``.
    Sigma is linear: ``sigma = sigma_encoded * max_sigma``. Both normalized
    components include endpoints; encoded sigma zero remains exactly zero.
    """

    def __init__(
        self,
        *,
        max_sigma: float,
        quantization_levels: Sequence[int] = INITIAL_8BIT_QUANTIZATION_MENU,
        experiment_bit_depth: int | None = None,
    ) -> None:
        if isinstance(max_sigma, (bool, np.bool_)) or not isinstance(
            max_sigma, (int, float, np.number)
        ):
            raise ValueError("max_sigma must be a finite number")
        if not np.isfinite(max_sigma) or not 0 < max_sigma <= MAX_SMOOTHING_STRENGTH:
            raise ValueError(
                f"max_sigma must be greater than 0 and no greater than {MAX_SMOOTHING_STRENGTH}"
            )
        if isinstance(quantization_levels, (str, bytes)):
            raise ValueError("quantization_levels must be an integer sequence")
        try:
            levels = tuple(quantization_levels)
        except TypeError as error:
            raise ValueError("quantization_levels must be an integer sequence") from error
        if len(levels) < 2:
            raise ValueError("quantization_levels must contain at least two menu entries")
        if any(
            isinstance(level, (bool, np.bool_))
            or not isinstance(level, (int, np.integer))
            or not MIN_QUANTIZATION_LEVELS <= level <= MAX_QUANTIZATION_LEVELS
            for level in levels
        ):
            raise ValueError(
                f"Each menu level must be an integer in [{MIN_QUANTIZATION_LEVELS}, "
                f"{MAX_QUANTIZATION_LEVELS}]"
            )
        normalized_levels = tuple(int(level) for level in levels)
        if tuple(sorted(set(normalized_levels))) != normalized_levels:
            raise ValueError("quantization_levels must be strictly increasing and unique")
        if experiment_bit_depth is not None and (
            isinstance(experiment_bit_depth, bool)
            or not isinstance(experiment_bit_depth, int)
            or experiment_bit_depth <= 0
        ):
            raise ValueError("experiment_bit_depth must be a positive integer or None")
        if (
            normalized_levels == INITIAL_8BIT_QUANTIZATION_MENU
            and experiment_bit_depth != 8
        ):
            raise ValueError(
                "The initial pilot menu requires experiment_bit_depth=8 to be declared explicitly"
            )

        self.quantization_levels = normalized_levels
        self.max_sigma = float(max_sigma)
        self.experiment_bit_depth = experiment_bit_depth

    def _vector(self, theta: Sequence[float] | np.ndarray) -> np.ndarray:
        try:
            vector = np.asarray(theta, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("theta must contain exactly two finite numeric values") from error
        if vector.shape != (2,):
            raise ValueError("theta must have shape (2,): [q_encoded, sigma_encoded]")
        if not np.all(np.isfinite(vector)):
            raise ValueError("theta values must be finite")
        if np.any(vector < 0.0) or np.any(vector > 1.0):
            raise ValueError("theta values must lie in [0, 1]")
        return vector

    def decode(
        self, theta: Sequence[float] | np.ndarray
    ) -> DecodedTransformationConfiguration:
        """Decode a normalized candidate into a canonical discrete/continuous config."""
        vector = self._vector(theta)
        q_encoded, sigma_encoded = map(float, vector)
        menu_index = int(np.rint(q_encoded * (len(self.quantization_levels) - 1)))
        menu_index = min(max(menu_index, 0), len(self.quantization_levels) - 1)
        sigma = 0.0 if sigma_encoded == 0.0 else float(sigma_encoded * self.max_sigma)
        return DecodedTransformationConfiguration(
            quantization_levels=self.quantization_levels[menu_index],
            smoothing_strength=sigma,
        )

    def encode(
        self, quantization_levels: int, smoothing_strength: float
    ) -> np.ndarray:
        """Encode an exactly representable menu level and valid sigma."""
        if (
            isinstance(quantization_levels, (bool, np.bool_))
            or not isinstance(quantization_levels, (int, np.integer))
            or int(quantization_levels) not in self.quantization_levels
        ):
            raise ValueError("quantization_levels must be an entry in this codec's menu")
        if isinstance(smoothing_strength, (bool, np.bool_)) or not isinstance(
            smoothing_strength, (int, float, np.number)
        ):
            raise ValueError("smoothing_strength must be a finite number within [0, max_sigma]")
        if (
            not np.isfinite(smoothing_strength)
            or smoothing_strength < 0
            or smoothing_strength > self.max_sigma
        ):
            raise ValueError("smoothing_strength must be within [0, max_sigma]")
        menu_index = self.quantization_levels.index(int(quantization_levels))
        q_encoded = menu_index / (len(self.quantization_levels) - 1)
        sigma_encoded = float(smoothing_strength) / self.max_sigma
        if smoothing_strength == 0:
            sigma_encoded = 0.0
        return np.asarray([q_encoded, sigma_encoded], dtype=np.float64)

    def configuration_key(
        self, theta: Sequence[float] | np.ndarray
    ) -> tuple[int, str]:
        """Return a deterministic key; candidates with the same decoded config match."""
        return self.decode(theta).canonical_key

    def are_duplicate_configurations(
        self,
        first: Sequence[float] | np.ndarray,
        second: Sequence[float] | np.ndarray,
    ) -> bool:
        """Whether two vectors decode to the same transformation phenotype."""
        return self.configuration_key(first) == self.configuration_key(second)

    def expand_to_blocks(
        self, theta: Sequence[float] | np.ndarray, block_count: int
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Replicate one global config over a caller-supplied row-major block count."""
        if (
            isinstance(block_count, bool)
            or not isinstance(block_count, (int, np.integer))
            or block_count <= 0
        ):
            raise ValueError("block_count must be a positive integer")
        config = self.decode(theta)
        return (
            (config.quantization_levels,) * int(block_count),
            (config.smoothing_strength,) * int(block_count),
        )

    def expand_for_image(
        self,
        theta: Sequence[float] | np.ndarray,
        *,
        height: int,
        width: int,
        block_size: int,
    ) -> ExpandedBlockParameters:
        """Expand a global config using row-major image block geometry.

        The codec itself is image-independent; dimensions are used only here
        to determine the required number and ordering of repeated values.
        """
        for name, value in (("height", height), ("width", width), ("block_size", block_size)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        block_rows = math.ceil(int(height) / int(block_size))
        block_columns = math.ceil(int(width) / int(block_size))
        count = block_rows * block_columns
        levels, strengths = self.expand_to_blocks(theta, count)
        coordinates = tuple(
            (top, left)
            for top in range(0, int(height), int(block_size))
            for left in range(0, int(width), int(block_size))
        )
        return ExpandedBlockParameters(
            block_rows=block_rows,
            block_columns=block_columns,
            block_coordinates=coordinates,
            levels_per_block=levels,
            strengths_per_block=strengths,
        )
