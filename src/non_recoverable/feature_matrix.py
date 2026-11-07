"""Convert prepared-image block features into a validated numeric matrix."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.non_recoverable.image_input import PreparedImage, extract_image_block_features


DEFAULT_FEATURE_NAMES = (
    "mean",
    "variance",
    "std",
    "gradient",
    "edge_density",
    "entropy",
    "roi_score",
)


@dataclass(frozen=True)
class ImageFeatureMatrix:
    """Numeric block features and their geometry for one prepared image.

    ``values`` has one row per block in row-major block order and one column
    per ``feature_names`` entry. ``blocks`` records the corresponding block
    coordinates and valid source-pixel counts, including partial edge blocks.
    """

    values: np.ndarray
    feature_names: tuple[str, ...]
    blocks: tuple[dict[str, int], ...]
    image_metadata: dict[str, object]


def extract_feature_matrix(
    prepared: PreparedImage,
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
) -> ImageFeatureMatrix:
    """Extract selected Stage 3A block features as a finite ``float64`` matrix.

    The underlying feature calculations remain in
    :func:`extract_image_block_features`; partial-block values therefore keep
    its documented valid-source-pixel and boundary policies. This function
    only selects, orders, and validates feature columns.
    """
    names = tuple(feature_names)
    if not names:
        raise ValueError("feature_names must contain at least one feature")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must not contain duplicates")

    if not isinstance(prepared, PreparedImage):
        raise TypeError("prepared must be a PreparedImage")
    records = extract_image_block_features(prepared)
    if not records:
        raise ValueError("Prepared image produced no blocks")

    missing = [name for name in names if any(name not in row for row in records)]
    if missing:
        raise ValueError(f"Unknown block feature(s): {', '.join(missing)}")

    try:
        raw_values = [[row[name] for name in names] for row in records]
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
            for row in raw_values
            for value in row
        ):
            raise ValueError("Feature values must be numeric scalars")
        values = np.asarray(raw_values, dtype=np.float64)
    except (TypeError, OverflowError) as error:
        raise ValueError("Feature values must form a numeric matrix") from error
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Feature matrix must contain only finite numeric values")

    block_fields = ("block_id", "row", "column", "valid_height", "valid_width", "valid_pixel_count")
    try:
        blocks = tuple(
            {
                field: _require_integer(record[field], field)
                for field in block_fields
            }
            for record in records
        )
    except KeyError as error:
        raise ValueError(f"Block record is missing required field: {error.args[0]}") from error

    block_size = prepared.metadata.get("block_size")
    height = prepared.metadata.get("original_height")
    width = prepared.metadata.get("original_width")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (block_size, height, width)
    ):
        raise ValueError("Prepared image metadata has invalid block geometry")
    expected_coordinates = [
        (row, column)
        for row in range(0, int(height), int(block_size))
        for column in range(0, int(width), int(block_size))
    ]
    if len(blocks) != len(expected_coordinates):
        raise ValueError("Block record count does not match prepared image geometry")
    for index, (block, (expected_row, expected_column)) in enumerate(
        zip(blocks, expected_coordinates)
    ):
        expected_height = min(int(block_size), int(height) - expected_row)
        expected_width = min(int(block_size), int(width) - expected_column)
        if (
            block["block_id"] != index
            or (block["row"], block["column"]) != (expected_row, expected_column)
            or (block["valid_height"], block["valid_width"])
            != (expected_height, expected_width)
            or block["valid_pixel_count"] != expected_height * expected_width
        ):
            raise ValueError("Block records are inconsistent with prepared image geometry")

    return ImageFeatureMatrix(
        values=values,
        feature_names=names,
        blocks=blocks,
        image_metadata=dict(prepared.metadata),
    )


def _require_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Block field {name} must be an integer")
    return int(value)
