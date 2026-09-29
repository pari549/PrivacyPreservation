"""Structural, checksum and keyed-replay validation for block permutations."""
from __future__ import annotations
from pathlib import Path
from src.recoverable.block_permutation import BLOCK_COUNT, GRID_SIZE, generate_permutation
from src.recoverable.permutation_map import mapping_checksum
from src.preprocessing.data_loader import DEFAULT_LOG_PATH, get_dataset_logger

class PermutationValidationError(ValueError):
    """Raised when a permutation map is incomplete, corrupt, or non-reproducible."""

def validate_permutation(mappings: list[dict], key: bytes, priorities: list[float],
                         log_path: Path = DEFAULT_LOG_PATH) -> bool:
    logger = get_dataset_logger(Path(log_path))
    try:
        if len(mappings) != BLOCK_COUNT:
            raise PermutationValidationError(f"Expected 49 mappings, found {len(mappings)}")
        ids = [item.get("block_id") for item in mappings]
        sources = [tuple(item.get("from", ())) for item in mappings]
        destinations = [tuple(item.get("to", ())) for item in mappings]
        expected = {(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE)}
        if len(set(ids)) != BLOCK_COUNT or set(ids) != set(range(BLOCK_COUNT)):
            raise PermutationValidationError("Block IDs are duplicated or missing")
        if len(set(sources)) != BLOCK_COUNT or set(sources) != expected:
            raise PermutationValidationError("Source coordinates are duplicated or missing")
        if len(set(destinations)) != BLOCK_COUNT or set(destinations) != expected:
            raise PermutationValidationError("Destination coordinates are duplicated or missing")
        for item in mappings:
            if item.get("from") != [item["block_id"] // GRID_SIZE, item["block_id"] % GRID_SIZE]:
                raise PermutationValidationError("Block ID does not match its source coordinate")
            if item.get("checksum") != mapping_checksum(item):
                raise PermutationValidationError(f"Checksum mismatch for block {item['block_id']}")
        replay = generate_permutation(key, priorities)
        replay_destinations = {item["block_id"]: item["to"] for item in replay}
        actual_destinations = {item["block_id"]: item["to"] for item in mappings}
        if actual_destinations != replay_destinations:
            raise PermutationValidationError("Deterministic replay did not reproduce the mapping")
    except (KeyError, TypeError, ValueError) as error:
        logger.exception("Block permutation validation failed")
        if isinstance(error, PermutationValidationError):
            raise
        raise PermutationValidationError(f"Invalid permutation map: {error}") from error
    logger.info("Validated all 49 block mappings, checksums, and deterministic replay")
    return True
