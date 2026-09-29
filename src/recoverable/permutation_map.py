"""Canonical permutation-map serialization and per-mapping checksums."""
from __future__ import annotations
import hashlib
import json

MAP_FIELDS = ("block_id", "from", "to", "priority")

def mapping_checksum(mapping: dict) -> str:
    """Return SHA-256 over a canonical representation excluding checksum."""
    payload = {field: mapping[field] for field in MAP_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def seal_mapping(mapping: dict) -> dict:
    """Copy a mapping record and attach its canonical SHA-256 checksum."""
    sealed = {field: mapping[field] for field in MAP_FIELDS}
    sealed["checksum"] = mapping_checksum(sealed)
    return sealed
