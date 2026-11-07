"""Versioned HMAC authentication helpers for recovery metadata and artifacts."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path

KEY_BYTES = 16
AUTHENTICATION_CONTEXT = b"PrivacyPreservation/recovery-metadata/v2"
METADATA_HMAC_FIELD = "metadata_hmac"


def derive_authentication_key(secret_key: bytes) -> bytes:
    """Derive a version-specific HMAC key from the project's 16-byte secret.

    The context separates metadata authentication from other uses of the
    master key. Neither the supplied key nor the derived key is logged.
    """
    if not isinstance(secret_key, bytes) or len(secret_key) != KEY_BYTES:
        raise ValueError(f"Secret key must be exactly {KEY_BYTES} bytes")
    return hmac.new(secret_key, AUTHENTICATION_CONTEXT, hashlib.sha256).digest()


def canonicalize_metadata(metadata: dict) -> bytes:
    """Serialize metadata deterministically as UTF-8 JSON, excluding its tag.

    Dictionary keys are sorted and separators are compact. JSON's non-standard
    NaN and infinity values are rejected so all accepted values have a defined
    JSON representation.
    """
    def validate_json_value(value: object, location: str = "metadata") -> None:
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"Non-finite number at {location}")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                validate_json_value(item, f"{location}[{index}]")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"Non-string dictionary key at {location}")
                validate_json_value(item, f"{location}.{key}")
            return
        raise ValueError(f"Unsupported value type at {location}: {type(value).__name__}")

    if not isinstance(metadata, dict):
        raise TypeError("Metadata must be a dictionary")
    validate_json_value(metadata)
    unsigned = {
        key: value
        for key, value in metadata.items()
        if key != METADATA_HMAC_FIELD
    }
    try:
        serialized = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        encoded = serialized.encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError(f"Metadata cannot be canonically serialized: {error}") from error
    return encoded


def generate_metadata_hmac(metadata: dict, secret_key: bytes) -> str:
    """Return a lowercase hexadecimal HMAC-SHA256 tag for metadata."""
    authentication_key = derive_authentication_key(secret_key)
    return hmac.new(
        authentication_key, canonicalize_metadata(metadata), hashlib.sha256
    ).hexdigest()


def verify_metadata_hmac(metadata: dict, secret_key: bytes) -> bool:
    """Verify metadata's HMAC, raising ``ValueError`` for invalid or absent tags."""
    if not isinstance(metadata, dict):
        raise TypeError("Metadata must be a dictionary")
    stored_tag = metadata.get(METADATA_HMAC_FIELD)
    if not isinstance(stored_tag, str) or len(stored_tag) != hashlib.sha256().digest_size * 2:
        raise ValueError("Metadata HMAC is missing or malformed")
    try:
        bytes.fromhex(stored_tag)
    except ValueError as error:
        raise ValueError("Metadata HMAC is missing or malformed") from error

    expected_tag = generate_metadata_hmac(metadata, secret_key)
    if not hmac.compare_digest(stored_tag, expected_tag):
        raise ValueError("Metadata HMAC verification failed")
    return True


def calculate_file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_bytes_sha256(content: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of an already-read byte string."""
    return hashlib.sha256(content).hexdigest()
