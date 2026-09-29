"""Generate and persist a cryptographically secure 128-bit secret key."""
from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from src.preprocessing.data_loader import DEFAULT_LOG_PATH, PROJECT_ROOT, get_dataset_logger

DEFAULT_KEY_PATH = PROJECT_ROOT / "keys" / "secret_key.bin"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "keys" / "key_metadata.json"
KEY_BYTES = 16


class KeyExistsError(FileExistsError):
    """Raised when key generation would replace a key without confirmation."""


def generate_key(
    key_path: Path = DEFAULT_KEY_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    *,
    confirm_overwrite: bool = False,
    log_path: Path = DEFAULT_LOG_PATH,
) -> bytes:
    """Generate and save 16 random bytes, refusing unconfirmed replacement.

    ``confirm_overwrite=True`` explicitly authorizes replacing an existing key.
    The key itself is never written to logs or metadata.
    """
    key_path, metadata_path = Path(key_path), Path(metadata_path)
    logger = get_dataset_logger(Path(log_path))
    key_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if key_path.exists() and not confirm_overwrite:
        logger.error("Refusing to overwrite existing secret key at %s", key_path)
        raise KeyExistsError(f"Key already exists at {key_path}; explicit overwrite confirmation is required")

    key = secrets.token_bytes(KEY_BYTES)
    if confirm_overwrite:
        temp_path = key_path.with_name(key_path.name + ".tmp")
        try:
            temp_path.write_bytes(key)
            temp_path.replace(key_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    else:
        # Exclusive creation prevents a concurrent process from replacing a key.
        with key_path.open("xb") as key_file:
            key_file.write(key)

    metadata = {
        "algorithm": "secrets",
        "key_length": 128,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("Generated a 128-bit secret key at %s", key_path)
    logger.info("Key metadata saved to %s", metadata_path)
    return key
