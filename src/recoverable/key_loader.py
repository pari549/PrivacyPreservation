"""Load and length-check an existing binary secret key."""
from pathlib import Path

from src.recoverable.key_generator import DEFAULT_KEY_PATH, KEY_BYTES


class KeyLoadError(ValueError):
    """Raised when a key file cannot be read or has an invalid size."""


def load_key(key_path: Path = DEFAULT_KEY_PATH) -> bytes:
    """Load a key and require exactly 128 bits (16 bytes)."""
    key_path = Path(key_path)
    try:
        key = key_path.read_bytes()
    except OSError as error:
        raise KeyLoadError(f"Could not read secret key at {key_path}: {error}") from error
    if len(key) != KEY_BYTES:
        raise KeyLoadError(f"Secret key at {key_path} must be {KEY_BYTES} bytes; found {len(key)}")
    return key
