"""Validate secret key length, binary integrity, and reload consistency."""
from pathlib import Path

from src.recoverable.key_generator import DEFAULT_KEY_PATH, KEY_BYTES
from src.recoverable.key_loader import KeyLoadError, load_key
from src.preprocessing.data_loader import DEFAULT_LOG_PATH, get_dataset_logger


class KeyValidationError(ValueError):
    """Raised when a stored key fails an integrity check."""


def validate_key(key_path: Path = DEFAULT_KEY_PATH, log_path: Path = DEFAULT_LOG_PATH) -> bool:
    """Verify 128-bit length, bytes integrity, and successful identical reload."""
    key_path = Path(key_path)
    logger = get_dataset_logger(Path(log_path))
    try:
        original = key_path.read_bytes()
        if not isinstance(original, bytes) or len(original) != KEY_BYTES:
            raise KeyValidationError(f"Key must contain exactly {KEY_BYTES} binary bytes")
        reloaded = load_key(key_path)
        if reloaded != original:
            raise KeyValidationError("Reloaded key does not match the stored key bytes")
    except (OSError, KeyLoadError, KeyValidationError) as error:
        logger.exception("Secret key validation failed for %s", key_path)
        if isinstance(error, KeyValidationError):
            raise
        raise KeyValidationError(f"Secret key validation failed: {error}") from error
    logger.info("Secret key validated successfully at %s", key_path)
    return True
