"""Per-block deterministic noise streams derived from the 128-bit key."""
from __future__ import annotations
import hashlib
import hmac
import numpy as np

NOISE_CONTEXT = b"privacy-preservation:adaptive-reversible-noise:v1"

def derive_noise_seed(key: bytes) -> str:
    """Return a stable key-derived seed identifier without exposing the key."""
    if len(key) != 16: raise ValueError("Noise key must be exactly 128 bits (16 bytes)")
    return hmac.new(key, NOISE_CONTEXT, hashlib.sha256).hexdigest()

def generate_block_noise(key: bytes, block_index: int, alpha: float,
                         shape: tuple[int, int, int] = (4, 4, 3)) -> np.ndarray:
    """Generate deterministic signed integer noise for one image block."""
    if len(key) != 16: raise ValueError("Noise key must be exactly 128 bits (16 bytes)")
    if not 0.10 <= float(alpha) <= 0.70: raise ValueError("Alpha must be in [0.10, 0.70]")
    if block_index < 0: raise ValueError("Block index cannot be negative")
    message = NOISE_CONTEXT + b":" + int(block_index).to_bytes(4, "big")
    seed_bytes = hmac.new(key, message, hashlib.sha256).digest()
    seed = int.from_bytes(seed_bytes[:16], "big")
    generator = np.random.default_rng(seed)
    amplitude = max(1, int(np.ceil(float(alpha) * 255.0)))
    return generator.integers(-amplitude, amplitude + 1, size=shape, dtype=np.int16)
