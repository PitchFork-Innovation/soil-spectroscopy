"""Stable seed derivation.

Python's `hash()` is randomized per process via PYTHONHASHSEED, which would
violate the "cross-process determinism" guarantee. We derive 32-bit seeds
from a SHA-256 of the inputs instead.
"""

from __future__ import annotations

import hashlib
from typing import Any


def stable_seed(*parts: Any) -> int:
    raw = "/".join(repr(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")
