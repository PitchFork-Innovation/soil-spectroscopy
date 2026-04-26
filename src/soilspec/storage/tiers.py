"""Six-tier storage abstraction.

Callers ask for "the embedding for tile T at time t" via tier-specific key
helpers and never see the underlying backend. The default backend is
in-memory so unit tests run without any filesystem or network.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterator, Protocol


class StorageTier(str, Enum):
    RAW = "raw"
    PREPROCESSED = "preprocessed"
    EMBEDDING = "embedding"
    TEMPORAL = "temporal"
    MODEL = "model"
    MAP = "map"


# ---------------------------------------------------------------------------
# Key schemas — one helper per tier so callers cannot accidentally collide
# keys across tiers.
# ---------------------------------------------------------------------------


def raw_key(provider: str, observation_id: str) -> str:
    return f"{provider}/{observation_id}"


def preprocessed_key(aoi_id: str, tile_id: str, time: int, modality: str) -> str:
    return f"{aoi_id}/{tile_id}/{time}/{modality}"


def embedding_key(tile_id: str, time: int, modality_or_fused: str) -> str:
    return f"{tile_id}/{time}/{modality_or_fused}"


def temporal_key(spatial_cell_id: str) -> str:
    return f"cell/{spatial_cell_id}"


def model_key(model_family: str, version: str) -> str:
    return f"{model_family}/{version}"


def map_key(aoi_id: str, output_type: str, generation_time: int) -> str:
    return f"{aoi_id}/{output_type}/{generation_time}"


class StorageBackend(Protocol):
    """The minimal contract every backend implements."""

    def get(self, tier: StorageTier, key: str) -> Any: ...
    def put(self, tier: StorageTier, key: str, value: Any) -> None: ...
    def exists(self, tier: StorageTier, key: str) -> bool: ...
    def list(self, tier: StorageTier, prefix: str = "") -> Iterator[str]: ...
    def delete(self, tier: StorageTier, key: str) -> None: ...


class InMemoryBackend:
    """Default backend used in tests and reference deployments."""

    def __init__(self) -> None:
        self._store: dict[StorageTier, dict[str, Any]] = {t: {} for t in StorageTier}

    def get(self, tier: StorageTier, key: str) -> Any:
        try:
            return self._store[tier][key]
        except KeyError:
            raise KeyError(f"{tier.value}:{key} not found") from None

    def put(self, tier: StorageTier, key: str, value: Any) -> None:
        self._store[tier][key] = value

    def exists(self, tier: StorageTier, key: str) -> bool:
        return key in self._store[tier]

    def list(self, tier: StorageTier, prefix: str = "") -> Iterator[str]:
        for k in sorted(self._store[tier]):
            if k.startswith(prefix):
                yield k

    def delete(self, tier: StorageTier, key: str) -> None:
        self._store[tier].pop(key, None)


class StorageTierManager:
    """High-level facade over a backend.

    Pipeline stages depend on this rather than the backend protocol so the
    backend can be swapped (in-memory, filesystem, object store, geospatial DB)
    without touching call sites.
    """

    def __init__(self, backend: StorageBackend | None = None) -> None:
        self._backend: StorageBackend = backend or InMemoryBackend()

    @property
    def backend(self) -> StorageBackend:
        return self._backend

    def get(self, tier: StorageTier, key: str) -> Any:
        return self._backend.get(tier, key)

    def put(self, tier: StorageTier, key: str, value: Any) -> None:
        self._backend.put(tier, key, value)

    def exists(self, tier: StorageTier, key: str) -> bool:
        return self._backend.exists(tier, key)

    def list(self, tier: StorageTier, prefix: str = "") -> Iterator[str]:
        return self._backend.list(tier, prefix)

    def delete(self, tier: StorageTier, key: str) -> None:
        self._backend.delete(tier, key)
