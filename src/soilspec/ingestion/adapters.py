"""Source adapters.

Each adapter accepts an AOI + time window and returns raw observation records
with stable IDs. Synthetic adapters are sufficient for tests; real Sentinel
adapters would slot in by registering against the same registry under the
same modality name.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Protocol

import numpy as np

from ..registry import Registry
from ..types import (
    AOI,
    AssetMetadata,
    BoundingBox,
    SENTINEL1,
    SENTINEL2,
    VECTOR,
    INSITU,
    TimeWindow,
)


class UnreachableSourceError(RuntimeError):
    """Raised when a source adapter cannot reach its provider after retries."""


@dataclass(frozen=True)
class RawAsset:
    """Adapter output: payload + a (mostly already filled) metadata sidecar.

    The metadata is parsed by the metadata parser into a `AssetMetadata`; the
    parser may add provider-specific fields the adapter could not populate.
    """

    observation_id: str
    provider: str
    modality: str
    payload: Any  # bytes-equivalent: numpy array, dict, etc.
    metadata: AssetMetadata


class SourceAdapter(Protocol):
    provider: str
    modality: str

    def fetch(self, aoi: AOI, window: TimeWindow) -> Iterable[RawAsset]: ...


def _stable_id(*parts: Any) -> str:
    h = hashlib.sha256("/".join(str(p) for p in parts).encode()).hexdigest()
    return h[:16]


def _walk_window(window: TimeWindow, step: int) -> Iterator[int]:
    t = window.start
    while t <= window.end:
        yield t
        t += step


class SyntheticSentinel1Adapter:
    """Synthetic SAR backscatter generator. Deterministic for the same AOI/window."""

    provider = "synthetic-s1"
    modality = SENTINEL1

    def __init__(self, revisit_seconds: int = 6 * 86400, tile_size: int = 32) -> None:
        self.revisit_seconds = revisit_seconds
        self.tile_size = tile_size

    def fetch(self, aoi: AOI, window: TimeWindow) -> Iterable[RawAsset]:
        for t in _walk_window(window, self.revisit_seconds):
            seed = int(_stable_id(aoi.aoi_id, t, self.modality), 16) % (2**32)
            rng = np.random.default_rng(seed)
            payload = rng.uniform(low=-25.0, high=-5.0, size=(2, self.tile_size, self.tile_size))
            obs_id = _stable_id(aoi.aoi_id, t, self.provider)
            meta = AssetMetadata(
                observation_id=obs_id,
                request_id=_stable_id("req", aoi.aoi_id, window.start, window.end),
                provider=self.provider,
                modality=self.modality,
                timestamp=t,
                bbox=aoi.bbox,
                bands=("VV", "VH"),
                missing_entries=(),
                extra={"sensor_geometry": "IW", "calibration": "raw_dn"},
            )
            yield RawAsset(obs_id, self.provider, self.modality, payload, meta)


class SyntheticSentinel2Adapter:
    """Synthetic multispectral generator with a deterministic cloud mask."""

    provider = "synthetic-s2"
    modality = SENTINEL2

    BANDS = ("B02", "B03", "B04", "B08", "B11", "B12", "QA")

    def __init__(self, revisit_seconds: int = 5 * 86400, tile_size: int = 32) -> None:
        self.revisit_seconds = revisit_seconds
        self.tile_size = tile_size

    def fetch(self, aoi: AOI, window: TimeWindow) -> Iterable[RawAsset]:
        for t in _walk_window(window, self.revisit_seconds):
            seed = int(_stable_id(aoi.aoi_id, t, self.modality), 16) % (2**32)
            rng = np.random.default_rng(seed)
            n_bands = len(self.BANDS) - 1  # last is QA
            data = rng.uniform(low=0.0, high=1.0, size=(n_bands, self.tile_size, self.tile_size))
            qa = (rng.random(size=(self.tile_size, self.tile_size)) > 0.85).astype(np.uint8)
            payload = {"reflectance": data, "qa": qa}
            obs_id = _stable_id(aoi.aoi_id, t, self.provider)
            meta = AssetMetadata(
                observation_id=obs_id,
                request_id=_stable_id("req", aoi.aoi_id, window.start, window.end),
                provider=self.provider,
                modality=self.modality,
                timestamp=t,
                bbox=aoi.bbox,
                bands=self.BANDS,
                missing_entries=(),
                extra={"reflectance_geometry": "BOA"},
            )
            yield RawAsset(obs_id, self.provider, self.modality, payload, meta)


class SyntheticVectorAdapter:
    """Synthetic vector environmental layers (slope, land cover, soil grids)."""

    provider = "synthetic-vector"
    modality = VECTOR
    ATTRIBUTES = ("slope", "elevation", "land_cover", "soil_organic_carbon", "clay_pct")

    def __init__(self, tile_size: int = 32) -> None:
        self.tile_size = tile_size

    def fetch(self, aoi: AOI, window: TimeWindow) -> Iterable[RawAsset]:
        seed = int(_stable_id(aoi.aoi_id, "vector"), 16) % (2**32)
        rng = np.random.default_rng(seed)
        records = {
            "slope": rng.uniform(0, 30, size=self.tile_size).tolist(),
            "elevation": rng.uniform(0, 1500, size=self.tile_size).tolist(),
            "land_cover": rng.integers(1, 7, size=self.tile_size).tolist(),
            "soil_organic_carbon": rng.uniform(0, 5, size=self.tile_size).tolist(),
            "clay_pct": rng.uniform(5, 60, size=self.tile_size).tolist(),
        }
        # inject some NaNs to exercise the imputation step downstream
        records["soil_organic_carbon"][0] = float("nan")
        records["slope"][1] = float("nan")
        obs_id = _stable_id(aoi.aoi_id, "vector", self.provider)
        meta = AssetMetadata(
            observation_id=obs_id,
            request_id=_stable_id("req", aoi.aoi_id, window.start, window.end),
            provider=self.provider,
            modality=self.modality,
            timestamp=window.start,
            bbox=aoi.bbox,
            bands=tuple(self.ATTRIBUTES),
            missing_entries=("soil_organic_carbon[0]", "slope[1]"),
            extra={},
        )
        yield RawAsset(obs_id, self.provider, self.modality, records, meta)


class SyntheticInsituAdapter:
    """Optional in-situ measurements. Off by default."""

    provider = "synthetic-insitu"
    modality = INSITU

    def fetch(self, aoi: AOI, window: TimeWindow) -> Iterable[RawAsset]:
        return ()


# Factory registry. Real Sentinel adapters can register here.
AdapterRegistry: Registry[SourceAdapter] = Registry("source-adapters")
AdapterRegistry.register(SENTINEL1, lambda **kw: SyntheticSentinel1Adapter(**kw))
AdapterRegistry.register(SENTINEL2, lambda **kw: SyntheticSentinel2Adapter(**kw))
AdapterRegistry.register(VECTOR, lambda **kw: SyntheticVectorAdapter(**kw))
AdapterRegistry.register(INSITU, lambda **kw: SyntheticInsituAdapter(**kw))


__all__ = [
    "RawAsset",
    "SourceAdapter",
    "SyntheticSentinel1Adapter",
    "SyntheticSentinel2Adapter",
    "SyntheticVectorAdapter",
    "SyntheticInsituAdapter",
    "AdapterRegistry",
    "UnreachableSourceError",
]


def _bbox_overlaps(a: BoundingBox, b: BoundingBox) -> bool:  # pragma: no cover - aux
    return not (
        a.max_lon < b.min_lon
        or a.min_lon > b.max_lon
        or a.max_lat < b.min_lat
        or a.min_lat > b.max_lat
    )
