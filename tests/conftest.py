"""Pytest fixtures: synthetic AOI, time window, raw assets, storage."""

from __future__ import annotations

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--update-goldens", action="store_true", default=False,
                     help="Refresh golden-file fixtures instead of comparing.")

from soilspec.ingestion import (
    AdapterRegistry, Ingestion, MetadataParser, MCPAdapter,
)
from soilspec.preprocessing.spatial import Raster
from soilspec.preprocessing.vector import VectorRecords
from soilspec.storage import StorageTierManager
from soilspec.types import (
    AOI, BoundingBox, SENTINEL1, SENTINEL2, VECTOR, TimeWindow,
)


@pytest.fixture
def aoi() -> AOI:
    return AOI(aoi_id="test-aoi", bbox=BoundingBox(0.0, 0.0, 1.0, 1.0))


@pytest.fixture
def small_window() -> TimeWindow:
    # ~30 days from epoch — covers both S1 (6d revisit) and S2 (5d revisit)
    return TimeWindow(start=0, end=30 * 86400)


@pytest.fixture
def storage() -> StorageTierManager:
    return StorageTierManager()


@pytest.fixture
def adapters():
    return {
        SENTINEL1: AdapterRegistry.create(SENTINEL1),
        SENTINEL2: AdapterRegistry.create(SENTINEL2),
        VECTOR: AdapterRegistry.create(VECTOR),
    }


@pytest.fixture
def ingestion(storage, adapters) -> Ingestion:
    return Ingestion(storage=storage, adapters=adapters, parser=MetadataParser())


@pytest.fixture
def mcp_adapter(adapters) -> MCPAdapter:
    return MCPAdapter(adapters=adapters)


# ---------------------------------------------------------------------------
# Pure synthetic builders (used by unit tests that don't go through ingestion)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_s2_raster() -> Raster:
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, size=(6, 32, 32)).astype(np.float32)
    return Raster(
        data=data, crs="EPSG:4326", bounds=BoundingBox(0, 0, 1, 1), pixel_size=1.0,
    )


@pytest.fixture
def synthetic_s2_qa() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random(size=(32, 32)) > 0.85).astype(np.uint8)


@pytest.fixture
def synthetic_s1_raster() -> Raster:
    rng = np.random.default_rng(7)
    data = rng.uniform(low=-25.0, high=-5.0, size=(2, 32, 32)).astype(np.float32)
    return Raster(
        data=data, crs="EPSG:4326", bounds=BoundingBox(0, 0, 1, 1), pixel_size=1.0,
    )


@pytest.fixture
def synthetic_vector_records() -> VectorRecords:
    return VectorRecords(attributes={
        "slope": np.array([0.0, np.nan, 5.0, 10.0]),
        "elevation": np.array([100.0, 200.0, 300.0, 400.0]),
        "soc": np.array([1.0, 2.0, np.nan, 4.0]),
    })
