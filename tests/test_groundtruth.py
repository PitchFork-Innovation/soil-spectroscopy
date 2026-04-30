"""Tests for the ground-truth point-to-tile join and aggregation.

The critical contract is that ``TileGrid.locate`` agrees with the tile
partition produced by ``tile_extraction`` — if it didn't, training labels
would land on the wrong cells.
"""

from __future__ import annotations

import numpy as np
import pytest

from soilspec.groundtruth import (
    GroundTruthDataset,
    Measurement,
    TileGrid,
    grid_for_request,
)
from soilspec.preprocessing.spatial import Raster, tile_extraction
from soilspec.types import AOI, BoundingBox


@pytest.fixture
def square_grid() -> TileGrid:
    bbox = BoundingBox(0.0, 0.0, 1.0, 1.0)
    return TileGrid.from_shape(bbox, raster_shape=(32, 32), tile_size=16)


# ---------------------------------------------------------------------------
# locate() agrees with tile_extraction
# ---------------------------------------------------------------------------


def test_locate_agrees_with_tile_extraction(square_grid):
    """Sample one point per tile (its centroid) and assert round-trip."""
    bbox = square_grid.aoi_bbox
    raster = Raster(
        data=np.zeros((32, 32), dtype=np.float32),
        crs="EPSG:4326",
        bounds=bbox,
        pixel_size=1.0,
    )
    tiles = tile_extraction(raster, bbox, tile_size=16)
    assert len(tiles) == 4  # 2x2 grid

    for tile in tiles:
        centroid_lon = (tile.bounds.min_lon + tile.bounds.max_lon) / 2
        centroid_lat = (tile.bounds.min_lat + tile.bounds.max_lat) / 2
        located = square_grid.locate(centroid_lon, centroid_lat)
        assert located == tile.tile_id, (
            f"centroid ({centroid_lon},{centroid_lat}) of {tile.tile_id} "
            f"resolved to {located}"
        )


def test_locate_returns_none_outside_aoi(square_grid):
    assert square_grid.locate(-0.5, 0.5) is None
    assert square_grid.locate(1.5, 0.5) is None
    assert square_grid.locate(0.5, -0.5) is None
    assert square_grid.locate(0.5, 1.5) is None


def test_locate_clamps_to_last_tile_on_max_edge(square_grid):
    # A point exactly on max_lon / min_lat must still be assigned to a tile
    # (the bottom-right corner), not dropped by integer-floor overshoot.
    assert square_grid.locate(1.0, 0.0) == "r001c001"
    assert square_grid.locate(0.0, 1.0) == "r000c000"


def test_grid_for_request_matches_pipeline_config():
    aoi = AOI(aoi_id="x", bbox=BoundingBox(10.0, 20.0, 14.0, 24.0))
    grid = grid_for_request(aoi, target_shape=(64, 64), tile_size=16)
    assert grid.n_rows == 4 and grid.n_cols == 4


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregation_means_and_stderr(square_grid):
    ds = GroundTruthDataset(square_grid, time_bucket_seconds=86400)
    # Three samples landing in the same tile + same day, same source.
    for val in (0.20, 0.24, 0.28):
        ds.add(Measurement(
            lon=0.25, lat=0.75, time=0,
            properties={"soil_moisture": val}, source="ismn",
        ))
    samples = list(ds.samples())
    assert len(samples) == 1
    s = samples[0]
    assert s.tile_id == "r000c000"
    assert s.n_observations == 3
    assert s.properties["soil_moisture"] == pytest.approx(0.24, abs=1e-9)
    # std = 0.04, n=3 -> stderr = 0.04 / sqrt(3)
    assert s.uncertainty["soil_moisture"] == pytest.approx(0.04 / np.sqrt(3), rel=1e-6)


def test_single_observation_uses_reported_uncertainty(square_grid):
    ds = GroundTruthDataset(square_grid)
    ds.add(Measurement(
        lon=0.25, lat=0.75, time=0,
        properties={"soil_moisture": 0.3},
        uncertainty={"soil_moisture": 0.05},
        source="ismn",
    ))
    [s] = list(ds.samples())
    assert s.uncertainty["soil_moisture"] == pytest.approx(0.05)
    assert s.n_observations == 1


def test_different_sources_kept_separate(square_grid):
    ds = GroundTruthDataset(square_grid)
    ds.add(Measurement(
        lon=0.25, lat=0.75, time=0,
        properties={"soil_moisture": 0.2}, source="ismn",
    ))
    ds.add(Measurement(
        lon=0.25, lat=0.75, time=0,
        properties={"soil_moisture": 0.3}, source="smap",
    ))
    samples = list(ds.samples())
    assert len(samples) == 2
    sources = {s.source for s in samples}
    assert sources == {"ismn", "smap"}


def test_time_bucketing_groups_within_bucket(square_grid):
    ds = GroundTruthDataset(square_grid, time_bucket_seconds=86400)
    # Two points in the same day-bucket get aggregated; one in the next day
    # stays separate.
    ds.add(Measurement(lon=0.25, lat=0.75, time=0,
                      properties={"soil_moisture": 0.2}, source="ismn"))
    ds.add(Measurement(lon=0.25, lat=0.75, time=3600,
                      properties={"soil_moisture": 0.3}, source="ismn"))
    ds.add(Measurement(lon=0.25, lat=0.75, time=86400,
                      properties={"soil_moisture": 0.4}, source="ismn"))
    samples = sorted(ds.samples(), key=lambda s: s.time)
    assert len(samples) == 2
    assert samples[0].time == 0
    assert samples[0].n_observations == 2
    assert samples[1].time == 86400
    assert samples[1].n_observations == 1


def test_out_of_aoi_points_are_dropped(square_grid):
    ds = GroundTruthDataset(square_grid)
    ds.add(Measurement(lon=2.0, lat=0.5, time=0,
                      properties={"soil_moisture": 0.3}, source="ismn"))
    ds.add(Measurement(lon=0.5, lat=0.5, time=0,
                      properties={"soil_moisture": 0.3}, source="ismn"))
    assert ds.dropped == 1
    assert len(list(ds.samples())) == 1


def test_nonfinite_values_are_skipped(square_grid):
    ds = GroundTruthDataset(square_grid)
    ds.add(Measurement(lon=0.25, lat=0.75, time=0,
                      properties={"soil_moisture": float("nan"), "soc": 12.0},
                      source="lucas"))
    [s] = list(ds.samples())
    assert "soil_moisture" not in s.properties
    assert s.properties["soc"] == 12.0
