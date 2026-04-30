"""Tests for the ISMN ground-truth adapter.

Covers parsing, AOI/time filtering, QC + depth gates, and end-to-end
integration with :class:`GroundTruthDataset` so we know real-shaped fixture
data lands on the right tiles with sensible aggregates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from soilspec.groundtruth import (
    GroundTruthDataset,
    ISMNAdapter,
    TileGrid,
)
from soilspec.types import AOI, BoundingBox, TimeWindow


ISMN_HEADER = (
    "network,station,lon,lat,timestamp,soil_moisture,"
    "depth_from,depth_to,qc_flag,soil_moisture_uncertainty\n"
)


@pytest.fixture
def aoi() -> AOI:
    return AOI(aoi_id="test", bbox=BoundingBox(0.0, 0.0, 1.0, 1.0))


@pytest.fixture
def window() -> TimeWindow:
    return TimeWindow(start=0, end=30 * 86400)


@pytest.fixture
def grid(aoi) -> TileGrid:
    return TileGrid.from_shape(aoi.bbox, raster_shape=(32, 32), tile_size=16)


def _write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text(ISMN_HEADER + "\n".join(rows) + "\n")
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parses_basic_rows(tmp_path, aoi, window):
    p = _write_csv(tmp_path / "ismn.csv", [
        "SCAN,STN1,0.25,0.75,3600,0.234,0,5,G,0.01",
        "SCAN,STN2,0.75,0.25,7200,0.180,0,5,G,",
    ])
    out = list(ISMNAdapter(csv_path=p).fetch(aoi, window))
    assert len(out) == 2
    assert out[0].properties == {"soil_moisture": pytest.approx(0.234)}
    assert out[0].uncertainty == {"soil_moisture": pytest.approx(0.01)}
    assert out[1].uncertainty == {}  # blank uncertainty cell -> dropped
    assert all(m.source == "ismn" for m in out)


def test_drops_bad_qc(tmp_path, aoi, window):
    p = _write_csv(tmp_path / "ismn.csv", [
        "SCAN,A,0.5,0.5,1000,0.3,0,5,G,",
        "SCAN,B,0.5,0.5,2000,0.3,0,5,D,",   # dry — fail QC
        "SCAN,C,0.5,0.5,3000,0.3,0,5,C,",   # frozen — fail QC
    ])
    out = list(ISMNAdapter(csv_path=p).fetch(aoi, window))
    assert [int(m.time) for m in out] == [1000]


def test_drops_deep_sensors(tmp_path, aoi, window):
    p = _write_csv(tmp_path / "ismn.csv", [
        "SCAN,A,0.5,0.5,1000,0.3,0,5,G,",
        "SCAN,B,0.5,0.5,2000,0.3,20,30,G,",   # too deep for default 10cm
    ])
    out = list(ISMNAdapter(csv_path=p).fetch(aoi, window))
    assert len(out) == 1
    assert out[0].time == 1000


def test_custom_max_depth_lets_deeper_sensors_through(tmp_path, aoi, window):
    p = _write_csv(tmp_path / "ismn.csv", [
        "SCAN,B,0.5,0.5,2000,0.3,20,30,G,",
    ])
    out = list(ISMNAdapter(csv_path=p, max_depth_cm=50.0).fetch(aoi, window))
    assert len(out) == 1


# ---------------------------------------------------------------------------
# AOI / time filtering
# ---------------------------------------------------------------------------


def test_aoi_bbox_filter(tmp_path, aoi, window):
    p = _write_csv(tmp_path / "ismn.csv", [
        "SCAN,IN,0.5,0.5,1000,0.3,0,5,G,",
        "SCAN,WEST,-1.0,0.5,2000,0.3,0,5,G,",
        "SCAN,NORTH,0.5,2.0,3000,0.3,0,5,G,",
    ])
    out = list(ISMNAdapter(csv_path=p).fetch(aoi, window))
    assert [m.time for m in out] == [1000]


def test_time_window_filter(tmp_path, aoi):
    p = _write_csv(tmp_path / "ismn.csv", [
        "SCAN,A,0.5,0.5,500,0.3,0,5,G,",
        "SCAN,B,0.5,0.5,5000,0.3,0,5,G,",
        "SCAN,C,0.5,0.5,500000,0.3,0,5,G,",
    ])
    w = TimeWindow(start=1000, end=10_000)
    out = list(ISMNAdapter(csv_path=p).fetch(aoi, w))
    assert [m.time for m in out] == [5000]


def test_skips_malformed_rows(tmp_path, aoi, window):
    p = _write_csv(tmp_path / "ismn.csv", [
        "SCAN,A,0.5,0.5,1000,0.3,0,5,G,",
        "SCAN,B,not-a-number,0.5,2000,0.3,0,5,G,",
        "SCAN,C,0.5,0.5,3000,nan,0,5,G,",
    ])
    out = list(ISMNAdapter(csv_path=p).fetch(aoi, window))
    assert [m.time for m in out] == [1000]


def test_missing_required_column_raises(tmp_path, aoi, window):
    p = tmp_path / "bad.csv"
    p.write_text("network,station,lon,lat\nSCAN,A,0.5,0.5\n")
    with pytest.raises(ValueError, match="missing required columns"):
        list(ISMNAdapter(csv_path=p).fetch(aoi, window))


def test_missing_file_raises(tmp_path, aoi, window):
    with pytest.raises(FileNotFoundError):
        list(ISMNAdapter(csv_path=tmp_path / "nope.csv").fetch(aoi, window))


# ---------------------------------------------------------------------------
# End-to-end: ISMN adapter feeding GroundTruthDataset
# ---------------------------------------------------------------------------


def test_ismn_into_dataset_aggregates_per_tile_per_day(tmp_path, aoi, window, grid):
    # Three measurements in the upper-left tile within one day, plus one in
    # the lower-right tile on a different day. After aggregation we should
    # get two samples.
    p = _write_csv(tmp_path / "ismn.csv", [
        "SCAN,A,0.10,0.90,3600,0.20,0,5,G,",
        "SCAN,B,0.20,0.80,7200,0.24,0,5,G,",
        "SCAN,C,0.30,0.60,10800,0.28,0,5,G,",
        "SCAN,D,0.80,0.10,90000,0.15,0,5,G,",   # next day, opposite corner
    ])
    ds = GroundTruthDataset(grid, time_bucket_seconds=86400)
    ds.extend(ISMNAdapter(csv_path=p).fetch(aoi, window))
    samples = sorted(ds.samples(), key=lambda s: (s.tile_id, s.time))
    assert len(samples) == 2
    upper_left = next(s for s in samples if s.tile_id == "r000c000")
    assert upper_left.n_observations == 3
    assert upper_left.properties["soil_moisture"] == pytest.approx(0.24)
    lower_right = next(s for s in samples if s.tile_id == "r001c001")
    assert lower_right.n_observations == 1
    assert lower_right.time == 86400  # bucketed to start of day 1
