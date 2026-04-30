"""Tests for the SoilGrids covariate adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from soilspec.groundtruth import (
    GroundTruthDataset,
    SoilGridsAdapter,
    TileGrid,
)
from soilspec.types import AOI, BoundingBox, TimeWindow


@pytest.fixture
def aoi() -> AOI:
    return AOI(aoi_id="test", bbox=BoundingBox(0.0, 0.0, 1.0, 1.0))


@pytest.fixture
def window() -> TimeWindow:
    return TimeWindow(start=1_700_000_000, end=1_700_000_000 + 30 * 86400)


@pytest.fixture
def grid(aoi) -> TileGrid:
    return TileGrid.from_shape(aoi.bbox, raster_shape=(32, 32), tile_size=16)


def _write_csv(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def test_parses_multiple_properties(tmp_path, aoi, window):
    p = _write_csv(
        tmp_path / "sg.csv",
        "lon,lat,clay_pct,sand_pct,soc,bulk_density,ph,depth_to",
        [
            "0.25,0.75,28.5,42.1,15.0,1.32,6.4,5",
            "0.75,0.25,35.0,30.0,18.5,1.25,5.9,5",
        ],
    )
    out = list(SoilGridsAdapter(csv_path=p).fetch(aoi, window))
    assert len(out) == 2
    assert out[0].properties == {
        "clay_pct": pytest.approx(28.5),
        "sand_pct": pytest.approx(42.1),
        "soc": pytest.approx(15.0),
        "bulk_density": pytest.approx(1.32),
        "ph": pytest.approx(6.4),
    }
    assert all(m.source == "soilgrids" for m in out)


def test_static_timestamp_is_window_start(tmp_path, aoi, window):
    p = _write_csv(
        tmp_path / "sg.csv",
        "lon,lat,clay_pct",
        ["0.5,0.5,30.0"],
    )
    [m] = list(SoilGridsAdapter(csv_path=p).fetch(aoi, window))
    assert m.time == window.start


def test_aoi_filter_drops_outside_points(tmp_path, aoi, window):
    p = _write_csv(
        tmp_path / "sg.csv",
        "lon,lat,clay_pct",
        ["0.5,0.5,30.0", "5.0,5.0,40.0"],
    )
    out = list(SoilGridsAdapter(csv_path=p).fetch(aoi, window))
    assert len(out) == 1
    assert out[0].lon == 0.5


def test_depth_filter_drops_subsurface(tmp_path, aoi, window):
    p = _write_csv(
        tmp_path / "sg.csv",
        "lon,lat,clay_pct,depth_to",
        ["0.5,0.5,30.0,5", "0.5,0.5,32.0,30"],
    )
    out = list(SoilGridsAdapter(csv_path=p).fetch(aoi, window))
    assert len(out) == 1
    assert out[0].properties["clay_pct"] == 30.0


def test_skips_properties_not_in_csv(tmp_path, aoi, window):
    """Adapter is configured for 5 properties but CSV only has 2."""
    p = _write_csv(
        tmp_path / "sg.csv",
        "lon,lat,clay_pct,soc",
        ["0.5,0.5,30.0,12.5"],
    )
    [m] = list(SoilGridsAdapter(csv_path=p).fetch(aoi, window))
    assert set(m.properties.keys()) == {"clay_pct", "soc"}


def test_raises_when_no_configured_properties_present(tmp_path, aoi, window):
    p = _write_csv(
        tmp_path / "sg.csv",
        "lon,lat,unknown_field",
        ["0.5,0.5,99"],
    )
    with pytest.raises(ValueError, match="none of the configured"):
        list(SoilGridsAdapter(csv_path=p).fetch(aoi, window))


def test_into_dataset_lands_on_correct_tiles(tmp_path, aoi, window, grid):
    p = _write_csv(
        tmp_path / "sg.csv",
        "lon,lat,clay_pct,soc",
        [
            "0.10,0.90,25.0,12.0",
            "0.85,0.15,40.0,18.0",
        ],
    )
    ds = GroundTruthDataset(grid)
    ds.extend(SoilGridsAdapter(csv_path=p).fetch(aoi, window))
    samples = {s.tile_id: s for s in ds.samples()}
    assert "r000c000" in samples
    assert "r001c001" in samples
    assert samples["r000c000"].source == "soilgrids"
