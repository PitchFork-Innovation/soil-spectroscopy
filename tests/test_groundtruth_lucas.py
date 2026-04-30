"""Tests for the LUCAS soil-survey adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from soilspec.groundtruth import LUCASAdapter
from soilspec.types import AOI, BoundingBox, TimeWindow


@pytest.fixture
def aoi() -> AOI:
    return AOI(aoi_id="eu", bbox=BoundingBox(-10.0, 35.0, 30.0, 60.0))


@pytest.fixture
def window_2018() -> TimeWindow:
    start = int(datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp())
    end = int(datetime(2019, 12, 31, tzinfo=timezone.utc).timestamp())
    return TimeWindow(start=start, end=end)


def _write_csv(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def test_parses_multi_property_row(tmp_path, aoi, window_2018):
    p = _write_csv(
        tmp_path / "lucas.csv",
        "point_id,lon,lat,year,soc,nitrogen,phosphorus,potassium,ph,clay_pct,sand_pct",
        ["EU0001,10.5,48.2,2018,18.5,1.6,42,180,6.2,22.0,45.0"],
    )
    [m] = list(LUCASAdapter(csv_path=p).fetch(aoi, window_2018))
    assert m.properties["soc"] == pytest.approx(18.5)
    assert m.properties["nitrogen"] == pytest.approx(1.6)
    assert m.properties["ph"] == pytest.approx(6.2)
    assert m.source == "lucas"
    # Year 2018 → Jan 1 2018 UTC
    assert m.time == int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp())


def test_aoi_filter(tmp_path, aoi, window_2018):
    p = _write_csv(
        tmp_path / "lucas.csv",
        "point_id,lon,lat,year,soc",
        [
            "IN,10.0,48.0,2018,15.0",
            "OUTSIDE,-50.0,48.0,2018,15.0",
        ],
    )
    out = list(LUCASAdapter(csv_path=p).fetch(aoi, window_2018))
    assert len(out) == 1
    assert out[0].lon == 10.0


def test_time_window_filter(tmp_path, aoi):
    p = _write_csv(
        tmp_path / "lucas.csv",
        "point_id,lon,lat,year,soc",
        [
            "OLD,10.0,48.0,2009,12.0",
            "RECENT,10.0,48.0,2018,18.0",
        ],
    )
    only_2018 = TimeWindow(
        start=int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp()),
        end=int(datetime(2018, 12, 31, tzinfo=timezone.utc).timestamp()),
    )
    out = list(LUCASAdapter(csv_path=p).fetch(aoi, only_2018))
    assert len(out) == 1
    assert out[0].properties["soc"] == 18.0


def test_partial_properties_emit_subset(tmp_path, aoi, window_2018):
    """A row that only fills two of the configured columns still produces a
    Measurement, with just those two properties."""
    p = _write_csv(
        tmp_path / "lucas.csv",
        "point_id,lon,lat,year,soc,nitrogen,phosphorus,potassium,ph,clay_pct,sand_pct",
        ["EU0002,10.5,48.2,2018,18.5,,,,6.2,,"],
    )
    [m] = list(LUCASAdapter(csv_path=p).fetch(aoi, window_2018))
    assert set(m.properties.keys()) == {"soc", "ph"}


def test_row_with_no_useful_properties_is_dropped(tmp_path, aoi, window_2018):
    p = _write_csv(
        tmp_path / "lucas.csv",
        "point_id,lon,lat,year,soc,ph",
        ["EU0003,10.5,48.2,2018,,"],
    )
    out = list(LUCASAdapter(csv_path=p).fetch(aoi, window_2018))
    assert out == []


def test_missing_year_column_raises(tmp_path, aoi, window_2018):
    p = _write_csv(
        tmp_path / "lucas.csv",
        "point_id,lon,lat,soc",
        ["EU0004,10.5,48.2,18.5"],
    )
    with pytest.raises(ValueError, match="missing required column"):
        list(LUCASAdapter(csv_path=p).fetch(aoi, window_2018))
