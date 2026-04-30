"""Tests for the real-mode ground-truth adapters.

Uses mocked clients (no network, no real ISMN archive) so the tests run
deterministically. The adapters' integration with their upstream libraries
(ismn, urllib) is exercised through narrow seams so we don't accidentally
hit live services from CI.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from soilspec.groundtruth import (
    GroundTruthSourceError,
    ISMNArchiveAdapter,
    LUCASAdapter,
    SoilGridsRESTAdapter,
    normalize_lucas_esdac_csv,
)
from soilspec.types import AOI, BoundingBox, TimeWindow


# ---------------------------------------------------------------------------
# ISMN archive adapter
# ---------------------------------------------------------------------------


def _ismn_meta(lon: float, lat: float) -> pd.Series:
    return pd.Series({"longitude": lon, "latitude": lat})


def _ismn_ts(times: list[datetime], values: list[float], qc: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"soil_moisture": values, "soil_moisture_flag": qc},
        index=pd.DatetimeIndex(times, tz="UTC"),
    )


def test_ismn_archive_adapter_yields_filtered_measurements(tmp_path):
    archive = tmp_path / "fake_archive"
    archive.mkdir()  # path must exist; ismn package is mocked entirely

    aoi = AOI(aoi_id="t", bbox=BoundingBox(0.0, 0.0, 1.0, 1.0))
    window = TimeWindow(
        start=int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()),
        end=int(datetime(2024, 1, 31, tzinfo=timezone.utc).timestamp()),
    )

    interface = MagicMock()
    interface.get_dataset_ids.return_value = [101, 102]
    interface.read_ts.side_effect = [
        # station 101: in AOI, two readings — second has bad QC
        (
            _ismn_ts(
                [datetime(2024, 1, 5, 12, tzinfo=timezone.utc),
                 datetime(2024, 1, 6, 12, tzinfo=timezone.utc)],
                [0.234, 0.250],
                ["G", "C"],
            ),
            _ismn_meta(0.5, 0.5),
        ),
        # station 102: outside AOI, should be skipped entirely
        (
            _ismn_ts([datetime(2024, 1, 7, tzinfo=timezone.utc)], [0.30], ["G"]),
            _ismn_meta(5.0, 5.0),
        ),
    ]

    with patch("ismn.interface.ISMN_Interface", return_value=interface):
        adapter = ISMNArchiveAdapter(archive_path=archive)
        out = list(adapter.fetch(aoi, window))

    assert len(out) == 1
    assert out[0].properties["soil_moisture"] == pytest.approx(0.234)
    assert out[0].source == "ismn"
    interface.get_dataset_ids.assert_called_once()
    kwargs = interface.get_dataset_ids.call_args.kwargs
    assert kwargs["variable"] == "soil_moisture"
    assert kwargs["min_depth"] == 0.0
    assert kwargs["max_depth"] == 0.10  # default surface depth


def test_ismn_archive_adapter_filters_outside_time_window(tmp_path):
    archive = tmp_path / "arch"
    archive.mkdir()
    aoi = AOI(aoi_id="t", bbox=BoundingBox(0.0, 0.0, 1.0, 1.0))
    window = TimeWindow(
        start=int(datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp()),
        end=int(datetime(2024, 6, 30, tzinfo=timezone.utc).timestamp()),
    )

    interface = MagicMock()
    interface.get_dataset_ids.return_value = [1]
    interface.read_ts.return_value = (
        _ismn_ts(
            [datetime(2023, 1, 1, tzinfo=timezone.utc),
             datetime(2024, 6, 15, tzinfo=timezone.utc),
             datetime(2025, 1, 1, tzinfo=timezone.utc)],
            [0.1, 0.2, 0.3],
            ["G", "G", "G"],
        ),
        _ismn_meta(0.5, 0.5),
    )

    with patch("ismn.interface.ISMN_Interface", return_value=interface):
        out = list(ISMNArchiveAdapter(archive_path=archive).fetch(aoi, window))
    assert [m.properties["soil_moisture"] for m in out] == [pytest.approx(0.2)]


def test_ismn_archive_missing_path_raises(tmp_path):
    aoi = AOI(aoi_id="t", bbox=BoundingBox(0.0, 0.0, 1.0, 1.0))
    window = TimeWindow(start=0, end=1)
    with pytest.raises(FileNotFoundError):
        list(ISMNArchiveAdapter(archive_path=tmp_path / "nope").fetch(aoi, window))


# ---------------------------------------------------------------------------
# SoilGrids REST adapter
# ---------------------------------------------------------------------------


def _soilgrids_response(values: dict[str, int | None]) -> dict:
    """Build a JSON payload matching ISRIC's actual schema."""
    layers = []
    for layer_name, mean in values.items():
        layers.append({
            "name": layer_name,
            "depths": [
                {"label": "0-5cm", "values": {"mean": mean}},
                {"label": "5-15cm", "values": {"mean": (mean or 0) + 1}},
            ],
        })
    return {"properties": {"layers": layers}}


def test_soilgrids_parse_handles_unit_conversions():
    # Construct a minimal payload with values in raw SoilGrids units;
    # adapter should divide each by the documented divisor.
    adapter = SoilGridsRESTAdapter()
    payload = _soilgrids_response({
        "clay": 285,    # dg/kg -> 28.5 %
        "sand": 421,    # dg/kg -> 42.1 %
        "soc": 150,     # dg/kg -> 15.0 g/kg
        "bdod": 132,    # cg/cm3 -> 1.32 g/cm3
        "phh2o": 64,    # 64 / 10 = 6.4
    })
    out = adapter._parse_properties(payload)
    assert out == {
        "clay_pct": pytest.approx(28.5),
        "sand_pct": pytest.approx(42.1),
        "soc": pytest.approx(15.0),
        "bulk_density": pytest.approx(1.32),
        "ph": pytest.approx(6.4),
    }


def test_soilgrids_parse_skips_missing_means():
    adapter = SoilGridsRESTAdapter()
    payload = _soilgrids_response({"clay": 285, "sand": None})
    out = adapter._parse_properties(payload)
    assert "clay_pct" in out
    assert "sand_pct" not in out


def test_soilgrids_sample_points_centroids():
    aoi = AOI(aoi_id="t", bbox=BoundingBox(0.0, 0.0, 1.0, 1.0))
    adapter = SoilGridsRESTAdapter(sample_rows=2, sample_cols=2)
    pts = list(adapter._sample_points(aoi))
    # 2x2 grid centroids: each cell is 0.5 wide -> centroids at .25 and .75
    assert pts == [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]


def test_soilgrids_fetch_uses_mocked_query():
    aoi = AOI(aoi_id="t", bbox=BoundingBox(0.0, 0.0, 1.0, 1.0))
    window = TimeWindow(start=1_700_000_000, end=1_700_000_000 + 86400)
    adapter = SoilGridsRESTAdapter(
        properties=("clay_pct", "soc"), sample_rows=1, sample_cols=1,
    )
    response = _soilgrids_response({"clay": 250, "soc": 120})
    fake_resp = MagicMock()
    fake_resp.read.return_value = json.dumps(response).encode("utf-8")
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda self, *a: False

    with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
        out = list(adapter.fetch(aoi, window))

    assert len(out) == 1
    m = out[0]
    assert m.lon == pytest.approx(0.5)  # single-cell centroid
    assert m.lat == pytest.approx(0.5)
    assert m.time == window.start  # static layer
    assert m.source == "soilgrids"
    assert m.properties == {
        "clay_pct": pytest.approx(25.0),
        "soc": pytest.approx(12.0),
    }
    # The constructed URL must include both requested layers and 0-5cm depth.
    called_url = mock_open.call_args.args[0].full_url
    assert "property=clay" in called_url
    assert "property=soc" in called_url
    assert "depth=0-5cm" in called_url


def test_soilgrids_network_error_wraps_in_source_error():
    from urllib.error import URLError

    aoi = AOI(aoi_id="t", bbox=BoundingBox(0.0, 0.0, 1.0, 1.0))
    window = TimeWindow(start=0, end=1)
    adapter = SoilGridsRESTAdapter(sample_rows=1, sample_cols=1)
    with patch("urllib.request.urlopen", side_effect=URLError("boom")):
        with pytest.raises(GroundTruthSourceError, match="SoilGrids query failed"):
            list(adapter.fetch(aoi, window))


# ---------------------------------------------------------------------------
# LUCAS ESDAC normalization
# ---------------------------------------------------------------------------


def test_normalize_lucas_esdac_csv_handles_official_columns(tmp_path):
    src = tmp_path / "esdac.csv"
    # Mimic the 2018 ESDAC release column names + decimal-comma values
    src.write_text(
        "POINT_ID,GPS_LONG,GPS_LAT,SURVEY_YEAR,OC,N,P,K,pH_CaCl2,Clay,Sand\n"
        "1001,10,5,48,2,2018,18,5,1,6,42,180,6,2,22,0,45,0\n"  # bad: too many cols
    )
    # Re-do with proper formatting (decimal-comma but consistent columns)
    src.write_text(
        "POINT_ID;GPS_LONG;GPS_LAT;SURVEY_YEAR;OC;N;P;K;pH_CaCl2;Clay;Sand\n"
        "1001;10,52;48,12;2018;18,5;1,6;42;180;6,2;22,0;45,0\n"
        "1002;-15,0;48,12;2018;12,0;0,9;30;120;5,8;15,0;55,0\n"  # outside AOI later
    )

    dst = tmp_path / "lucas.csv"
    normalize_lucas_esdac_csv(src, dst)

    text = dst.read_text()
    assert text.startswith(
        "point_id,lon,lat,year,soc,nitrogen,phosphorus,potassium,ph,clay_pct,sand_pct\n"
    )
    rows = text.strip().splitlines()[1:]
    assert len(rows) == 2
    first = rows[0].split(",")
    assert first[0] == "1001"
    assert float(first[1]) == pytest.approx(10.52)
    assert float(first[2]) == pytest.approx(48.12)
    assert int(first[3]) == 2018
    assert float(first[4]) == pytest.approx(18.5)  # soc
    assert float(first[8]) == pytest.approx(6.2)   # ph


def test_normalize_then_lucas_adapter_round_trip(tmp_path):
    src = tmp_path / "esdac.csv"
    src.write_text(
        "POINT_ID;GPS_LONG;GPS_LAT;SURVEY_YEAR;OC;pH_CaCl2\n"
        "P1;10,5;48,2;2018;18,5;6,2\n"
    )
    dst = tmp_path / "norm.csv"
    normalize_lucas_esdac_csv(src, dst)

    aoi = AOI(aoi_id="eu", bbox=BoundingBox(0.0, 35.0, 30.0, 60.0))
    window = TimeWindow(
        start=int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp()),
        end=int(datetime(2018, 12, 31, tzinfo=timezone.utc).timestamp()),
    )
    out = list(LUCASAdapter(csv_path=dst).fetch(aoi, window))
    assert len(out) == 1
    assert out[0].properties["soc"] == pytest.approx(18.5)
    assert out[0].properties["ph"] == pytest.approx(6.2)


def test_normalize_lucas_missing_columns_raises(tmp_path):
    src = tmp_path / "bad.csv"
    src.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="lon/lat/year"):
        normalize_lucas_esdac_csv(src, tmp_path / "out.csv")
