"""Planetary Computer adapter tests.

These tests don't hit the network. They mock pystac-client + rasterio so
they verify the adapter constructs the right STAC query, scales reflectance
correctly, builds the cloud mask from SCL, and converts S1 RTC to dB.

Skipped entirely if the [planetary] extra isn't installed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("pystac_client")
pytest.importorskip("planetary_computer")
pytest.importorskip("rasterio")

from soilspec.ingestion.adapters import UnreachableSourceError
from soilspec.ingestion.planetary import (
    PlanetaryComputerSentinel1Adapter,
    PlanetaryComputerSentinel2Adapter,
    S2_BANDS,
)
from soilspec.types import AOI, BoundingBox, SENTINEL1, SENTINEL2, TimeWindow


def _aoi():
    return AOI(aoi_id="test-aoi", bbox=BoundingBox(0.0, 0.0, 0.5, 0.5))


def _window():
    return TimeWindow(
        start=int(datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp()),
        end=int(datetime(2024, 6, 30, tzinfo=timezone.utc).timestamp()),
    )


def _make_item(item_id: str, dt: datetime, asset_keys: list[str], extra_props=None):
    assets = {k: SimpleNamespace(href=f"https://blob/{item_id}/{k}.tif") for k in asset_keys}
    props = {"eo:cloud_cover": 5.0}
    if extra_props:
        props.update(extra_props)
    return SimpleNamespace(id=item_id, datetime=dt, assets=assets, properties=props)


def test_s2_adapter_scales_reflectance_and_builds_cloud_mask():
    item = _make_item(
        "S2_TEST_001",
        datetime(2024, 6, 15, 12, tzinfo=timezone.utc),
        list(S2_BANDS) + ["SCL"],
    )

    def fake_read(href, _bbox, target_shape):
        # raw S2 L2A values are uint16-scaled; 5000 → 0.5 reflectance
        if href.endswith("SCL.tif"):
            arr = np.zeros(target_shape, dtype=np.float32)
            arr[0, 0] = 9  # cloud high → masked
            arr[0, 1] = 4  # vegetation → not masked
            return arr
        return np.full(target_shape, 5000.0, dtype=np.float32)

    with patch("soilspec.ingestion.planetary._open_pc_client") as open_client, \
         patch("soilspec.ingestion.planetary._read_windowed", side_effect=fake_read):
        client = MagicMock()
        client.search.return_value.items.return_value = [item]
        open_client.return_value = client

        adapter = PlanetaryComputerSentinel2Adapter(tile_size=8, max_cloud_cover=20.0)
        assets = list(adapter.fetch(_aoi(), _window()))

    # STAC query shape
    args, kwargs = client.search.call_args
    assert kwargs["collections"] == ["sentinel-2-l2a"]
    assert kwargs["query"] == {"eo:cloud_cover": {"lt": 20.0}}
    assert "/" in kwargs["datetime"]

    assert len(assets) == 1
    payload = assets[0].payload
    assert payload["reflectance"].shape == (6, 8, 8)
    assert np.allclose(payload["reflectance"], 0.5)
    assert payload["qa"].shape == (8, 8)
    assert payload["qa"][0, 0] == 1  # cloud
    assert payload["qa"][0, 1] == 0  # clear
    assert assets[0].metadata.bands[-1] == "QA"
    assert assets[0].modality == SENTINEL2


def test_s1_rtc_adapter_converts_to_db():
    item = _make_item(
        "S1_RTC_001",
        datetime(2024, 6, 10, 6, tzinfo=timezone.utc),
        ["vv", "vh"],
    )

    def fake_read(href, _bbox, target_shape):
        # linear sigma0 = 0.1 → 10*log10(0.1) = -10 dB
        return np.full(target_shape, 0.1, dtype=np.float32)

    with patch("soilspec.ingestion.planetary._open_pc_client") as open_client, \
         patch("soilspec.ingestion.planetary._read_windowed", side_effect=fake_read):
        client = MagicMock()
        client.search.return_value.items.return_value = [item]
        open_client.return_value = client

        adapter = PlanetaryComputerSentinel1Adapter(tile_size=8, collection="sentinel-1-rtc")
        assets = list(adapter.fetch(_aoi(), _window()))

    payload = assets[0].payload
    assert payload.shape == (2, 8, 8)
    assert np.allclose(payload, -10.0, atol=1e-4)
    assert assets[0].metadata.extra["calibration"] == "sigma0_db"
    assert assets[0].modality == SENTINEL1


def test_s1_grd_adapter_keeps_raw_values():
    item = _make_item(
        "S1_GRD_001",
        datetime(2024, 6, 12, 6, tzinfo=timezone.utc),
        ["vv", "vh"],
    )

    def fake_read(href, _bbox, target_shape):
        return np.full(target_shape, 250.0, dtype=np.float32)

    with patch("soilspec.ingestion.planetary._open_pc_client") as open_client, \
         patch("soilspec.ingestion.planetary._read_windowed", side_effect=fake_read):
        client = MagicMock()
        client.search.return_value.items.return_value = [item]
        open_client.return_value = client

        adapter = PlanetaryComputerSentinel1Adapter(tile_size=8, collection="sentinel-1-grd")
        assets = list(adapter.fetch(_aoi(), _window()))

    payload = assets[0].payload
    assert np.allclose(payload, 250.0)  # not log-transformed
    assert assets[0].metadata.extra["calibration"] == "raw_dn"


def test_s1_default_collection_depends_on_subscription_key(monkeypatch):
    monkeypatch.delenv("PC_SDK_SUBSCRIPTION_KEY", raising=False)
    a = PlanetaryComputerSentinel1Adapter()
    assert a.collection == "sentinel-1-grd"

    monkeypatch.setenv("PC_SDK_SUBSCRIPTION_KEY", "secret")
    b = PlanetaryComputerSentinel1Adapter()
    assert b.collection == "sentinel-1-rtc"


def test_stac_failure_is_wrapped_in_unreachable():
    with patch("soilspec.ingestion.planetary._open_pc_client") as open_client:
        client = MagicMock()
        client.search.side_effect = RuntimeError("503 Service Unavailable")
        open_client.return_value = client

        adapter = PlanetaryComputerSentinel2Adapter(tile_size=8)
        with pytest.raises(UnreachableSourceError, match="STAC query failed"):
            list(adapter.fetch(_aoi(), _window()))
