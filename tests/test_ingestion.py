import pytest

from soilspec.ingestion import (
    AdapterRegistry, MCP_TOOL_SCHEMA, MCPRequestError, MetadataParser,
    MetadataValidationError, UnreachableSourceError,
)
from soilspec.storage import StorageTier
from soilspec.types import (
    AOI, AssetMetadata, BoundingBox, SENTINEL1, SENTINEL2, VECTOR, TimeWindow,
)


def test_adapter_registry_has_expected_modalities():
    names = AdapterRegistry.names()
    for m in (SENTINEL1, SENTINEL2, VECTOR):
        assert m in names


def test_synthetic_s2_adapter_is_deterministic(aoi, small_window):
    a = AdapterRegistry.create(SENTINEL2)
    out1 = list(a.fetch(aoi, small_window))
    out2 = list(a.fetch(aoi, small_window))
    assert [o.observation_id for o in out1] == [o.observation_id for o in out2]
    assert out1[0].metadata.bands == out2[0].metadata.bands


def test_metadata_parser_accepts_well_formed_metadata(aoi, small_window):
    a = AdapterRegistry.create(SENTINEL1)
    asset = next(iter(a.fetch(aoi, small_window)))
    meta = MetadataParser().parse(asset)
    assert isinstance(meta, AssetMetadata)
    assert meta.modality == SENTINEL1
    assert meta.timestamp >= 0


def test_metadata_parser_rejects_bad_dict_input():
    with pytest.raises(MetadataValidationError):
        MetadataParser().parse_dict({"observation_id": "o1"})  # missing fields


def test_ingestion_persists_to_raw_store(ingestion, storage, aoi, small_window):
    handles = ingestion.fetch(aoi, small_window, [SENTINEL2, VECTOR])
    assert len(handles) > 0
    for h in handles:
        assert storage.exists(StorageTier.RAW, h.storage_key)


def test_ingestion_is_idempotent(ingestion, storage, aoi, small_window):
    handles_a = ingestion.fetch(aoi, small_window, [SENTINEL2])
    keys_a = sorted(h.storage_key for h in handles_a)
    handles_b = ingestion.fetch(aoi, small_window, [SENTINEL2])
    keys_b = sorted(h.storage_key for h in handles_b)
    assert keys_a == keys_b


def test_ingestion_skips_unknown_modality(ingestion, aoi, small_window):
    # 'hyperspectral' is in the type vocabulary but no adapter registered here
    handles = ingestion.fetch(aoi, small_window, ["hyperspectral"])
    assert handles == []


def test_mcp_list_returns_observation_summaries(mcp_adapter, aoi, small_window):
    out = mcp_adapter.list({
        "aoi_id": aoi.aoi_id,
        "bbox": [aoi.bbox.min_lon, aoi.bbox.min_lat, aoi.bbox.max_lon, aoi.bbox.max_lat],
        "time_window": {"start": small_window.start, "end": small_window.end},
        "modality": SENTINEL2,
    })
    assert len(out) > 0
    for entry in out:
        for k in ("observation_id", "provider", "modality", "timestamp"):
            assert k in entry


def test_mcp_describe_then_fetch_round_trip(mcp_adapter, aoi, small_window):
    out = mcp_adapter.list({
        "aoi_id": aoi.aoi_id,
        "bbox": [aoi.bbox.min_lon, aoi.bbox.min_lat, aoi.bbox.max_lon, aoi.bbox.max_lat],
        "time_window": {"start": small_window.start, "end": small_window.end},
        "modality": SENTINEL1,
    })
    obs_id = out[0]["observation_id"]
    desc = mcp_adapter.describe({"observation_id": obs_id})
    assert desc["observation_id"] == obs_id
    fetched = mcp_adapter.fetch({"observation_id": obs_id})
    assert fetched["observation_id"] == obs_id
    assert fetched["modality"] == SENTINEL1


def test_mcp_unknown_modality_raises(mcp_adapter, aoi, small_window):
    with pytest.raises(UnreachableSourceError):
        mcp_adapter.list({
            "aoi_id": aoi.aoi_id,
            "bbox": [0, 0, 1, 1],
            "time_window": {"start": 0, "end": 1},
            "modality": "nonexistent",
        })


def test_mcp_request_validation(mcp_adapter):
    with pytest.raises(MCPRequestError):
        mcp_adapter.list({"aoi_id": "x"})  # missing required fields


def test_mcp_tool_schema_is_well_formed():
    assert MCP_TOOL_SCHEMA["title"] == "soilspec.mcp.tools"
    tools = MCP_TOOL_SCHEMA["properties"]["tools"]["properties"]
    for tool in ("list", "describe", "fetch"):
        assert tool in tools
        assert "request" in tools[tool]["properties"]
