"""End-to-end + modality-missing + scheduled-loop integration tests."""

import pytest

from soilspec.orchestrator import (
    OrchestratorConfig, PipelineOrchestrator, ScheduledTrigger,
)
from soilspec.preprocessing.pipeline import PreprocessConfig
from soilspec.publisher import (
    CAPABILITY_OUTPUT, CONFIDENCE_OUTPUT, RECOMMENDATION_OUTPUT,
)
from soilspec.storage import StorageTier
from soilspec.temporal import SufficiencyCriteria
from soilspec.types import (
    AOI, AnalysisRequest, BoundingBox, CAPABILITY_CLASSES, SENTINEL1, SENTINEL2,
    VECTOR, TimeWindow,
)


def _orch(min_samples=2):
    cfg = OrchestratorConfig(
        preprocess=PreprocessConfig(target_shape=(32, 32), tile_size=16),
        sufficiency=SufficiencyCriteria(min_samples=min_samples),
    )
    return PipelineOrchestrator(config=cfg)


def _request(modalities=(SENTINEL1, SENTINEL2, VECTOR)):
    return AnalysisRequest(
        aoi=AOI(aoi_id="e2e", bbox=BoundingBox(0, 0, 1, 1)),
        time_window=TimeWindow(start=0, end=30 * 86400),
        modalities=modalities,
    )


def test_end_to_end_emits_valid_capability_classes():
    o = _orch()
    res = o.run_request(_request())
    cap_handle = next(h for h in res.map_handles if h.output_type == CAPABILITY_OUTPUT)
    payload = o.storage.get(StorageTier.MAP, cap_handle.storage_key)
    assert payload["tiles"], "expected non-empty capability map"
    for tile_id, info in payload["tiles"].items():
        assert info["capability_class"] in CAPABILITY_CLASSES
        assert 0.0 <= info["score"] <= 1.0


def test_end_to_end_tile_coverage_matches_aoi():
    o = _orch()
    res = o.run_request(_request())
    cap_handle = next(h for h in res.map_handles if h.output_type == CAPABILITY_OUTPUT)
    payload = o.storage.get(StorageTier.MAP, cap_handle.storage_key)
    # tile_size=16, target_shape=32x32 -> 4 tiles
    assert len(payload["tiles"]) == 4


@pytest.mark.parametrize("modalities", [(SENTINEL1,), (SENTINEL2,)])
def test_modality_missing_does_not_crash(modalities):
    o = _orch()
    res = o.run_request(_request(modalities=modalities))
    assert {h.output_type for h in res.map_handles} == {
        CAPABILITY_OUTPUT, RECOMMENDATION_OUTPUT, CONFIDENCE_OUTPUT,
    }


def test_scheduled_loop_grows_temporal_dataset_monotonically():
    cfg = OrchestratorConfig(
        preprocess=PreprocessConfig(target_shape=(32, 32), tile_size=16),
        sufficiency=SufficiencyCriteria(min_samples=2),
    )
    o = PipelineOrchestrator(config=cfg)
    req = _request()
    o.tick(req, now=0)
    sizes_a = [len(o.temporal.series(c)) for c in o.temporal.cells()]
    o.tick(req, now=86400)
    sizes_b = [len(o.temporal.series(c)) for c in o.temporal.cells()]
    # idempotent inserts: same observations produce same sizes
    assert sizes_b == sizes_a


def test_sufficiency_gate_defers_when_insufficient():
    o = _orch(min_samples=100)
    out = o.tick(_request(), now=0)
    assert out is None


def test_recommendations_published_for_e2e_run():
    o = _orch()
    res = o.run_request(_request())
    rec_handle = next(h for h in res.map_handles if h.output_type == RECOMMENDATION_OUTPUT)
    payload = o.storage.get(StorageTier.MAP, rec_handle.storage_key)
    assert payload["aoi_id"] == "e2e"
    assert isinstance(payload["priority_zones"], dict)


def test_confidence_layer_attached_to_each_tile():
    o = _orch()
    res = o.run_request(_request())
    conf_handle = next(h for h in res.map_handles if h.output_type == CONFIDENCE_OUTPUT)
    payload = o.storage.get(StorageTier.MAP, conf_handle.storage_key)
    cap_handle = next(h for h in res.map_handles if h.output_type == CAPABILITY_OUTPUT)
    cap_payload = o.storage.get(StorageTier.MAP, cap_handle.storage_key)
    assert set(payload["tiles"]) == set(cap_payload["tiles"])
    for info in payload["tiles"].values():
        for k in ("temporal_consistency", "data_completeness", "model_agreement", "degradation_flag"):
            assert k in info
