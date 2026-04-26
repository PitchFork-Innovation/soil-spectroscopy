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
    AOI, AnalysisRequest, BoundingBox, SENTINEL1, SENTINEL2, VECTOR, TimeWindow,
)


def _request(start=0, end=30 * 86400, modalities=(SENTINEL1, SENTINEL2, VECTOR)):
    return AnalysisRequest(
        aoi=AOI(aoi_id="aoi1", bbox=BoundingBox(0, 0, 1, 1)),
        time_window=TimeWindow(start=start, end=end),
        modalities=modalities,
    )


def _orch():
    cfg = OrchestratorConfig(
        preprocess=PreprocessConfig(target_shape=(32, 32), tile_size=16),
        sufficiency=SufficiencyCriteria(min_samples=2),
    )
    return PipelineOrchestrator(config=cfg)


def test_run_request_publishes_three_layers():
    o = _orch()
    res = o.run_request(_request())
    assert res.aoi_id == "aoi1"
    assert {h.output_type for h in res.map_handles} == {
        CAPABILITY_OUTPUT, RECOMMENDATION_OUTPUT, CONFIDENCE_OUTPUT,
    }


def test_run_request_writes_all_storage_tiers():
    o = _orch()
    o.run_request(_request())
    assert any(o.storage.list(StorageTier.RAW))
    assert any(o.storage.list(StorageTier.PREPROCESSED))
    assert any(o.storage.list(StorageTier.TEMPORAL))
    assert any(o.storage.list(StorageTier.MAP))


def test_run_request_works_with_only_s2():
    o = _orch()
    res = o.run_request(_request(modalities=(SENTINEL2,)))
    assert {h.output_type for h in res.map_handles} == {
        CAPABILITY_OUTPUT, RECOMMENDATION_OUTPUT, CONFIDENCE_OUTPUT,
    }


def test_run_request_works_with_only_s1():
    o = _orch()
    res = o.run_request(_request(modalities=(SENTINEL1,)))
    assert {h.output_type for h in res.map_handles} == {
        CAPABILITY_OUTPUT, RECOMMENDATION_OUTPUT, CONFIDENCE_OUTPUT,
    }


def test_tick_returns_none_when_insufficient():
    cfg = OrchestratorConfig(
        preprocess=PreprocessConfig(target_shape=(32, 32), tile_size=16),
        sufficiency=SufficiencyCriteria(min_samples=100),
    )
    o = PipelineOrchestrator(config=cfg)
    assert o.tick(_request()) is None


def test_scheduled_trigger_fake_clock():
    trig = ScheduledTrigger(interval_seconds=10, clock=lambda: 0)
    assert trig.due(now=0)
    trig.fire(now=0)
    assert not trig.due(now=5)
    assert trig.due(now=15)
