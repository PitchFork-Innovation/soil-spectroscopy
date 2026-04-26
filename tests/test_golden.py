"""Golden-file tests.

A small fixed AOI + window is checked into the test suite. The expected
capability map is computed once and stored as a checked-in golden file.
A regression in any layer (encoder, fusion, ensemble, scoring, rules)
surfaces as a mismatch.

Update goldens with `pytest -m golden --update-goldens`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soilspec.orchestrator import (
    OrchestratorConfig, PipelineOrchestrator,
)
from soilspec.preprocessing.pipeline import PreprocessConfig
from soilspec.publisher import CAPABILITY_OUTPUT, RECOMMENDATION_OUTPUT
from soilspec.storage import StorageTier
from soilspec.temporal import SufficiencyCriteria
from soilspec.types import (
    AOI, AnalysisRequest, BoundingBox, SENTINEL1, SENTINEL2, VECTOR, TimeWindow,
)


GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def update_goldens(request):
    return bool(request.config.getoption("--update-goldens", default=False))


def _request():
    return AnalysisRequest(
        aoi=AOI(aoi_id="golden-aoi", bbox=BoundingBox(0, 0, 1, 1)),
        time_window=TimeWindow(start=0, end=30 * 86400),
        modalities=(SENTINEL1, SENTINEL2, VECTOR),
    )


def _orch():
    cfg = OrchestratorConfig(
        preprocess=PreprocessConfig(target_shape=(32, 32), tile_size=16),
        sufficiency=SufficiencyCriteria(min_samples=2),
    )
    return PipelineOrchestrator(config=cfg)


@pytest.mark.golden
def test_capability_classifications_match_golden(update_goldens):
    o = _orch()
    res = o.run_request(_request())
    cap_handle = next(h for h in res.map_handles if h.output_type == CAPABILITY_OUTPUT)
    payload = o.storage.get(StorageTier.MAP, cap_handle.storage_key)
    data = {tid: info["capability_class"] for tid, info in sorted(payload["tiles"].items())}
    path = GOLDEN_DIR / "capability.json"
    if update_goldens or not path.exists():
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
    expected = json.loads(path.read_text())
    assert data == expected


@pytest.mark.golden
def test_recommendation_matches_golden(update_goldens):
    o = _orch()
    res = o.run_request(_request())
    rec_handle = next(h for h in res.map_handles if h.output_type == RECOMMENDATION_OUTPUT)
    payload = o.storage.get(StorageTier.MAP, rec_handle.storage_key)
    data = {
        "priority_zones": dict(sorted(payload["priority_zones"].items())),
        "risk_areas": dict(sorted(payload["risk_areas"].items())),
    }
    path = GOLDEN_DIR / "recommendation.json"
    if update_goldens or not path.exists():
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
    expected = json.loads(path.read_text())
    assert data == expected
