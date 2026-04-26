import numpy as np
import pytest

from soilspec.storage import StorageTierManager
from soilspec.temporal import (
    InsufficientHistoryError, PayloadConflictError, SufficiencyCriteria,
    TemporalAnalysisModule, TemporalDataset, TemporalFeatureExtractor,
)


@pytest.fixture
def temporal():
    return TemporalDataset(StorageTierManager())


# ----------------------------- dataset -------------------------------------


def test_append_query_round_trip(temporal):
    v = np.array([1.0, 2.0, 3.0])
    temporal.append("c1", 100, v)
    out = temporal.query_by_tile_and_time("c1", 100)
    assert np.array_equal(out, v)


def test_query_returns_time_ascending(temporal):
    v1 = np.array([1.0])
    v2 = np.array([2.0])
    temporal.append("c1", 200, v2)
    temporal.append("c1", 100, v1)
    series = temporal.series("c1")
    assert series.times == [100, 200]


def test_idempotent_insert(temporal):
    v = np.array([1.0, 2.0])
    temporal.append("c1", 100, v)
    temporal.append("c1", 100, v)  # no-op
    assert len(temporal.series("c1")) == 1


def test_conflicting_insert_raises(temporal):
    temporal.append("c1", 100, np.array([1.0]))
    with pytest.raises(PayloadConflictError):
        temporal.append("c1", 100, np.array([2.0]))


def test_sufficiency_criteria(temporal):
    crit = SufficiencyCriteria(min_samples=3)
    temporal.append("c1", 1, np.array([1.0]))
    temporal.append("c1", 2, np.array([2.0]))
    assert not temporal.sufficient("c1", crit)
    temporal.append("c1", 3, np.array([3.0]))
    assert temporal.sufficient("c1", crit)


def test_sufficiency_max_gap(temporal):
    crit = SufficiencyCriteria(min_samples=2, max_gap=10)
    temporal.append("c1", 0, np.array([1.0]))
    temporal.append("c1", 100, np.array([2.0]))
    assert not temporal.sufficient("c1", crit)


# ----------------------------- features ------------------------------------


def _seed_series(temporal, cell="c1", n=5, dim=3):
    rng = np.random.default_rng(0)
    for i in range(n):
        v = rng.standard_normal(dim).astype(np.float32) + i * 0.1
        temporal.append(cell, i * 100, v)


def test_feature_extraction_short_series_raises(temporal):
    temporal.append("c1", 0, np.array([1.0]))
    extractor = TemporalFeatureExtractor(min_samples=3)
    with pytest.raises(InsufficientHistoryError):
        extractor.extract(temporal.series("c1"))


def test_features_have_documented_keys(temporal):
    _seed_series(temporal)
    f = TemporalFeatureExtractor(min_samples=3).extract(temporal.series("c1"))
    assert f.n_samples == 5
    assert set(f.trend) == set(f.rate_of_change) == set(f.persistence)
    assert set(f.volatility) == set(f.recovery) == set(f.baseline_deviation)


def test_feature_baseline_uses_provided_baseline(temporal):
    _seed_series(temporal)
    extractor = TemporalFeatureExtractor(
        min_samples=3, baselines={"c1": {"dim_0": 0.0}}
    )
    f = extractor.extract(temporal.series("c1"))
    assert "dim_0" in f.baseline_deviation


# --------------------------- expert analysis -------------------------------


def test_temporal_analysis_emits_signals(temporal):
    _seed_series(temporal)
    f = TemporalFeatureExtractor(min_samples=3).extract(temporal.series("c1"))
    sig = TemporalAnalysisModule().analyze(f)
    assert sig.tile_id == "c1"
    assert sig.trend_label in {"increasing", "decreasing", "flat"}
    assert sig.behavior_class in {"improving", "stable", "stressed", "degrading"}
    assert 0.0 <= sig.anomaly_score <= 1.0
    assert set(sig.expert_outputs) == {
        "trend_detection", "anomaly_identification", "behavior_classification"
    }
