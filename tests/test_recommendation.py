import pytest

from soilspec.recommendation import (
    HybridRecommender, LearnedRecommender, RecommendationLogicEngine,
    RecommendationRegistry, RuleBasedRecommender,
)
from soilspec.types import (
    AOI, BoundingBox, SoilFunctionalProperties, TemporalSignals,
)


def _aoi():
    return AOI(aoi_id="aoi1", bbox=BoundingBox(0, 0, 1, 1))


def _props(tile_id, smi=0.5, infiltration=0.5, erosion=0.3):
    return SoilFunctionalProperties(
        tile_id=tile_id, time=0,
        properties={"smi": smi, "infiltration_potential": infiltration, "erosion_susceptibility": erosion},
        uncertainty={"smi": 0.0, "infiltration_potential": 0.0, "erosion_susceptibility": 0.0},
    )


def _signals(tile_id, behavior="stable", anomaly=0.1):
    return TemporalSignals(
        tile_id=tile_id, trend_label="flat", anomaly_score=anomaly, behavior_class=behavior,
    )


def test_rules_recommender_schema():
    aoi = _aoi()
    props = {"t1": _props("t1", smi=0.1), "t2": _props("t2", erosion=0.8)}
    sigs = {"t1": _signals("t1"), "t2": _signals("t2", behavior="degrading")}
    out = RuleBasedRecommender().recommend(aoi, sigs, props)
    assert out.aoi_id == "aoi1"
    assert "t1" in out.priority_zones and "t2" in out.priority_zones
    assert "t2" in out.risk_areas
    assert "apply_irrigation" in out.management_actions["t1"]


def test_learned_recommender_thresholds_on_anomaly():
    aoi = _aoi()
    props = {"t1": _props("t1")}
    sigs = {"t1": _signals("t1", anomaly=0.9)}
    out = LearnedRecommender().recommend(aoi, sigs, props)
    assert out.priority_zones["t1"] == "high"
    assert "investigate" in out.management_actions["t1"]


def test_hybrid_combines_actions_without_duplicates():
    aoi = _aoi()
    props = {"t1": _props("t1", smi=0.1)}
    sigs = {"t1": _signals("t1", anomaly=0.9)}
    out = HybridRecommender().recommend(aoi, sigs, props)
    actions = out.management_actions["t1"]
    assert "apply_irrigation" in actions
    assert "investigate" in actions
    assert len(actions) == len(set(actions))


def test_engine_dispatches_to_strategy():
    eng = RecommendationLogicEngine(strategy="rules")
    out = eng.recommend(_aoi(), {}, {})
    assert out.priority_zones == {}


def test_registry_exposes_all_strategies():
    assert {"rules", "learned", "hybrid"} <= set(RecommendationRegistry.names())
