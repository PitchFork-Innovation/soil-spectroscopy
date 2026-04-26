import pytest

from soilspec.capability import (
    CapabilityScoringEngine, DEFAULT_BOUNDARIES, DEFAULT_WEIGHTS, RulesEngine,
)
from soilspec.types import (
    CAPABILITY_CLASSES, CharacteristicScores, SoilFunctionalProperties,
    TemporalFeatureSet, TemporalSignals,
)


def _features(tile="t"):
    return TemporalFeatureSet(
        tile_id=tile,
        trend={"a": 0.1}, rate_of_change={"a": 0.0},
        persistence={"a": 0.5}, volatility={"a": 0.1},
        recovery={"a": 0.5}, baseline_deviation={"a": 0.0}, n_samples=5,
    )


def _properties(tile="t", smi=0.6, infiltration=0.7, erosion=0.2):
    return SoilFunctionalProperties(
        tile_id=tile, time=0,
        properties={"smi": smi, "infiltration_potential": infiltration, "erosion_susceptibility": erosion},
        uncertainty={"smi": 0.05, "infiltration_potential": 0.05, "erosion_susceptibility": 0.05},
    )


def _signals(tile="t"):
    return TemporalSignals(tile_id=tile, trend_label="flat", anomaly_score=0.1, behavior_class="stable")


def test_scoring_returns_unit_interval():
    out = CapabilityScoringEngine().score(_features(), _properties(), _signals())
    for v in out.scores.values():
        assert 0.0 <= v <= 1.0


def test_scoring_is_deterministic():
    a = CapabilityScoringEngine().score(_features(), _properties(), _signals())
    b = CapabilityScoringEngine().score(_features(), _properties(), _signals())
    assert a.scores == b.scores


def test_rules_engine_classifies_into_known_classes():
    rules = RulesEngine()
    out = rules.classify(CharacteristicScores(tile_id="t", scores={k: 1.0 for k in DEFAULT_WEIGHTS}))
    assert out.capability_class == "I"
    out = rules.classify(CharacteristicScores(tile_id="t", scores={k: 0.0 for k in DEFAULT_WEIGHTS}))
    assert out.capability_class == "VIII"


def test_rules_engine_explanation_carries_weights_and_scores():
    rules = RulesEngine()
    out = rules.classify(CharacteristicScores(tile_id="t", scores={k: 0.5 for k in DEFAULT_WEIGHTS}))
    assert out.explanation["weights"] == dict(DEFAULT_WEIGHTS)
    assert "scores" in out.explanation
    assert "rules_version" in out.explanation


def test_rules_engine_validates_weights_sum_to_one():
    with pytest.raises(ValueError):
        RulesEngine(weights={"a": 0.5})


def test_rules_classes_are_in_known_vocabulary():
    for _, cls in DEFAULT_BOUNDARIES:
        assert cls in CAPABILITY_CLASSES
