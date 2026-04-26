"""Capability scoring and rules engine.

Two stages:
  1. Capability scoring engine: ML head produces characteristic scores in [0,1]
     per dimension from temporal decision features.
  2. Rules engine: aggregates characteristic scores with weights and a class
     boundary table into an ordinal land capability class (I..VIII).

Weight tables and class boundaries are versioned in the model store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .types import (
    CAPABILITY_CLASSES, CapabilityClassification, CharacteristicScores,
    SoilFunctionalProperties, TemporalFeatureSet, TemporalSignals,
)


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------


@dataclass
class CapabilityScoringEngine:
    """Deterministic scoring head."""

    def score(
        self,
        features: TemporalFeatureSet,
        properties: SoilFunctionalProperties,
        signals: TemporalSignals,
    ) -> CharacteristicScores:
        smi = float(properties.properties.get("smi", 0.0))
        infiltration = float(properties.properties.get("infiltration_potential", 0.0))
        erosion = float(properties.properties.get("erosion_susceptibility", 0.0))
        # Stability score is high when volatility is low and trend is non-negative
        vol = sum(features.volatility.values()) / max(len(features.volatility), 1)
        trend = sum(features.trend.values()) / max(len(features.trend), 1)
        stability = max(0.0, min(1.0, 0.5 - vol + 0.5 * max(0.0, trend) + 0.5))
        # Resilience: high persistence + recovery, penalize anomalies
        persistence = sum(features.persistence.values()) / max(len(features.persistence), 1)
        recovery = sum(features.recovery.values()) / max(len(features.recovery), 1)
        resilience = max(0.0, min(1.0, 0.4 * (persistence + 1) / 2 + 0.4 * recovery + 0.2 * (1 - signals.anomaly_score)))
        scores = {
            "moisture_capacity": float(max(0.0, min(1.0, smi))),
            "infiltration_capacity": float(max(0.0, min(1.0, infiltration))),
            "erosion_resistance": float(max(0.0, min(1.0, 1.0 - erosion))),
            "stability": float(max(0.0, min(1.0, stability))),
            "resilience": float(max(0.0, min(1.0, resilience))),
        }
        return CharacteristicScores(tile_id=features.tile_id, scores=scores)


# ---------------------------------------------------------------------------
# Rules engine
# ---------------------------------------------------------------------------


DEFAULT_WEIGHTS: dict[str, float] = {
    "moisture_capacity": 0.25,
    "infiltration_capacity": 0.20,
    "erosion_resistance": 0.25,
    "stability": 0.15,
    "resilience": 0.15,
}

# class boundaries on the aggregate score, ascending
DEFAULT_BOUNDARIES: tuple[tuple[float, str], ...] = (
    (0.875, "I"),
    (0.750, "II"),
    (0.625, "III"),
    (0.500, "IV"),
    (0.375, "V"),
    (0.250, "VI"),
    (0.125, "VII"),
    (0.000, "VIII"),
)


@dataclass
class RulesEngine:
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    boundaries: tuple[tuple[float, str], ...] = DEFAULT_BOUNDARIES
    version: str = "v1"

    def __post_init__(self) -> None:
        wsum = sum(self.weights.values())
        if abs(wsum - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0; got {wsum}")
        for _, cls in self.boundaries:
            if cls not in CAPABILITY_CLASSES:
                raise ValueError(f"unknown capability class in boundaries: {cls}")

    def classify(self, scores: CharacteristicScores) -> CapabilityClassification:
        agg = sum(self.weights.get(k, 0.0) * v for k, v in scores.scores.items())
        agg = float(max(0.0, min(1.0, agg)))
        for threshold, cls in self.boundaries:
            if agg >= threshold:
                return CapabilityClassification(
                    tile_id=scores.tile_id,
                    capability_class=cls,
                    score=agg,
                    explanation={
                        "weights": dict(self.weights),
                        "scores": dict(scores.scores),
                        "rules_version": self.version,
                    },
                )
        # safety: should never reach because boundaries cover 0.0
        return CapabilityClassification(  # pragma: no cover - defensive
            tile_id=scores.tile_id,
            capability_class="VIII",
            score=agg,
            explanation={"weights": dict(self.weights), "scores": dict(scores.scores), "rules_version": self.version},
        )
