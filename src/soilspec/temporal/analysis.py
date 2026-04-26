"""Temporal analysis module — expert ensemble.

Three independently versioned experts:
  - trend detection
  - anomaly identification
  - behavior classification

A soil-evolution inference model produces preliminary cross-time relationships;
the experts refine the analysis and the combiner emits a single
`TemporalSignals` record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..registry import Registry
from ..types import SoilFunctionalProperties, TemporalFeatureSet, TemporalSignals


class TemporalExpert(Protocol):
    name: str

    def evaluate(self, features: TemporalFeatureSet) -> dict[str, float]: ...


# ---------------------------------------------------------------------------
# Experts
# ---------------------------------------------------------------------------


class TrendDetectionExpert:
    name = "trend_detection"

    def evaluate(self, features: TemporalFeatureSet) -> dict[str, float]:
        # mean trend across feature dimensions
        if not features.trend:
            return {"slope": 0.0, "label": 0.0}
        slopes = list(features.trend.values())
        mean = float(sum(slopes) / len(slopes))
        return {"slope": mean}


class AnomalyIdentificationExpert:
    name = "anomaly_identification"

    def evaluate(self, features: TemporalFeatureSet) -> dict[str, float]:
        if not features.volatility:
            return {"score": 0.0}
        vol = max(features.volatility.values())
        dev = max(abs(v) for v in features.baseline_deviation.values()) if features.baseline_deviation else 0.0
        score = float(min(1.0, 0.5 * vol + 0.5 * dev))
        return {"score": score}


class BehaviorClassificationExpert:
    name = "behavior_classification"

    LABELS = ("improving", "stable", "stressed", "degrading")

    def evaluate(self, features: TemporalFeatureSet) -> dict[str, float]:
        slopes = list(features.trend.values()) or [0.0]
        vol = list(features.volatility.values()) or [0.0]
        mean_slope = sum(slopes) / len(slopes)
        mean_vol = sum(vol) / len(vol)
        # logits-shaped scores
        return {
            "improving": float(max(0.0, mean_slope) - 0.5 * mean_vol),
            "stable":    float(0.5 - abs(mean_slope) - 0.5 * mean_vol),
            "stressed":  float(max(0.0, mean_vol - 0.1)),
            "degrading": float(max(0.0, -mean_slope) - 0.25 * mean_vol),
        }


ExpertRegistry: Registry[TemporalExpert] = Registry("temporal-experts")
ExpertRegistry.register("trend_detection", lambda: TrendDetectionExpert())
ExpertRegistry.register("anomaly_identification", lambda: AnomalyIdentificationExpert())
ExpertRegistry.register("behavior_classification", lambda: BehaviorClassificationExpert())


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


@dataclass
class ExpertEnsembleConfig:
    experts: tuple[str, ...] = ("trend_detection", "anomaly_identification", "behavior_classification")


@dataclass
class TemporalAnalysisModule:
    config: ExpertEnsembleConfig = field(default_factory=ExpertEnsembleConfig)

    def __post_init__(self) -> None:
        self._experts = {name: ExpertRegistry.create(name) for name in self.config.experts}

    def analyze(
        self,
        features: TemporalFeatureSet,
        current: SoilFunctionalProperties | None = None,
    ) -> TemporalSignals:
        outputs: dict[str, dict[str, float]] = {}
        for name, expert in self._experts.items():
            outputs[name] = expert.evaluate(features)
        # combine
        slope = outputs.get("trend_detection", {}).get("slope", 0.0)
        anomaly = outputs.get("anomaly_identification", {}).get("score", 0.0)
        behavior_scores = outputs.get("behavior_classification", {})
        if behavior_scores:
            label = max(behavior_scores, key=behavior_scores.get)  # type: ignore[arg-type]
        else:
            label = "stable"
        trend_label = "increasing" if slope > 0.05 else ("decreasing" if slope < -0.05 else "flat")
        return TemporalSignals(
            tile_id=features.tile_id,
            trend_label=trend_label,
            anomaly_score=anomaly,
            behavior_class=label,
            expert_outputs=outputs,
        )
