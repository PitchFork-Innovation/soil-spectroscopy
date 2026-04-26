"""Recommendation logic engine.

Translates temporal signals + soil functional properties into actionable
recommendations: priority zones, risk areas, management actions. Rule-based
implementation here; learned and hybrid variants register through the
strategy registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol

from .registry import Registry
from .types import (
    AOI, RecommendationLayers, SoilFunctionalProperties, TemporalSignals,
)


class RecommendationStrategy(Protocol):
    name: str

    def recommend(
        self,
        aoi: AOI,
        signals: Mapping[str, TemporalSignals],
        properties: Mapping[str, SoilFunctionalProperties],
    ) -> RecommendationLayers: ...


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class RuleBasedRecommender:
    name = "rules"

    def __init__(
        self,
        priority_threshold: float = 0.6,
        risk_threshold: float = 0.5,
    ) -> None:
        self.priority_threshold = priority_threshold
        self.risk_threshold = risk_threshold

    def recommend(self, aoi, signals, properties):
        priority_zones: dict[str, str] = {}
        risk_areas: dict[str, str] = {}
        actions: dict[str, list[str]] = {}
        for tile_id, props in properties.items():
            sig = signals.get(tile_id)
            smi = props.properties.get("smi", 0.0)
            erosion = props.properties.get("erosion_susceptibility", 0.0)
            infiltration = props.properties.get("infiltration_potential", 0.0)
            tile_actions: list[str] = []
            if smi < 0.3:
                priority_zones[tile_id] = "irrigate_high"
                tile_actions.append("apply_irrigation")
            elif smi > self.priority_threshold:
                priority_zones[tile_id] = "irrigate_hold"
            else:
                priority_zones[tile_id] = "monitor"
            if erosion > self.risk_threshold:
                risk_areas[tile_id] = "erosion_high"
                tile_actions.append("install_cover_crop")
            elif sig is not None and sig.behavior_class == "degrading":
                risk_areas[tile_id] = "degradation_watch"
            if infiltration > 0.7:
                tile_actions.append("eligible_for_managed_recharge")
            actions[tile_id] = tile_actions
        return RecommendationLayers(
            aoi_id=aoi.aoi_id,
            priority_zones=priority_zones,
            risk_areas=risk_areas,
            management_actions=actions,
        )


class LearnedRecommender:
    """Stub for the learned variant — uses anomaly_score as the signal."""

    name = "learned"

    def recommend(self, aoi, signals, properties):
        priority_zones: dict[str, str] = {}
        risk_areas: dict[str, str] = {}
        actions: dict[str, list[str]] = {}
        for tile_id, sig in signals.items():
            label = "high" if sig.anomaly_score > 0.5 else "normal"
            priority_zones[tile_id] = label
            if sig.anomaly_score > 0.5:
                risk_areas[tile_id] = "anomalous"
                actions[tile_id] = ["investigate"]
            else:
                actions[tile_id] = []
        return RecommendationLayers(
            aoi_id=aoi.aoi_id,
            priority_zones=priority_zones,
            risk_areas=risk_areas,
            management_actions=actions,
        )


class HybridRecommender:
    """Combine rule-based and learned outputs (rules dominate, learned augments actions)."""

    name = "hybrid"

    def __init__(self) -> None:
        self._rules = RuleBasedRecommender()
        self._learned = LearnedRecommender()

    def recommend(self, aoi, signals, properties):
        a = self._rules.recommend(aoi, signals, properties)
        b = self._learned.recommend(aoi, signals, properties)
        merged_actions = {tid: list(a.management_actions.get(tid, [])) for tid in a.management_actions}
        for tid, acts in b.management_actions.items():
            merged_actions.setdefault(tid, [])
            for act in acts:
                if act not in merged_actions[tid]:
                    merged_actions[tid].append(act)
        return RecommendationLayers(
            aoi_id=aoi.aoi_id,
            priority_zones=a.priority_zones,
            risk_areas=a.risk_areas,
            management_actions=merged_actions,
        )


RecommendationRegistry: Registry[RecommendationStrategy] = Registry("recommendation-strategies")
RecommendationRegistry.register("rules", lambda **kw: RuleBasedRecommender(**kw))
RecommendationRegistry.register("learned", lambda **kw: LearnedRecommender(**kw))
RecommendationRegistry.register("hybrid", lambda **kw: HybridRecommender(**kw))


@dataclass
class RecommendationLogicEngine:
    """Wraps the chosen strategy."""

    strategy: str = "rules"

    def __post_init__(self) -> None:
        self._impl = RecommendationRegistry.create(self.strategy)

    def recommend(self, aoi, signals, properties):
        return self._impl.recommend(aoi, signals, properties)
