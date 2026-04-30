"""Capability scoring and rules engine.

Two stages:
  1. Capability scoring engine: ML head produces characteristic scores in [0,1]
     per dimension from temporal decision features.
  2. Rules engine: aggregates characteristic scores with weights and a class
     boundary table into an ordinal land capability class (I..VIII).

Weight tables and class boundaries are versioned in the model store.

Also exposes :class:`MeasuredToFunctional` — derives the three functional
properties consumed by the scoring engine (smi / infiltration_potential /
erosion_susceptibility) from directly measurable quantities (soil_moisture,
clay/sand/silt %, SOC, bulk density). Used by the orchestrator when a
trained pipeline is loaded to bridge from measured-property predictions
into the existing functional-property scoring pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .types import (
    CAPABILITY_CLASSES, CapabilityClassification, CharacteristicScores,
    SoilFunctionalProperties, TemporalFeatureSet, TemporalSignals,
)


# ---------------------------------------------------------------------------
# Measured -> Functional derivation
# ---------------------------------------------------------------------------


def _clip01(x: float) -> float:
    if x is None or not (x == x):  # NaN guard
        return 0.0
    return max(0.0, min(1.0, float(x)))


@dataclass
class MeasuredToFunctional:
    """Derive `SoilFunctionalProperties` from measured property predictions.

    Formulas chosen for transparency over fidelity — all three derivations
    are documented soil-science approximations, not learned mappings:

    * **smi** = clip(soil_moisture, 0, 1). Volumetric water content is
      already in [0,1]; the clip handles measurement noise.
    * **infiltration_potential** = sand-/SOC-weighted approximation of the
      Saxton & Rawls (2006) saturated hydraulic conductivity surrogate:
      higher sand fraction → faster infiltration; higher SOC → modest
      enhancement; higher bulk density → reduced infiltration. Output
      normalized to [0,1].
    * **erosion_susceptibility** ≈ USLE K-factor (Wischmeier 1971), simplified.
      Higher silt fraction and lower SOC drive higher K (more erodible).
      Output normalized to [0,1] so it slots into the scoring engine's
      `erosion_susceptibility` slot directly (note the engine treats it as
      "high = bad", inverted to `erosion_resistance` = 1 - this).

    Missing measured properties default to neutral midpoints so the
    pipeline degrades gracefully when (e.g.) ISMN-only training never
    saw clay/sand labels.
    """

    sand_default: float = 40.0
    silt_default: float = 30.0
    clay_default: float = 25.0
    soc_default: float = 15.0          # g/kg
    bulk_density_default: float = 1.4  # g/cm³

    def derive(
        self, tile_id: str, time: int, measured: Mapping[str, float],
        member_outputs: Mapping[str, Mapping[str, float]] | None = None,
    ) -> SoilFunctionalProperties:
        sand = float(measured.get("sand_pct", self.sand_default))
        silt = float(measured.get("silt_pct", self.silt_default))
        clay = float(measured.get("clay_pct", self.clay_default))
        soc = float(measured.get("soc", self.soc_default))
        bd = float(measured.get("bulk_density", self.bulk_density_default))
        sm = float(measured.get("soil_moisture", 0.25))

        smi = _clip01(sm)
        # Saxton & Rawls-inspired infiltration proxy (normalized).
        infiltration_raw = (
            0.6 * (sand / 100.0)
            + 0.2 * min(1.0, soc / 30.0)
            - 0.3 * max(0.0, (bd - 1.0))
            - 0.1 * (clay / 100.0)
        )
        # rescale from approximate [-0.4, 0.8] -> [0,1]
        infiltration_potential = _clip01(0.5 + infiltration_raw)

        # USLE K proxy. Wischmeier nomograph: K rises with silt+very_fine_sand,
        # falls with SOC. We approximate with silt fraction and inverse SOC.
        m_factor = (silt + 0.2 * sand) * (100.0 - clay)  # higher when erodible
        soc_term = max(0.0, 12.0 - soc)  # SOC above 12 g/kg saturates
        # k_raw scaled empirically so realistic inputs land near 0.2-0.5
        k_raw = (m_factor * soc_term) / 1.0e5
        erosion = _clip01(k_raw)

        # Per-property uncertainty: trivial floor (0.0) when derived; any
        # uncertainty information would have to come from the upstream
        # measured-property predictor.
        return SoilFunctionalProperties(
            tile_id=tile_id, time=int(time),
            properties={
                "smi": smi,
                "infiltration_potential": infiltration_potential,
                "erosion_susceptibility": erosion,
            },
            uncertainty={
                "smi": 0.0,
                "infiltration_potential": 0.0,
                "erosion_susceptibility": 0.0,
            },
            member_outputs=dict(member_outputs) if member_outputs else {},
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
