"""Ensemble inference engine.

Maps fused multimodal embeddings to soil functional property estimates with
calibrated uncertainty. Internal architecture per the patent:
  1. lifting layer projects fused embedding into higher-dim space
  2. parallel members: classic ML, deep learning, mathematical interpolation
  3. fusion meta-model combines members into a unified estimate

Members and the meta-model are swappable through `EnsembleMemberRegistry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from ._seed import stable_seed
from .registry import Registry
from .types import (
    SOIL_PROPERTY_NAMES, FusedRepresentation, SoilFunctionalProperties,
)


class EnsembleMember(Protocol):
    name: str

    def predict(self, lifted: np.ndarray) -> dict[str, float]: ...


# ---------------------------------------------------------------------------
# Lifting layer
# ---------------------------------------------------------------------------


@dataclass
class LiftingLayer:
    """Projects fused embeddings into a higher-dimensional space."""

    in_dim: int
    out_dim: int
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(stable_seed("lifting", self.in_dim, self.out_dim, self.seed))
        self._w = rng.standard_normal(size=(self.out_dim, self.in_dim)).astype(np.float32) / np.sqrt(self.in_dim)
        self._b = rng.standard_normal(size=self.out_dim).astype(np.float32) * 0.01

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self._w @ x.astype(np.float32) + self._b)


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------


def _bounded(x: float) -> float:
    """Map any real number to [0, 1] via sigmoid for property scores."""
    return float(1.0 / (1.0 + np.exp(-x)))


class ClassicMLMember:
    """Linear-regression-shaped baseline. Deterministic per (in_dim, seed)."""

    name = "classic_ml"

    def __init__(self, in_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(stable_seed(self.name, in_dim, seed))
        self._w = rng.standard_normal(size=(len(SOIL_PROPERTY_NAMES), in_dim)).astype(np.float32) / np.sqrt(in_dim)

    def predict(self, lifted: np.ndarray) -> dict[str, float]:
        raw = self._w @ lifted
        return {name: _bounded(float(raw[i])) for i, name in enumerate(SOIL_PROPERTY_NAMES)}


class DeepLearningMember:
    """Tiny 2-layer MLP."""

    name = "deep_learning"

    def __init__(self, in_dim: int, hidden: int = 16, seed: int = 0) -> None:
        rng = np.random.default_rng(stable_seed(self.name, in_dim, hidden, seed))
        self._w1 = rng.standard_normal(size=(hidden, in_dim)).astype(np.float32) / np.sqrt(in_dim)
        self._w2 = rng.standard_normal(size=(len(SOIL_PROPERTY_NAMES), hidden)).astype(np.float32) / np.sqrt(hidden)

    def predict(self, lifted: np.ndarray) -> dict[str, float]:
        h = np.tanh(self._w1 @ lifted)
        raw = self._w2 @ h
        return {name: _bounded(float(raw[i])) for i, name in enumerate(SOIL_PROPERTY_NAMES)}


class MathematicalInterpolationMember:
    """Channel-aware aggregation: each property reads its capability channel mean."""

    name = "mathematical_interpolation"

    def __init__(self, channels: Mapping[str, slice] | None = None) -> None:
        # Map property -> channel name (best-effort; falls back to mean of all dims)
        self._mapping = {
            "smi": "moisture",
            "infiltration_potential": "infiltration",
            "erosion_susceptibility": "erosion",
        }
        self._channels = dict(channels) if channels else {}

    def with_channels(self, channels: Mapping[str, slice]) -> "MathematicalInterpolationMember":
        return MathematicalInterpolationMember(channels=channels)

    def predict(self, lifted: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        for prop in SOIL_PROPERTY_NAMES:
            ch = self._mapping.get(prop)
            if ch and ch in self._channels:
                segment = lifted[self._channels[ch]]
                out[prop] = _bounded(float(segment.mean()) if segment.size else 0.0)
            else:
                out[prop] = _bounded(float(lifted.mean()))
        return out


EnsembleMemberRegistry: Registry[EnsembleMember] = Registry("ensemble-members")
EnsembleMemberRegistry.register("classic_ml", lambda **kw: ClassicMLMember(**kw))
EnsembleMemberRegistry.register("deep_learning", lambda **kw: DeepLearningMember(**kw))
EnsembleMemberRegistry.register("mathematical_interpolation", lambda **kw: MathematicalInterpolationMember(**kw))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    members: tuple[str, ...] = ("classic_ml", "deep_learning", "mathematical_interpolation")
    lifting_dim: int = 64
    seed: int = 0


class EnsembleInferenceEngine:
    """Lifts -> runs members in parallel -> meta-model combines."""

    def __init__(self, fused_dim: int, channels: Mapping[str, slice], config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self._lift = LiftingLayer(in_dim=fused_dim, out_dim=self.config.lifting_dim, seed=self.config.seed)
        self._channels = channels
        self._members: list[EnsembleMember] = []
        for name in self.config.members:
            if name == "classic_ml":
                self._members.append(ClassicMLMember(in_dim=self.config.lifting_dim, seed=self.config.seed))
            elif name == "deep_learning":
                self._members.append(DeepLearningMember(in_dim=self.config.lifting_dim, seed=self.config.seed))
            elif name == "mathematical_interpolation":
                # remap channels to the lifted dim — if same dim, reuse; else
                # fall back to all-dim mean by passing empty channels.
                use = channels if self.config.lifting_dim == fused_dim else {}
                self._members.append(MathematicalInterpolationMember(channels=use))
            else:
                self._members.append(EnsembleMemberRegistry.create(name, in_dim=self.config.lifting_dim))

    def infer(self, fused: FusedRepresentation) -> SoilFunctionalProperties:
        lifted = self._lift(fused.vector)
        member_outputs: dict[str, dict[str, float]] = {}
        for m in self._members:
            member_outputs[m.name] = m.predict(lifted)
        # meta-model: simple averaging with per-property uncertainty as spread
        properties: dict[str, float] = {}
        uncertainty: dict[str, float] = {}
        for prop in SOIL_PROPERTY_NAMES:
            vals = [out[prop] for out in member_outputs.values()]
            properties[prop] = float(np.mean(vals))
            uncertainty[prop] = float(np.std(vals))
            # the meta-model output must be bounded by member outputs in a
            # documented way: the mean is always within [min, max] of members.
            assert min(vals) - 1e-9 <= properties[prop] <= max(vals) + 1e-9
        return SoilFunctionalProperties(
            tile_id=fused.tile_id,
            time=fused.time,
            properties=properties,
            uncertainty=uncertainty,
            member_outputs=member_outputs,
        )
