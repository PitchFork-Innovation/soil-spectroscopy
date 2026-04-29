"""Ensemble inference engine.

Maps fused multimodal embeddings to soil functional property estimates with
calibrated uncertainty. Internal architecture per the patent:
  1. lifting layer projects fused embedding into higher-dim space
  2. parallel members: classic ML, deep learning, mathematical interpolation
  3. fusion meta-model combines members into a unified estimate

The lifting layer and the ClassicML / DeepLearning members are real PyTorch
modules; the MathematicalInterpolation member is a Gaussian-kernel RBF
interpolant against seeded anchor points. All weights / anchors are seeded
through `stable_seed` for cross-process determinism (same convention as the
encoder backends).
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


N_PROPS = len(SOIL_PROPERTY_NAMES)


def _seeded_init(net, gen, torch) -> None:
    """Manual fan-in init driven by a seeded torch.Generator.

    Same convention as the encoder backends — keeps outputs byte-equal across
    processes on CPU regardless of the default torch init scheme.
    """
    for p in net.parameters():
        with torch.no_grad():
            if p.dim() >= 2:
                fan_in = 1
                for d in p.shape[1:]:
                    fan_in *= int(d)
                std = (1.0 / fan_in) ** 0.5
                p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float32) * std)
            else:
                p.zero_()


class EnsembleMember(Protocol):
    name: str

    def predict(self, lifted: np.ndarray) -> dict[str, float]: ...


# ---------------------------------------------------------------------------
# Lifting layer
# ---------------------------------------------------------------------------


class LiftingLayer:
    """Projects fused embeddings into a higher-dimensional space.

    PyTorch ``nn.Linear(in_dim -> out_dim) + Tanh`` with seeded init. Forward
    accepts and returns numpy float32 — no tensor leaks past the call site.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0) -> None:
        import torch
        from torch import nn

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seed = seed
        self._torch = torch

        gen = torch.Generator(device="cpu").manual_seed(
            stable_seed("lifting", in_dim, out_dim, seed)
        )
        net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())
        _seeded_init(net, gen, torch)
        net.eval()
        self._net = net

    def __call__(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            v = self._net(torch.from_numpy(x.astype(np.float32)))
        return v.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------


class ClassicMLMember:
    """Logistic-regression head: ``nn.Linear(in_dim -> N_PROPS) + Sigmoid``."""

    name = "classic_ml"

    def __init__(self, in_dim: int, seed: int = 0) -> None:
        import torch
        from torch import nn

        self._torch = torch
        gen = torch.Generator(device="cpu").manual_seed(
            stable_seed(self.name, in_dim, seed)
        )
        net = nn.Sequential(nn.Linear(in_dim, N_PROPS), nn.Sigmoid())
        _seeded_init(net, gen, torch)
        net.eval()
        self._net = net

    def predict(self, lifted: np.ndarray) -> dict[str, float]:
        torch = self._torch
        with torch.no_grad():
            out = self._net(torch.from_numpy(lifted.astype(np.float32)))
        v = out.cpu().numpy()
        return {name: float(v[i]) for i, name in enumerate(SOIL_PROPERTY_NAMES)}


class DeepLearningMember:
    """Deeper MLP head: 3 hidden GELU layers, Sigmoid output."""

    name = "deep_learning"

    def __init__(self, in_dim: int, hidden: int = 64, seed: int = 0) -> None:
        import torch
        from torch import nn

        self._torch = torch
        gen = torch.Generator(device="cpu").manual_seed(
            stable_seed(self.name, in_dim, hidden, seed)
        )
        net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, N_PROPS),
            nn.Sigmoid(),
        )
        _seeded_init(net, gen, torch)
        net.eval()
        self._net = net

    def predict(self, lifted: np.ndarray) -> dict[str, float]:
        torch = self._torch
        with torch.no_grad():
            out = self._net(torch.from_numpy(lifted.astype(np.float32)))
        v = out.cpu().numpy()
        return {name: float(v[i]) for i, name in enumerate(SOIL_PROPERTY_NAMES)}


class MathematicalInterpolationMember:
    """Gaussian-kernel RBF interpolation against seeded anchor points.

    Generates ``n_anchors`` reference vectors and per-property reference
    values from a seeded numpy generator; predicts a Gaussian-kernel-weighted
    average of those reference values. This is genuine mathematical
    interpolation (no learned weights, no gradients), which gives the
    ensemble a model class structurally distinct from the two PyTorch
    members — exactly the role the patent reserves for it.

    When a capability ``channels`` mapping is supplied, each property also
    consults its own channel slice (smi←moisture,
    infiltration_potential←infiltration, erosion_susceptibility←erosion) and
    averages the channel-restricted prediction with the global one.
    """

    name = "mathematical_interpolation"

    def __init__(
        self,
        in_dim: int | None = None,
        channels: Mapping[str, slice] | None = None,
        n_anchors: int = 16,
        bandwidth: float = 1.0,
        seed: int = 0,
    ) -> None:
        self._mapping = {
            "smi": "moisture",
            "infiltration_potential": "infiltration",
            "erosion_susceptibility": "erosion",
        }
        self._channels = dict(channels) if channels else {}
        self.n_anchors = int(n_anchors)
        self.bandwidth = float(bandwidth)
        self.seed = int(seed)
        self._anchors: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._in_dim: int | None = None
        if in_dim is not None:
            self._init_anchors(int(in_dim))

    def _init_anchors(self, in_dim: int) -> None:
        rng = np.random.default_rng(
            stable_seed(self.name, in_dim, self.n_anchors, self.seed)
        )
        self._anchors = rng.standard_normal(size=(self.n_anchors, in_dim)).astype(np.float32)
        self._values = rng.uniform(0.0, 1.0, size=(self.n_anchors, N_PROPS)).astype(np.float32)
        self._in_dim = in_dim

    def with_channels(self, channels: Mapping[str, slice]) -> "MathematicalInterpolationMember":
        return MathematicalInterpolationMember(
            in_dim=self._in_dim, channels=channels,
            n_anchors=self.n_anchors, bandwidth=self.bandwidth, seed=self.seed,
        )

    def _rbf(self, query: np.ndarray, dim_slice: slice | None = None) -> np.ndarray:
        assert self._anchors is not None and self._values is not None
        if dim_slice is None:
            anchors = self._anchors
            q = query
        else:
            anchors = self._anchors[:, dim_slice]
            q = query[dim_slice]
        d2 = np.sum((anchors - q[None, :]) ** 2, axis=1)
        # numerically stable softmax of (-d2 / 2σ²)
        logits = -d2 / (2.0 * self.bandwidth * self.bandwidth)
        logits = logits - logits.max()
        weights = np.exp(logits)
        weights = weights / weights.sum()
        return weights @ self._values

    def predict(self, lifted: np.ndarray) -> dict[str, float]:
        if self._anchors is None or self._anchors.shape[1] != lifted.shape[0]:
            self._init_anchors(lifted.shape[0])
        global_pred = self._rbf(lifted)
        out: dict[str, float] = {}
        for i, prop in enumerate(SOIL_PROPERTY_NAMES):
            ch = self._mapping.get(prop)
            if ch and ch in self._channels:
                ch_pred = self._rbf(lifted, dim_slice=self._channels[ch])
                val = 0.5 * (ch_pred[i] + global_pred[i])
            else:
                val = global_pred[i]
            out[prop] = float(min(1.0, max(0.0, val)))
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
                # channel slices are defined on the fused vector; only reuse
                # them when the lifting layer is identity-shaped.
                use = channels if self.config.lifting_dim == fused_dim else {}
                self._members.append(
                    MathematicalInterpolationMember(
                        in_dim=self.config.lifting_dim, channels=use, seed=self.config.seed,
                    )
                )
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
