"""Multimodal fusion engine.

Combines spectral and spatial embeddings into a unified `FusedRepresentation`
organized into named capability channels (moisture-relevant,
infiltration-relevant, erosion-relevant, etc.). Strategy is selected by name
through `FusionStrategyRegistry`.

Degraded fusion: if only one modality is available the engine falls back to
single-modality pass-through and stamps the output with `degraded=True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from ._seed import stable_seed
from .registry import Registry
from .types import FusedRepresentation, SpatialEmbedding, SpectralEmbedding


DEFAULT_CHANNELS: tuple[tuple[str, int], ...] = (
    ("moisture", 16),
    ("infiltration", 16),
    ("erosion", 16),
)


class FusionStrategy(Protocol):
    name: str

    def fuse(self, spectral: np.ndarray, spatial: np.ndarray) -> np.ndarray: ...


class ConcatFusion:
    name = "concat"

    def __init__(self, output_dim: int) -> None:
        self.output_dim = output_dim

    def fuse(self, spectral: np.ndarray, spatial: np.ndarray) -> np.ndarray:
        return _project(np.concatenate([spectral, spatial]), self.output_dim, self.name)


class AttentionFusion:
    name = "attention"

    def __init__(self, output_dim: int) -> None:
        self.output_dim = output_dim

    def fuse(self, spectral: np.ndarray, spatial: np.ndarray) -> np.ndarray:
        # softmax-weighted sum across the two modality vectors after dim alignment
        a = _project(spectral, self.output_dim, self.name + "/spec")
        b = _project(spatial, self.output_dim, self.name + "/spat")
        scores = np.array([np.linalg.norm(a), np.linalg.norm(b)])
        scores = scores - scores.max()
        w = np.exp(scores)
        w = w / w.sum()
        return (w[0] * a + w[1] * b).astype(np.float32)


class GatingFusion:
    name = "gating"

    def __init__(self, output_dim: int) -> None:
        self.output_dim = output_dim

    def fuse(self, spectral: np.ndarray, spatial: np.ndarray) -> np.ndarray:
        a = _project(spectral, self.output_dim, self.name + "/spec")
        b = _project(spatial, self.output_dim, self.name + "/spat")
        gate = _sigmoid(_project(np.concatenate([spectral, spatial]), self.output_dim, self.name + "/gate"))
        return (gate * a + (1.0 - gate) * b).astype(np.float32)


class DeepFusion:
    name = "deep"

    def __init__(self, output_dim: int) -> None:
        self.output_dim = output_dim

    def fuse(self, spectral: np.ndarray, spatial: np.ndarray) -> np.ndarray:
        h1 = np.tanh(_project(np.concatenate([spectral, spatial]), self.output_dim * 2, self.name + "/h1"))
        h2 = np.tanh(_project(h1, self.output_dim, self.name + "/h2"))
        return h2.astype(np.float32)


FusionStrategyRegistry: Registry[FusionStrategy] = Registry("fusion-strategies")
FusionStrategyRegistry.register("concat", lambda output_dim: ConcatFusion(output_dim))
FusionStrategyRegistry.register("attention", lambda output_dim: AttentionFusion(output_dim))
FusionStrategyRegistry.register("gating", lambda output_dim: GatingFusion(output_dim))
FusionStrategyRegistry.register("deep", lambda output_dim: DeepFusion(output_dim))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class FusionConfig:
    strategy: str = "concat"
    channels: tuple[tuple[str, int], ...] = DEFAULT_CHANNELS

    @property
    def output_dim(self) -> int:
        return sum(d for _, d in self.channels)


class FusionEngine:
    """Runs a configured fusion strategy and partitions output into channels."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self._strategy = FusionStrategyRegistry.create(
            self.config.strategy, output_dim=self.config.output_dim
        )

    def fuse(
        self,
        spectral: SpectralEmbedding | None,
        spatial: SpatialEmbedding | None,
    ) -> FusedRepresentation:
        missing: list[str] = []
        if spectral is None:
            missing.append("spectral")
        if spatial is None:
            missing.append("spatial")
        if spectral is None and spatial is None:
            raise ValueError("at least one modality must be present for fusion")

        degraded = bool(missing)

        # In degraded mode, pass through the available modality (projected to
        # the output dim) instead of running the full fusion strategy.
        if spectral is None or spatial is None:
            present = spectral if spectral is not None else spatial
            assert present is not None
            vec = _project(present.vector, self.config.output_dim, "degraded")
            tile_id, time = present.tile_id, present.time
        else:
            if spectral.tile_id != spatial.tile_id or spectral.time != spatial.time:
                raise ValueError(
                    f"spectral/spatial keys mismatch: ({spectral.tile_id},{spectral.time}) "
                    f"vs ({spatial.tile_id},{spatial.time})"
                )
            vec = self._strategy.fuse(spectral.vector, spatial.vector)
            tile_id, time = spectral.tile_id, spectral.time

        slices = _channel_slices(self.config.channels)
        return FusedRepresentation(
            tile_id=tile_id,
            time=time,
            vector=vec.astype(np.float32),
            channels=slices,
            strategy=self.config.strategy,
            degraded=degraded,
            missing_modalities=tuple(missing),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _channel_slices(channels: tuple[tuple[str, int], ...]) -> dict[str, slice]:
    out: dict[str, slice] = {}
    cur = 0
    for name, dim in channels:
        out[name] = slice(cur, cur + dim)
        cur += dim
    return out


def _project(x: np.ndarray, out_dim: int, tag: str) -> np.ndarray:
    rng = np.random.default_rng(stable_seed("fusion", tag, out_dim, x.shape[0]))
    w = rng.standard_normal(size=(out_dim, x.shape[0])).astype(np.float32) / np.sqrt(x.shape[0])
    return w @ x.astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
