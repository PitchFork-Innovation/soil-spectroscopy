"""Multimodal fusion engine.

Combines spectral and spatial embeddings into a unified `FusedRepresentation`
organized into named capability channels (moisture-relevant,
infiltration-relevant, erosion-relevant, etc.). Strategy is selected by name
through `FusionStrategyRegistry`.

Strategies are real PyTorch modules built lazily on first ``fuse()`` (input
dims aren't known until then). Weights are seeded by ``stable_seed`` for
cross-process determinism, matching the encoder/inference convention.

Degraded fusion: if only one modality is available the engine falls back to
a single-modality projection and stamps the output with ``degraded=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ._seed import stable_seed
from .registry import Registry
from .types import FusedRepresentation, SpatialEmbedding, SpectralEmbedding


DEFAULT_CHANNELS: tuple[tuple[str, int], ...] = (
    ("moisture", 16),
    ("infiltration", 16),
    ("erosion", 16),
)


def _seeded_init(net, gen, torch) -> None:
    """Manual fan-in init driven by a seeded torch.Generator."""
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


class FusionStrategy(Protocol):
    name: str

    def fuse(self, spectral: np.ndarray, spatial: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class _LazyTorchStrategy:
    """Common base: holds output_dim, lazy-builds the torch module on first call.

    Input dims (spectral, spatial) aren't known at construction — the registry
    only passes ``output_dim``. We rebuild if we ever see different shapes
    (cheap and safe; in practice the same engine is fed consistent inputs).
    """

    name: str = ""

    def __init__(self, output_dim: int) -> None:
        import torch  # lazy: only torch backends pull torch in
        self._torch = torch
        self.output_dim = int(output_dim)
        self._built_for: tuple[int, int] | None = None

    def _ensure_built(self, spec_dim: int, spat_dim: int) -> None:
        if self._built_for == (spec_dim, spat_dim):
            return
        self._build(spec_dim, spat_dim)
        self._built_for = (spec_dim, spat_dim)

    def _build(self, spec_dim: int, spat_dim: int) -> None:  # pragma: no cover
        raise NotImplementedError


class ConcatFusion(_LazyTorchStrategy):
    """Concatenate then project: ``nn.Linear(spec+spat -> output_dim) + Tanh``."""

    name = "concat"

    def _build(self, spec_dim: int, spat_dim: int) -> None:
        torch = self._torch
        from torch import nn

        gen = torch.Generator(device="cpu").manual_seed(
            stable_seed(self.name, self.output_dim, spec_dim, spat_dim)
        )
        net = nn.Sequential(nn.Linear(spec_dim + spat_dim, self.output_dim), nn.Tanh())
        _seeded_init(net, gen, torch)
        net.eval()
        self._net = net

    def fuse(self, spectral: np.ndarray, spatial: np.ndarray) -> np.ndarray:
        torch = self._torch
        self._ensure_built(spectral.shape[0], spatial.shape[0])
        x = np.concatenate([spectral, spatial]).astype(np.float32)
        with torch.no_grad():
            v = self._net(torch.from_numpy(x))
        return v.cpu().numpy().astype(np.float32)


class AttentionFusion(_LazyTorchStrategy):
    """Cross-modal attention.

    Each modality is projected to a shared token dim, the two tokens flow
    through ``nn.MultiheadAttention``, the attention output is mean-pooled
    across tokens and projected to ``output_dim``.
    """

    name = "attention"
    embed_dim: int = 32
    num_heads: int = 4

    def _build(self, spec_dim: int, spat_dim: int) -> None:
        torch = self._torch
        from torch import nn

        gen = torch.Generator(device="cpu").manual_seed(
            stable_seed(self.name, self.output_dim, spec_dim, spat_dim)
        )
        self._spec_proj = nn.Linear(spec_dim, self.embed_dim)
        self._spat_proj = nn.Linear(spat_dim, self.embed_dim)
        self._attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True,
        )
        self._head = nn.Sequential(nn.Linear(self.embed_dim, self.output_dim), nn.Tanh())
        for sub in (self._spec_proj, self._spat_proj, self._attn, self._head):
            _seeded_init(sub, gen, torch)
            sub.eval()

    def fuse(self, spectral: np.ndarray, spatial: np.ndarray) -> np.ndarray:
        torch = self._torch
        self._ensure_built(spectral.shape[0], spatial.shape[0])
        spec_t = torch.from_numpy(spectral.astype(np.float32))
        spat_t = torch.from_numpy(spatial.astype(np.float32))
        with torch.no_grad():
            tokens = torch.stack(
                [self._spec_proj(spec_t), self._spat_proj(spat_t)], dim=0
            ).unsqueeze(0)  # (1, 2, embed_dim) for batch_first=True
            attn_out, _ = self._attn(tokens, tokens, tokens, need_weights=False)
            pooled = attn_out.mean(dim=1).squeeze(0)  # (embed_dim,)
            v = self._head(pooled)
        return v.cpu().numpy().astype(np.float32)


class GatingFusion(_LazyTorchStrategy):
    """Gated fusion: ``gate * spec_proj + (1 - gate) * spat_proj``."""

    name = "gating"

    def _build(self, spec_dim: int, spat_dim: int) -> None:
        torch = self._torch
        from torch import nn

        gen = torch.Generator(device="cpu").manual_seed(
            stable_seed(self.name, self.output_dim, spec_dim, spat_dim)
        )
        self._spec_proj = nn.Linear(spec_dim, self.output_dim)
        self._spat_proj = nn.Linear(spat_dim, self.output_dim)
        self._gate = nn.Sequential(
            nn.Linear(spec_dim + spat_dim, self.output_dim), nn.Sigmoid()
        )
        for sub in (self._spec_proj, self._spat_proj, self._gate):
            _seeded_init(sub, gen, torch)
            sub.eval()

    def fuse(self, spectral: np.ndarray, spatial: np.ndarray) -> np.ndarray:
        torch = self._torch
        self._ensure_built(spectral.shape[0], spatial.shape[0])
        spec_t = torch.from_numpy(spectral.astype(np.float32))
        spat_t = torch.from_numpy(spatial.astype(np.float32))
        with torch.no_grad():
            a = self._spec_proj(spec_t)
            b = self._spat_proj(spat_t)
            gate = self._gate(torch.cat([spec_t, spat_t], dim=0))
            v = gate * a + (1.0 - gate) * b
        return v.cpu().numpy().astype(np.float32)


class DeepFusion(_LazyTorchStrategy):
    """3-layer MLP fusion: Linear → GELU → Linear → GELU → Linear → Tanh."""

    name = "deep"

    def _build(self, spec_dim: int, spat_dim: int) -> None:
        torch = self._torch
        from torch import nn

        gen = torch.Generator(device="cpu").manual_seed(
            stable_seed(self.name, self.output_dim, spec_dim, spat_dim)
        )
        net = nn.Sequential(
            nn.Linear(spec_dim + spat_dim, self.output_dim * 2),
            nn.GELU(),
            nn.Linear(self.output_dim * 2, self.output_dim * 2),
            nn.GELU(),
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.Tanh(),
        )
        _seeded_init(net, gen, torch)
        net.eval()
        self._net = net

    def fuse(self, spectral: np.ndarray, spatial: np.ndarray) -> np.ndarray:
        torch = self._torch
        self._ensure_built(spectral.shape[0], spatial.shape[0])
        x = np.concatenate([spectral, spatial]).astype(np.float32)
        with torch.no_grad():
            v = self._net(torch.from_numpy(x))
        return v.cpu().numpy().astype(np.float32)


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
        # nn.Linear cache for the degraded-mode passthrough, keyed by input dim
        self._degraded_cache: dict[int, object] = {}

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
            vec = self._degraded_project(present.vector)
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

    def _degraded_project(self, x: np.ndarray) -> np.ndarray:
        import torch
        from torch import nn

        in_dim = int(x.shape[0])
        net = self._degraded_cache.get(in_dim)
        if net is None:
            gen = torch.Generator(device="cpu").manual_seed(
                stable_seed("fusion_degraded", in_dim, self.config.output_dim)
            )
            net = nn.Sequential(nn.Linear(in_dim, self.config.output_dim), nn.Tanh())
            _seeded_init(net, gen, torch)
            net.eval()
            self._degraded_cache[in_dim] = net
        with torch.no_grad():
            v = net(torch.from_numpy(x.astype(np.float32)))
        return v.cpu().numpy().astype(np.float32)


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
