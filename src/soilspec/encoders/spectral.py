"""Spectral encoder module.

Maps a per-tile bandwise spectral input to a fixed-dimension latent vector.
Pluggable backends selected through `SpectralEncoderRegistry`:

- ``1d_cnn`` — real PyTorch 1D CNN over per-pixel spectra. Requires the
  ``torch`` extra; weights are seeded for cross-process determinism.
- ``transformer``, ``autoencoder``, ``statistical`` — numpy-only deterministic
  projections retained for fast tests and as fallbacks.

The contract is `encode(spectral_input) -> SpectralEmbedding` with a stable shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .._seed import stable_seed
from ..registry import Registry
from ..types import SpectralEmbedding, assert_finite


# ---------------------------------------------------------------------------
# Band-quality / imputation utilities
# ---------------------------------------------------------------------------


def band_quality_filter(spectral: np.ndarray, min_valid: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Identify which bands have any finite content. Returns (data, valid_mask)."""
    if spectral.ndim != 3:
        raise ValueError(f"spectral input must be (B, H, W); got {spectral.shape}")
    valid_mask = np.array([np.any(np.isfinite(b)) for b in spectral])
    if int(valid_mask.sum()) < min_valid:
        raise ValueError(f"too few valid bands: {int(valid_mask.sum())} < {min_valid}")
    return spectral, valid_mask


def impute_missing_bands(spectral: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Replace fully-missing bands with the per-pixel mean of the valid bands."""
    out = spectral.astype(np.float32, copy=True)
    for i, ok in enumerate(valid_mask):
        if not ok:
            mean_band = np.nanmean(out[valid_mask], axis=0)
            mean_band = np.where(np.isfinite(mean_band), mean_band, 0.0)
            out[i] = mean_band
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# ---------------------------------------------------------------------------
# Encoder backends
# ---------------------------------------------------------------------------


class SpectralEncoder(Protocol):
    name: str
    latent_dim: int

    def encode(self, tile_id: str, time: int, spectral: np.ndarray) -> SpectralEmbedding: ...


@dataclass
class _DeterministicProjection:
    """Shared utility: a seeded random projection from B*H*W -> latent_dim."""

    backend: str
    latent_dim: int
    seed: int

    def project(self, x: np.ndarray) -> np.ndarray:
        flat = x.reshape(-1)
        rng = np.random.default_rng(stable_seed(self.backend, self.latent_dim, self.seed))
        # generate weight matrix lazily; size limited by flat length
        w = rng.standard_normal(size=(self.latent_dim, flat.shape[0])).astype(np.float32) / np.sqrt(flat.shape[0])
        return w @ flat.astype(np.float32)


class SpectralCNN1DEncoder:
    """PyTorch 1D CNN over per-pixel spectra.

    Per-pixel spectrum (length B, single channel) flows through two Conv1d
    layers with ReLU, an adaptive average pool collapses the spectral axis,
    spatial mean across H*W produces a tile vector, and a final Linear+Tanh
    maps to ``latent_dim``. All parameters are initialised from a torch
    Generator seeded by ``stable_seed("1d_cnn", latent_dim, seed)``, which
    keeps outputs byte-equal across processes on CPU.
    """

    name = "1d_cnn"

    def __init__(self, latent_dim: int = 32, seed: int = 0) -> None:
        import torch  # lazy: only the 1d_cnn backend pulls torch in
        from torch import nn

        self.latent_dim = latent_dim
        self.seed = seed
        self._torch = torch

        gen = torch.Generator(device="cpu").manual_seed(
            stable_seed(self.name, latent_dim, seed)
        )
        net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, latent_dim),
            nn.Tanh(),
        )
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
        net.eval()
        self._net = net

    def encode(self, tile_id: str, time: int, spectral: np.ndarray) -> SpectralEmbedding:
        torch = self._torch
        data, valid = band_quality_filter(spectral, min_valid=1)
        data = impute_missing_bands(data, valid)  # (B, H, W), float32, finite
        b, h, w = data.shape
        # Each pixel is a 1D signal of length B with 1 channel: (HW, 1, B).
        x = np.ascontiguousarray(
            data.transpose(1, 2, 0).reshape(h * w, 1, b), dtype=np.float32
        )
        with torch.no_grad():
            feat = self._net(torch.from_numpy(x))  # (HW, latent_dim)
            tile_vec = feat.mean(dim=0)  # (latent_dim,)
        v = tile_vec.cpu().numpy().astype(np.float32)
        assert_finite(v, "spectral.1d_cnn.embedding")
        return SpectralEmbedding(
            tile_id=tile_id, time=time, vector=v,
            backend=self.name, valid_bands=int(valid.sum()),
        )


class SpectralTransformerEncoder:
    """Backend stamped as `transformer`. Token-pools per-band statistics, then projects."""

    name = "transformer"

    def __init__(self, latent_dim: int = 32, seed: int = 0) -> None:
        self.latent_dim = latent_dim
        self._proj = _DeterministicProjection(self.name, latent_dim, seed)

    def encode(self, tile_id: str, time: int, spectral: np.ndarray) -> SpectralEmbedding:
        data, valid = band_quality_filter(spectral, min_valid=1)
        data = impute_missing_bands(data, valid)
        # cls-token-style aggregation: weighted mean across bands by softmax scores
        scores = np.linalg.norm(data.reshape(data.shape[0], -1), axis=1)
        scores = scores - scores.max()
        weights = np.exp(scores)
        weights = weights / weights.sum()
        token = (data.reshape(data.shape[0], -1) * weights[:, None]).sum(axis=0)
        v = np.tanh(self._proj.project(token))
        assert_finite(v, "spectral.transformer.embedding")
        return SpectralEmbedding(
            tile_id=tile_id, time=time, vector=v.astype(np.float32),
            backend=self.name, valid_bands=int(valid.sum()),
        )


class SpectralAutoencoderEncoder:
    """Backend stamped as `autoencoder`. Encodes, exposes a reconstruction head used in training."""

    name = "autoencoder"

    def __init__(self, latent_dim: int = 32, seed: int = 0) -> None:
        self.latent_dim = latent_dim
        self._proj = _DeterministicProjection(self.name, latent_dim, seed)

    def encode(self, tile_id: str, time: int, spectral: np.ndarray) -> SpectralEmbedding:
        data, valid = band_quality_filter(spectral, min_valid=1)
        data = impute_missing_bands(data, valid)
        flat = data.mean(axis=(1, 2))  # per-band channel
        v = np.tanh(self._proj.project(flat))
        assert_finite(v, "spectral.autoencoder.embedding")
        return SpectralEmbedding(
            tile_id=tile_id, time=time, vector=v.astype(np.float32),
            backend=self.name, valid_bands=int(valid.sum()),
        )

    def reconstruct(self, embedding: SpectralEmbedding, n_bands: int) -> np.ndarray:
        """Tiny decoder; used by the training/back-prop loop in the patent claim."""
        rng = np.random.default_rng(stable_seed(self.name, "decode", n_bands))
        w = rng.standard_normal(size=(n_bands, embedding.vector.shape[0])).astype(np.float32) / np.sqrt(embedding.vector.shape[0])
        return w @ embedding.vector


class SpectralStatisticalEncoder:
    """Backend stamped as `statistical`. Pure-python summary statistics; no projection."""

    name = "statistical"

    def __init__(self, latent_dim: int = 32, seed: int = 0) -> None:
        self.latent_dim = latent_dim
        self._seed = seed

    def encode(self, tile_id: str, time: int, spectral: np.ndarray) -> SpectralEmbedding:
        data, valid = band_quality_filter(spectral, min_valid=1)
        data = impute_missing_bands(data, valid)
        flat = data.reshape(data.shape[0], -1)
        stats = np.concatenate([
            flat.mean(axis=1), flat.std(axis=1), flat.min(axis=1), flat.max(axis=1),
        ])
        v = np.zeros(self.latent_dim, dtype=np.float32)
        v[: min(self.latent_dim, stats.shape[0])] = stats[: self.latent_dim]
        assert_finite(v, "spectral.statistical.embedding")
        return SpectralEmbedding(
            tile_id=tile_id, time=time, vector=v,
            backend=self.name, valid_bands=int(valid.sum()),
        )


SpectralEncoderRegistry: Registry[SpectralEncoder] = Registry("spectral-encoders")
SpectralEncoderRegistry.register("1d_cnn", lambda **kw: SpectralCNN1DEncoder(**kw))
SpectralEncoderRegistry.register("transformer", lambda **kw: SpectralTransformerEncoder(**kw))
SpectralEncoderRegistry.register("autoencoder", lambda **kw: SpectralAutoencoderEncoder(**kw))
SpectralEncoderRegistry.register("statistical", lambda **kw: SpectralStatisticalEncoder(**kw))
