"""Spectral encoder module.

Maps a per-tile bandwise spectral input to a fixed-dimension latent vector.
Pluggable backends: 1D-CNN-shaped, transformer-shaped, autoencoder, and a
statistical feature extractor — all selected through `SpectralEncoderRegistry`.

The deep-learning backends are implemented as deterministic numpy projections
seeded by `(backend_name, latent_dim, seed)`. The patent does not require a
specific framework; the contract is `encode(spectral_input) -> SpectralEmbedding`
with a stable shape.
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
    """Backend stamped as `1d_cnn`. Aggregates per-pixel spectra via mean+std then projects."""

    name = "1d_cnn"

    def __init__(self, latent_dim: int = 32, seed: int = 0) -> None:
        self.latent_dim = latent_dim
        self._proj = _DeterministicProjection(self.name, latent_dim, seed)

    def encode(self, tile_id: str, time: int, spectral: np.ndarray) -> SpectralEmbedding:
        data, valid = band_quality_filter(spectral, min_valid=1)
        data = impute_missing_bands(data, valid)
        per_band_mean = data.reshape(data.shape[0], -1).mean(axis=1)
        per_band_std = data.reshape(data.shape[0], -1).std(axis=1)
        feature = np.concatenate([per_band_mean, per_band_std])
        # tanh nonlinearity on the projection — bounded output for tests
        v = np.tanh(self._proj.project(feature))
        assert_finite(v, "spectral.1d_cnn.embedding")
        return SpectralEmbedding(
            tile_id=tile_id, time=time, vector=v.astype(np.float32),
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
