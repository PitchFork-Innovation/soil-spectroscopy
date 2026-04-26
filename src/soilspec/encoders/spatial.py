"""Spatial encoder module.

Maps a preprocessed raster tile to a fixed-dimension latent vector capturing
terrain, context, and vegetation structure. Pluggable backends: CNN-shaped,
ViT-shaped, autoencoder. Sliding-window patch selection with optional
context-patch incorporation for adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .._seed import stable_seed
from ..registry import Registry
from ..types import SpatialEmbedding, assert_finite


class SpatialEncoder(Protocol):
    name: str
    latent_dim: int

    def encode(self, tile_id: str, time: int, raster: np.ndarray) -> SpatialEmbedding: ...


@dataclass
class _DeterministicProjection:
    backend: str
    latent_dim: int
    in_dim: int
    seed: int

    def project(self, x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(stable_seed(self.backend, self.latent_dim, self.in_dim, self.seed))
        w = rng.standard_normal(size=(self.latent_dim, self.in_dim)).astype(np.float32) / np.sqrt(self.in_dim)
        return w @ x.astype(np.float32)


def _ensure_3d(raster: np.ndarray) -> np.ndarray:
    if raster.ndim == 2:
        return raster[None, :, :]
    if raster.ndim == 3:
        return raster
    raise ValueError(f"raster must be (H,W) or (B,H,W); got {raster.shape}")


def _sliding_patches(raster: np.ndarray, patch_size: int, stride: int, context_patches: int) -> list[np.ndarray]:
    b, h, w = raster.shape
    pad = context_patches * patch_size
    if pad > 0:
        padded = np.pad(raster, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    else:
        padded = raster
    out: list[np.ndarray] = []
    h2, w2 = padded.shape[1], padded.shape[2]
    for r in range(0, h2 - patch_size + 1, stride):
        for c in range(0, w2 - patch_size + 1, stride):
            out.append(padded[:, r : r + patch_size, c : c + patch_size])
    return out


class SpatialCNNEncoder:
    """Backend stamped as `cnn`. Aggregates per-patch mean+std then projects."""

    name = "cnn"

    def __init__(self, latent_dim: int = 32, patch_size: int = 8, stride: int = 4, context_patches: int = 0, seed: int = 0) -> None:
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.stride = stride
        self.context_patches = context_patches
        self.seed = seed

    def encode(self, tile_id: str, time: int, raster: np.ndarray) -> SpatialEmbedding:
        data = _ensure_3d(raster).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        patches = _sliding_patches(data, self.patch_size, self.stride, self.context_patches)
        if not patches:
            patches = [data[:, : self.patch_size, : self.patch_size]]
        feats = []
        for p in patches:
            feats.append(p.mean(axis=(1, 2)))
            feats.append(p.std(axis=(1, 2)))
        feature = np.concatenate(feats)
        proj = _DeterministicProjection(self.name, self.latent_dim, feature.shape[0], self.seed)
        v = np.tanh(proj.project(feature))
        assert_finite(v, "spatial.cnn.embedding")
        return SpatialEmbedding(
            tile_id=tile_id, time=time, vector=v.astype(np.float32),
            backend=self.name, patch_size=self.patch_size,
        )


class SpatialViTEncoder:
    """Backend stamped as `vit`. Patch tokens with attention pooling."""

    name = "vit"

    def __init__(self, latent_dim: int = 32, patch_size: int = 8, stride: int = 4, context_patches: int = 0, seed: int = 0) -> None:
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.stride = stride
        self.context_patches = context_patches
        self.seed = seed

    def encode(self, tile_id: str, time: int, raster: np.ndarray) -> SpatialEmbedding:
        data = _ensure_3d(raster).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        patches = _sliding_patches(data, self.patch_size, self.stride, self.context_patches)
        if not patches:
            patches = [data[:, : self.patch_size, : self.patch_size]]
        tokens = np.stack([p.mean(axis=(1, 2)) for p in patches], axis=0)
        scores = np.linalg.norm(tokens, axis=1)
        scores = scores - scores.max()
        weights = np.exp(scores)
        weights = weights / weights.sum()
        token = (tokens * weights[:, None]).sum(axis=0)
        proj = _DeterministicProjection(self.name, self.latent_dim, token.shape[0], self.seed)
        v = np.tanh(proj.project(token))
        assert_finite(v, "spatial.vit.embedding")
        return SpatialEmbedding(
            tile_id=tile_id, time=time, vector=v.astype(np.float32),
            backend=self.name, patch_size=self.patch_size,
        )


class SpatialAutoencoderEncoder:
    """Backend stamped as `autoencoder`. Symmetric to the spectral autoencoder."""

    name = "autoencoder"

    def __init__(self, latent_dim: int = 32, patch_size: int = 8, stride: int = 4, context_patches: int = 0, seed: int = 0) -> None:
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.stride = stride
        self.context_patches = context_patches
        self.seed = seed

    def encode(self, tile_id: str, time: int, raster: np.ndarray) -> SpatialEmbedding:
        data = _ensure_3d(raster).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        flat = data.mean(axis=(1, 2))  # per-channel
        proj = _DeterministicProjection(self.name, self.latent_dim, flat.shape[0], self.seed)
        v = np.tanh(proj.project(flat))
        assert_finite(v, "spatial.autoencoder.embedding")
        return SpatialEmbedding(
            tile_id=tile_id, time=time, vector=v.astype(np.float32),
            backend=self.name, patch_size=self.patch_size,
        )

    def reconstruct(self, embedding: SpatialEmbedding, n_channels: int, hw: int) -> np.ndarray:  # pragma: no cover - aux
        rng = np.random.default_rng(stable_seed(self.name, "decode", n_channels, hw))
        w = rng.standard_normal(size=(n_channels * hw * hw, embedding.vector.shape[0])).astype(np.float32) / np.sqrt(embedding.vector.shape[0])
        return (w @ embedding.vector).reshape(n_channels, hw, hw)


SpatialEncoderRegistry: Registry[SpatialEncoder] = Registry("spatial-encoders")
SpatialEncoderRegistry.register("cnn", lambda **kw: SpatialCNNEncoder(**kw))
SpatialEncoderRegistry.register("vit", lambda **kw: SpatialViTEncoder(**kw))
SpatialEncoderRegistry.register("autoencoder", lambda **kw: SpatialAutoencoderEncoder(**kw))
