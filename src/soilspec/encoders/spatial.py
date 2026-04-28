"""Spatial encoder module.

Maps a preprocessed raster tile to a fixed-dimension latent vector capturing
terrain, context, and vegetation structure. All three backends are real
PyTorch modules:

- ``cnn`` — small 2D Conv stack over patches, average-pooled to a tile vector.
- ``vit`` — patch embedding + ``nn.MultiheadAttention`` self-attention.
- ``autoencoder`` — symmetric Conv encoder; exposes a ``reconstruct`` method
  for the patent's training/back-prop loop.

Sliding-window patch selection with optional context-patch incorporation is
applied uniformly before the encoder. Weights are initialised from a per-
instance ``torch.Generator`` seeded by ``stable_seed(name, latent_dim,
in_channels, seed)`` so outputs are byte-equal across processes on CPU.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .._seed import stable_seed
from ..registry import Registry
from ..types import SpatialEmbedding, assert_finite


class SpatialEncoder(Protocol):
    name: str
    latent_dim: int

    def encode(self, tile_id: str, time: int, raster: np.ndarray) -> SpatialEmbedding: ...


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


def _seeded_init(net, seed_int: int, torch) -> None:
    """Initialise every parameter from a single seeded torch Generator."""
    gen = torch.Generator(device="cpu").manual_seed(seed_int)
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


def _patch_batch(raster: np.ndarray, patch_size: int, stride: int, context_patches: int) -> np.ndarray:
    """Returns patches stacked as (N, B, P, P) float32 contiguous."""
    data = _ensure_3d(raster).astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)
    patches = _sliding_patches(data, patch_size, stride, context_patches)
    if not patches:
        patches = [data[:, :patch_size, :patch_size]]
    return np.ascontiguousarray(np.stack(patches, axis=0), dtype=np.float32)


# ---------------------------------------------------------------------------
# CNN backend
# ---------------------------------------------------------------------------


class SpatialCNNEncoder:
    """Small 2D Conv stack over patches; mean across patches; Linear -> Tanh."""

    name = "cnn"

    def __init__(
        self,
        latent_dim: int = 32,
        patch_size: int = 8,
        stride: int = 4,
        context_patches: int = 0,
        seed: int = 0,
    ) -> None:
        import torch  # lazy
        self._torch = torch
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.stride = stride
        self.context_patches = context_patches
        self.seed = seed
        self._net = None  # lazy-build once in_channels is known
        self._in_channels: int | None = None

    def _build(self, in_channels: int):
        torch = self._torch
        from torch import nn
        net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, self.latent_dim),
            nn.Tanh(),
        )
        _seeded_init(
            net,
            stable_seed(self.name, self.latent_dim, in_channels, self.context_patches, self.seed),
            torch,
        )
        net.eval()
        return net

    def encode(self, tile_id: str, time: int, raster: np.ndarray) -> SpatialEmbedding:
        torch = self._torch
        batch = _patch_batch(raster, self.patch_size, self.stride, self.context_patches)
        in_channels = batch.shape[1]
        if self._net is None or self._in_channels != in_channels:
            self._net = self._build(in_channels)
            self._in_channels = in_channels
        with torch.no_grad():
            feat = self._net(torch.from_numpy(batch))  # (N, latent_dim)
            tile_vec = feat.mean(dim=0)
        v = tile_vec.cpu().numpy().astype(np.float32)
        assert_finite(v, "spatial.cnn.embedding")
        return SpatialEmbedding(
            tile_id=tile_id, time=time, vector=v,
            backend=self.name, patch_size=self.patch_size,
        )


# ---------------------------------------------------------------------------
# ViT backend
# ---------------------------------------------------------------------------


class _ViTBlock:
    """Holds the patch-embedding + MHA + head as separate sub-modules so we
    can drive them through ``encode`` without baking a fixed seq length in."""

    def __init__(self, torch, in_channels: int, patch_size: int, latent_dim: int, seed_int: int):
        from torch import nn
        embed_dim = 32
        num_heads = 4
        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(embed_dim, latent_dim), nn.Tanh())
        # init each sub-module with the same generator so the whole block is reproducible
        _seeded_init(self.patch_embed, seed_int, torch)
        _seeded_init(self.attn, seed_int + 1, torch)
        _seeded_init(self.head, seed_int + 2, torch)
        self.patch_embed.eval()
        self.attn.eval()
        self.head.eval()

    def __call__(self, x):
        # x: (N, B, P, P) -> tokens: (N, B*P*P)
        n = x.shape[0]
        tokens = self.patch_embed(x.reshape(n, -1))  # (N, embed_dim)
        seq = tokens.unsqueeze(0)  # (1, N, embed_dim)
        attended, _ = self.attn(seq, seq, seq, need_weights=False)
        pooled = attended.mean(dim=1).squeeze(0)  # (embed_dim,)
        return self.head(pooled)


class SpatialViTEncoder:
    """Patch embedding + multi-head self-attention pooling."""

    name = "vit"

    def __init__(
        self,
        latent_dim: int = 32,
        patch_size: int = 8,
        stride: int = 4,
        context_patches: int = 0,
        seed: int = 0,
    ) -> None:
        import torch  # lazy
        self._torch = torch
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.stride = stride
        self.context_patches = context_patches
        self.seed = seed
        self._block: _ViTBlock | None = None
        self._in_channels: int | None = None

    def encode(self, tile_id: str, time: int, raster: np.ndarray) -> SpatialEmbedding:
        torch = self._torch
        batch = _patch_batch(raster, self.patch_size, self.stride, self.context_patches)
        in_channels = batch.shape[1]
        if self._block is None or self._in_channels != in_channels:
            seed_int = stable_seed(self.name, self.latent_dim, in_channels, self.context_patches, self.seed)
            self._block = _ViTBlock(torch, in_channels, self.patch_size, self.latent_dim, seed_int)
            self._in_channels = in_channels
        with torch.no_grad():
            v_t = self._block(torch.from_numpy(batch))
        v = v_t.cpu().numpy().astype(np.float32)
        assert_finite(v, "spatial.vit.embedding")
        return SpatialEmbedding(
            tile_id=tile_id, time=time, vector=v,
            backend=self.name, patch_size=self.patch_size,
        )


# ---------------------------------------------------------------------------
# Autoencoder backend
# ---------------------------------------------------------------------------


class SpatialAutoencoderEncoder:
    """Symmetric Conv2d encoder; exposes ``reconstruct`` via ConvTranspose2d."""

    name = "autoencoder"

    def __init__(
        self,
        latent_dim: int = 32,
        patch_size: int = 8,
        stride: int = 4,
        context_patches: int = 0,
        seed: int = 0,
    ) -> None:
        import torch  # lazy
        self._torch = torch
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.stride = stride
        self.context_patches = context_patches
        self.seed = seed
        self._enc = None
        self._dec = None
        self._in_channels: int | None = None

    def _build(self, in_channels: int):
        torch = self._torch
        from torch import nn
        enc = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, self.latent_dim),
            nn.Tanh(),
        )
        dec = nn.Sequential(
            nn.Linear(self.latent_dim, 32 * self.patch_size * self.patch_size),
            nn.ReLU(),
            nn.Unflatten(1, (32, self.patch_size, self.patch_size)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, kernel_size=3, padding=1),
        )
        seed_int = stable_seed(self.name, self.latent_dim, in_channels, self.context_patches, self.seed)
        _seeded_init(enc, seed_int, torch)
        _seeded_init(dec, seed_int + 1, torch)
        enc.eval()
        dec.eval()
        return enc, dec

    def encode(self, tile_id: str, time: int, raster: np.ndarray) -> SpatialEmbedding:
        torch = self._torch
        batch = _patch_batch(raster, self.patch_size, self.stride, self.context_patches)
        in_channels = batch.shape[1]
        if self._enc is None or self._in_channels != in_channels:
            self._enc, self._dec = self._build(in_channels)
            self._in_channels = in_channels
        with torch.no_grad():
            feat = self._enc(torch.from_numpy(batch))  # (N, latent_dim)
            tile_vec = feat.mean(dim=0)
        v = tile_vec.cpu().numpy().astype(np.float32)
        assert_finite(v, "spatial.autoencoder.embedding")
        return SpatialEmbedding(
            tile_id=tile_id, time=time, vector=v,
            backend=self.name, patch_size=self.patch_size,
        )

    def reconstruct(self, embedding: SpatialEmbedding, n_channels: int, hw: int) -> np.ndarray:  # pragma: no cover - aux
        torch = self._torch
        if self._dec is None or self._in_channels != n_channels:
            self._enc, self._dec = self._build(n_channels)
            self._in_channels = n_channels
        with torch.no_grad():
            v = torch.from_numpy(embedding.vector).unsqueeze(0)
            out = self._dec(v).squeeze(0)
        # If the decoder's spatial size does not match hw, resample with adaptive pool.
        if out.shape[-1] != hw or out.shape[-2] != hw:
            with torch.no_grad():
                out = torch.nn.functional.adaptive_avg_pool2d(out.unsqueeze(0), (hw, hw)).squeeze(0)
        return out.cpu().numpy().astype(np.float32)


SpatialEncoderRegistry: Registry[SpatialEncoder] = Registry("spatial-encoders")
SpatialEncoderRegistry.register("cnn", lambda **kw: SpatialCNNEncoder(**kw))
SpatialEncoderRegistry.register("vit", lambda **kw: SpatialViTEncoder(**kw))
SpatialEncoderRegistry.register("autoencoder", lambda **kw: SpatialAutoencoderEncoder(**kw))
