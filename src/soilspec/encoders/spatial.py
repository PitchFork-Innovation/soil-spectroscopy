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

from typing import Iterator, Protocol

import numpy as np

from .._seed import stable_seed
from ..registry import Registry
from ..types import SpatialEmbedding, assert_finite


class SpatialEncoder(Protocol):
    name: str
    latent_dim: int

    def encode(self, tile_id: str, time: int, raster: np.ndarray) -> SpatialEmbedding: ...


# ---------------------------------------------------------------------------
# Trainable patch extraction (tensor-native, batch-friendly)
# ---------------------------------------------------------------------------


def _extract_patches_tensor(
    raster_t, patch_size: int, stride: int, context_patches: int
):
    """Tensor equivalent of :func:`_sliding_patches` + edge-pad.

    Accepts a raster tensor of shape ``(B, H, W)`` or ``(N, B, H, W)`` and
    returns patches stacked as ``(N * n_patches, B, P, P)`` where ``N`` is
    the batch size (added if missing). Gradient-friendly throughout.
    """
    import torch
    import torch.nn.functional as F

    if raster_t.dim() == 3:
        raster_t = raster_t.unsqueeze(0)  # (1, B, H, W)
    if raster_t.dim() != 4:
        raise ValueError(
            f"raster tensor must be (B,H,W) or (N,B,H,W); got {tuple(raster_t.shape)}"
        )
    pad = context_patches * patch_size
    if pad > 0:
        # 'replicate' matches numpy's mode='edge'.
        raster_t = F.pad(raster_t, (pad, pad, pad, pad), mode="replicate")
    n, b, h, w = raster_t.shape
    if h < patch_size or w < patch_size:
        # Fallback: treat the whole raster as a single patch sized to (P, P)
        # via adaptive average pool — matches the numpy edge case at
        # _patch_batch line 79 ("if not patches").
        pooled = F.adaptive_avg_pool2d(raster_t, (patch_size, patch_size))
        return pooled.view(n, 1, b, patch_size, patch_size).reshape(
            n * 1, b, patch_size, patch_size
        ), n, 1
    # unfold extracts sliding patches: result shape (N, B, n_h, n_w, P, P)
    patches = raster_t.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    n_h, n_w = int(patches.shape[2]), int(patches.shape[3])
    # rearrange to (N, n_h*n_w, B, P, P) -> (N*n_patches, B, P, P)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(n * n_h * n_w, b, patch_size, patch_size)
    return patches, n, n_h * n_w


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

    # -------------------- trainable interface ----------------------------

    def build(self, in_channels: int) -> None:
        """Force lazy construction so the trainer can collect parameters."""
        if self._net is None or self._in_channels != in_channels:
            self._net = self._build(in_channels)
            self._in_channels = in_channels

    def forward_torch(self, raster_t):
        """Tensor-native forward. Accepts ``(B,H,W)`` or batched ``(N,B,H,W)``;
        returns ``(N, latent_dim)`` (or ``(latent_dim,)`` if input was unbatched).
        Gradient-friendly — no ``no_grad`` here, no numpy conversion."""
        if raster_t.dim() == 3:
            squeeze = True
            raster_t = raster_t.unsqueeze(0)
        else:
            squeeze = False
        in_channels = int(raster_t.shape[1])
        self.build(in_channels)
        patches, n, n_patches = _extract_patches_tensor(
            raster_t, self.patch_size, self.stride, self.context_patches,
        )
        feat = self._net(patches)  # (N*n_patches, latent_dim)
        tile_vec = feat.view(n, n_patches, -1).mean(dim=1)  # (N, latent_dim)
        return tile_vec.squeeze(0) if squeeze else tile_vec

    def parameters(self) -> Iterator:
        return self._net.parameters() if self._net is not None else iter(())

    def state_dict(self) -> dict:
        if self._net is None:
            return {}
        return {"net": self._net.state_dict(), "in_channels": int(self._in_channels or 0)}

    def load_state_dict(self, sd: dict) -> None:
        if not sd:
            return
        in_channels = int(sd.get("in_channels", 0))
        if in_channels > 0:
            self.build(in_channels)
        if self._net is not None and "net" in sd:
            self._net.load_state_dict(sd["net"])


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
        self.build(in_channels)
        with torch.no_grad():
            v_t = self._block(torch.from_numpy(batch))
        v = v_t.cpu().numpy().astype(np.float32)
        assert_finite(v, "spatial.vit.embedding")
        return SpatialEmbedding(
            tile_id=tile_id, time=time, vector=v,
            backend=self.name, patch_size=self.patch_size,
        )

    # -------------------- trainable interface ----------------------------

    def build(self, in_channels: int) -> None:
        if self._block is None or self._in_channels != in_channels:
            seed_int = stable_seed(
                self.name, self.latent_dim, in_channels, self.context_patches, self.seed,
            )
            self._block = _ViTBlock(
                self._torch, in_channels, self.patch_size, self.latent_dim, seed_int,
            )
            self._in_channels = in_channels

    def forward_torch(self, raster_t):
        if raster_t.dim() == 3:
            squeeze = True
            raster_t = raster_t.unsqueeze(0)
        else:
            squeeze = False
        in_channels = int(raster_t.shape[1])
        self.build(in_channels)
        patches, n, n_patches = _extract_patches_tensor(
            raster_t, self.patch_size, self.stride, self.context_patches,
        )
        # _ViTBlock processes a single example's patches; loop in Python for
        # batches. Batches ≤ 32 — the loop overhead is dwarfed by attention.
        outs = []
        for i in range(n):
            chunk = patches[i * n_patches : (i + 1) * n_patches]
            outs.append(self._block(chunk))
        v = self._torch.stack(outs, dim=0)  # (N, latent_dim)
        return v.squeeze(0) if squeeze else v

    def parameters(self) -> Iterator:
        if self._block is None:
            return iter(())
        from itertools import chain
        return chain(
            self._block.patch_embed.parameters(),
            self._block.attn.parameters(),
            self._block.head.parameters(),
        )

    def state_dict(self) -> dict:
        if self._block is None:
            return {}
        return {
            "patch_embed": self._block.patch_embed.state_dict(),
            "attn": self._block.attn.state_dict(),
            "head": self._block.head.state_dict(),
            "in_channels": int(self._in_channels or 0),
        }

    def load_state_dict(self, sd: dict) -> None:
        if not sd:
            return
        in_channels = int(sd.get("in_channels", 0))
        if in_channels > 0:
            self.build(in_channels)
        if self._block is None:
            return
        if "patch_embed" in sd:
            self._block.patch_embed.load_state_dict(sd["patch_embed"])
        if "attn" in sd:
            self._block.attn.load_state_dict(sd["attn"])
        if "head" in sd:
            self._block.head.load_state_dict(sd["head"])


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
        self.build(in_channels)
        with torch.no_grad():
            feat = self._enc(torch.from_numpy(batch))  # (N, latent_dim)
            tile_vec = feat.mean(dim=0)
        v = tile_vec.cpu().numpy().astype(np.float32)
        assert_finite(v, "spatial.autoencoder.embedding")
        return SpatialEmbedding(
            tile_id=tile_id, time=time, vector=v,
            backend=self.name, patch_size=self.patch_size,
        )

    # -------------------- trainable interface ----------------------------

    def build(self, in_channels: int) -> None:
        if self._enc is None or self._in_channels != in_channels:
            self._enc, self._dec = self._build(in_channels)
            self._in_channels = in_channels

    def forward_torch(self, raster_t):
        if raster_t.dim() == 3:
            squeeze = True
            raster_t = raster_t.unsqueeze(0)
        else:
            squeeze = False
        in_channels = int(raster_t.shape[1])
        self.build(in_channels)
        patches, n, n_patches = _extract_patches_tensor(
            raster_t, self.patch_size, self.stride, self.context_patches,
        )
        feat = self._enc(patches)
        tile_vec = feat.view(n, n_patches, -1).mean(dim=1)
        return tile_vec.squeeze(0) if squeeze else tile_vec

    def parameters(self) -> Iterator:
        if self._enc is None:
            return iter(())
        from itertools import chain
        # Decoder is part of the trainable surface only if the user wants
        # auxiliary reconstruction loss; for the supervised path we only
        # need the encoder. Expose both — `train_pipeline` collects encoder
        # params; SSL pretraining (Phase B) would also collect decoder.
        return chain(self._enc.parameters(), self._dec.parameters())

    def state_dict(self) -> dict:
        if self._enc is None:
            return {}
        return {
            "enc": self._enc.state_dict(),
            "dec": self._dec.state_dict(),
            "in_channels": int(self._in_channels or 0),
        }

    def load_state_dict(self, sd: dict) -> None:
        if not sd:
            return
        in_channels = int(sd.get("in_channels", 0))
        if in_channels > 0:
            self.build(in_channels)
        if self._enc is not None and "enc" in sd:
            self._enc.load_state_dict(sd["enc"])
        if self._dec is not None and "dec" in sd:
            self._dec.load_state_dict(sd["dec"])

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
