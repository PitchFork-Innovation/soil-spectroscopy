"""Spectral encoder module.

Maps a per-tile bandwise spectral input to a fixed-dimension latent vector.
Pluggable backends selected through `SpectralEncoderRegistry`:

- ``1d_cnn`` — real PyTorch 1D CNN over per-pixel spectra.
- ``transformer`` — per-band patch-embed + ``nn.MultiheadAttention`` self-
  attention over band tokens.
- ``autoencoder`` — real torch encoder/decoder pair; ``reconstruct`` runs the
  trained decoder.
- ``statistical`` — pure summary statistics; no learned weights by design.

All torch backends require the ``torch`` extra; weights are seeded for
cross-process determinism.

The contract is `encode(spectral_input) -> SpectralEmbedding` with a stable shape.
"""

from __future__ import annotations

from typing import Iterator, Protocol

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

    # -------------------- trainable interface ----------------------------

    def forward_torch(self, spectral_t):
        """Tensor-native forward over a (B,H,W) or batched (N,B,H,W) input.

        Imputation/quality-filter is gradient-pass-through (NaNs are zeroed).
        """
        if spectral_t.dim() == 3:
            squeeze = True
            spectral_t = spectral_t.unsqueeze(0)
        else:
            squeeze = False
        n, b, h, w = spectral_t.shape
        # Replace non-finite values with zeros so the encoder produces stable
        # output even when bands have NaNs (matches impute_missing_bands).
        spectral_t = self._torch.nan_to_num(spectral_t, nan=0.0, posinf=0.0, neginf=0.0)
        # (N*HW, 1, B) per the numpy path.
        x = spectral_t.permute(0, 2, 3, 1).reshape(n * h * w, 1, b)
        feat = self._net(x)  # (N*HW, latent_dim)
        tile_vec = feat.view(n, h * w, -1).mean(dim=1)
        return tile_vec.squeeze(0) if squeeze else tile_vec

    def build(self, n_bands: int | None = None) -> None:
        # 1d_cnn doesn't depend on n_bands at construction; method exists for
        # API parity with the other encoders.
        return None

    def parameters(self) -> Iterator:
        return self._net.parameters()

    def state_dict(self) -> dict:
        return {"net": self._net.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        if "net" in sd:
            self._net.load_state_dict(sd["net"])


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


class SpectralTransformerEncoder:
    """Per-band token transformer.

    Each band is summarised by 4 spatial statistics (mean, std, min, max),
    embedded via ``nn.Linear`` to ``embed_dim``, then a single
    ``nn.MultiheadAttention`` block aggregates band tokens. Mean-pool over
    output tokens projects to ``latent_dim``.

    The architecture is independent of ``n_bands`` (the patch-embed acts on
    the 4-stat per-band feature), so weights are built eagerly and reused.
    """

    name = "transformer"
    embed_dim: int = 32
    num_heads: int = 4

    def __init__(self, latent_dim: int = 32, seed: int = 0) -> None:
        import torch
        from torch import nn

        self.latent_dim = latent_dim
        self.seed = seed
        self._torch = torch

        gen = torch.Generator(device="cpu").manual_seed(
            stable_seed(self.name, latent_dim, seed)
        )
        self._patch_embed = nn.Linear(4, self.embed_dim)
        self._attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True,
        )
        self._head = nn.Sequential(nn.Linear(self.embed_dim, latent_dim), nn.Tanh())
        for sub in (self._patch_embed, self._attn, self._head):
            _seeded_init(sub, gen, torch)
            sub.eval()

    def encode(self, tile_id: str, time: int, spectral: np.ndarray) -> SpectralEmbedding:
        torch = self._torch
        data, valid = band_quality_filter(spectral, min_valid=1)
        data = impute_missing_bands(data, valid)
        flat = data.reshape(data.shape[0], -1)
        feats = np.stack(
            [flat.mean(axis=1), flat.std(axis=1), flat.min(axis=1), flat.max(axis=1)],
            axis=1,
        ).astype(np.float32)  # (B, 4)
        with torch.no_grad():
            tokens = self._patch_embed(torch.from_numpy(feats)).unsqueeze(0)  # (1, B, embed)
            attn_out, _ = self._attn(tokens, tokens, tokens, need_weights=False)
            pooled = attn_out.mean(dim=1).squeeze(0)  # (embed,)
            v = self._head(pooled)
        out = v.cpu().numpy().astype(np.float32)
        assert_finite(out, "spectral.transformer.embedding")
        return SpectralEmbedding(
            tile_id=tile_id, time=time, vector=out,
            backend=self.name, valid_bands=int(valid.sum()),
        )

    # -------------------- trainable interface ----------------------------

    def forward_torch(self, spectral_t):
        if spectral_t.dim() == 3:
            squeeze = True
            spectral_t = spectral_t.unsqueeze(0)
        else:
            squeeze = False
        spectral_t = self._torch.nan_to_num(spectral_t, nan=0.0, posinf=0.0, neginf=0.0)
        n, b, h, w = spectral_t.shape
        flat = spectral_t.view(n, b, h * w)
        # Per-band stats: mean/std/min/max over spatial dim.
        f_mean = flat.mean(dim=2)
        f_std = flat.std(dim=2)
        f_min = flat.min(dim=2).values
        f_max = flat.max(dim=2).values
        feats = self._torch.stack([f_mean, f_std, f_min, f_max], dim=2)  # (N, B, 4)
        tokens = self._patch_embed(feats)  # (N, B, embed)
        attn_out, _ = self._attn(tokens, tokens, tokens, need_weights=False)
        pooled = attn_out.mean(dim=1)  # (N, embed)
        v = self._head(pooled)  # (N, latent_dim)
        return v.squeeze(0) if squeeze else v

    def build(self, n_bands: int | None = None) -> None:
        return None

    def parameters(self) -> Iterator:
        from itertools import chain
        return chain(
            self._patch_embed.parameters(),
            self._attn.parameters(),
            self._head.parameters(),
        )

    def state_dict(self) -> dict:
        return {
            "patch_embed": self._patch_embed.state_dict(),
            "attn": self._attn.state_dict(),
            "head": self._head.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        if "patch_embed" in sd:
            self._patch_embed.load_state_dict(sd["patch_embed"])
        if "attn" in sd:
            self._attn.load_state_dict(sd["attn"])
        if "head" in sd:
            self._head.load_state_dict(sd["head"])


class SpectralAutoencoderEncoder:
    """Real torch encoder/decoder over per-band channel means.

    The encoder dim depends on ``n_bands``; we lazy-build (and cache) the
    encoder/decoder the first time we see a given ``n_bands`` so that
    ``reconstruct(emb, n_bands)`` always returns through the encoder's
    matching decoder rather than a fresh random matrix.
    """

    name = "autoencoder"
    hidden: int = 64

    def __init__(self, latent_dim: int = 32, seed: int = 0) -> None:
        import torch
        self._torch = torch
        self.latent_dim = latent_dim
        self.seed = seed
        self._enc: object | None = None
        self._dec: object | None = None
        self._built_for: int | None = None

    def _build(self, n_bands: int) -> None:
        torch = self._torch
        from torch import nn

        gen = torch.Generator(device="cpu").manual_seed(
            stable_seed(self.name, n_bands, self.latent_dim, self.seed)
        )
        enc = nn.Sequential(
            nn.Linear(n_bands, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.latent_dim),
            nn.Tanh(),
        )
        dec = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, n_bands),
        )
        _seeded_init(enc, gen, torch)
        _seeded_init(dec, gen, torch)
        enc.eval()
        dec.eval()
        self._enc = enc
        self._dec = dec
        self._built_for = n_bands

    def encode(self, tile_id: str, time: int, spectral: np.ndarray) -> SpectralEmbedding:
        torch = self._torch
        data, valid = band_quality_filter(spectral, min_valid=1)
        data = impute_missing_bands(data, valid)
        flat = data.mean(axis=(1, 2)).astype(np.float32)  # (B,)
        n_bands = int(flat.shape[0])
        self.build(n_bands)
        with torch.no_grad():
            v = self._enc(torch.from_numpy(flat))  # type: ignore[misc]
        out = v.cpu().numpy().astype(np.float32)
        assert_finite(out, "spectral.autoencoder.embedding")
        return SpectralEmbedding(
            tile_id=tile_id, time=time, vector=out,
            backend=self.name, valid_bands=int(valid.sum()),
        )

    # -------------------- trainable interface ----------------------------

    def build(self, n_bands: int | None = None) -> None:
        if n_bands is None:
            return  # caller can build lazily on first forward
        if self._built_for != n_bands:
            self._build(int(n_bands))

    def forward_torch(self, spectral_t):
        if spectral_t.dim() == 3:
            squeeze = True
            spectral_t = spectral_t.unsqueeze(0)
        else:
            squeeze = False
        spectral_t = self._torch.nan_to_num(spectral_t, nan=0.0, posinf=0.0, neginf=0.0)
        # Per-band channel mean: (N, B)
        flat = spectral_t.mean(dim=(2, 3))
        n_bands = int(flat.shape[1])
        self.build(n_bands)
        v = self._enc(flat)  # (N, latent_dim)
        return v.squeeze(0) if squeeze else v

    def parameters(self) -> Iterator:
        if self._enc is None:
            return iter(())
        from itertools import chain
        return chain(self._enc.parameters(), self._dec.parameters())

    def state_dict(self) -> dict:
        if self._enc is None:
            return {}
        return {
            "enc": self._enc.state_dict(),
            "dec": self._dec.state_dict(),
            "n_bands": int(self._built_for or 0),
        }

    def load_state_dict(self, sd: dict) -> None:
        if not sd:
            return
        n_bands = int(sd.get("n_bands", 0))
        if n_bands > 0:
            self.build(n_bands)
        if self._enc is not None and "enc" in sd:
            self._enc.load_state_dict(sd["enc"])
        if self._dec is not None and "dec" in sd:
            self._dec.load_state_dict(sd["dec"])

    def reconstruct(self, embedding: SpectralEmbedding, n_bands: int) -> np.ndarray:
        """Run the matching decoder; rebuilds for n_bands if first time seen."""
        torch = self._torch
        if self._built_for != n_bands:
            self._build(n_bands)
        with torch.no_grad():
            r = self._dec(torch.from_numpy(embedding.vector.astype(np.float32)))  # type: ignore[misc]
        return r.cpu().numpy().astype(np.float32)


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

    # No trainable params; forward_torch returns the same statistics as a tensor.

    def forward_torch(self, spectral_t):
        import torch as _t

        if spectral_t.dim() == 3:
            squeeze = True
            spectral_t = spectral_t.unsqueeze(0)
        else:
            squeeze = False
        spectral_t = _t.nan_to_num(spectral_t, nan=0.0, posinf=0.0, neginf=0.0)
        n, b, h, w = spectral_t.shape
        flat = spectral_t.view(n, b, h * w)
        stats = _t.cat([
            flat.mean(dim=2),
            flat.std(dim=2),
            flat.amin(dim=2),
            flat.amax(dim=2),
        ], dim=1)  # (N, 4*B)
        out = _t.zeros(n, self.latent_dim, dtype=stats.dtype)
        take = min(self.latent_dim, int(stats.shape[1]))
        out[:, :take] = stats[:, :take]
        return out.squeeze(0) if squeeze else out

    def build(self, n_bands: int | None = None) -> None:
        return None

    def parameters(self) -> Iterator:
        return iter(())

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, sd: dict) -> None:
        return None


SpectralEncoderRegistry: Registry[SpectralEncoder] = Registry("spectral-encoders")
SpectralEncoderRegistry.register("1d_cnn", lambda **kw: SpectralCNN1DEncoder(**kw))
SpectralEncoderRegistry.register("transformer", lambda **kw: SpectralTransformerEncoder(**kw))
SpectralEncoderRegistry.register("autoencoder", lambda **kw: SpectralAutoencoderEncoder(**kw))
SpectralEncoderRegistry.register("statistical", lambda **kw: SpectralStatisticalEncoder(**kw))
