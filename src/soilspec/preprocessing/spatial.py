"""Spatial preprocessing pathway.

Strict order (per PRD):
  cloud_shadow_mask -> radar_calibration -> resolution_harmonization -> tile_extraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..types import BoundingBox


@dataclass(frozen=True)
class Raster:
    """A georeferenced raster bundle."""

    data: np.ndarray  # (B, H, W) or (H, W)
    crs: str
    bounds: BoundingBox
    pixel_size: float
    nodata: float = float("nan")
    mask: np.ndarray | None = None  # boolean array True = invalid

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def with_data(self, data: np.ndarray, mask: np.ndarray | None = None) -> "Raster":
        return Raster(
            data=data,
            crs=self.crs,
            bounds=self.bounds,
            pixel_size=self.pixel_size,
            nodata=self.nodata,
            mask=mask if mask is not None else self.mask,
        )


# ---------------------------------------------------------------------------
# Step 1: cloud and shadow masking
# ---------------------------------------------------------------------------


def cloud_shadow_mask(raster: Raster, qa_band: np.ndarray | None = None) -> Raster:
    """Mask cloud/shadow pixels.

    If a QA band is provided (Sentinel-2-style), pixels flagged as cloud or
    shadow are masked. Idempotent — applying twice yields the same mask.
    """
    if qa_band is None:
        # No quality info available — return raster unchanged but with an
        # explicit empty mask so downstream code can still reason about it.
        if raster.mask is None:
            mask = np.zeros(_spatial_shape(raster.data), dtype=bool)
            return raster.with_data(raster.data, mask=mask)
        return raster
    if qa_band.shape != _spatial_shape(raster.data):
        raise ValueError(
            f"qa_band shape {qa_band.shape} does not match raster spatial shape "
            f"{_spatial_shape(raster.data)}"
        )
    cloud = qa_band.astype(bool)
    existing = raster.mask if raster.mask is not None else np.zeros_like(cloud)
    mask = np.logical_or(existing, cloud)
    return raster.with_data(raster.data, mask=mask)


# ---------------------------------------------------------------------------
# Step 2: radar calibration (SAR-only)
# ---------------------------------------------------------------------------


def radar_calibration(raster: Raster) -> Raster:
    """Convert raw SAR amplitudes to calibrated sigma-nought (dB).

    Deterministic. The transformation `10*log10(max(x, eps))` is a textbook
    radiometric calibration; output range is bounded by clamping.
    """
    eps = 1e-6
    data = raster.data
    # If data already looks like dB (negative range), pass through. Heuristic
    # used so cloud-only step before this can be re-run.
    if data.min() < 0 and data.max() <= 5:
        calibrated = np.clip(data, -50.0, 5.0)
    else:
        calibrated = 10.0 * np.log10(np.maximum(np.abs(data), eps))
        calibrated = np.clip(calibrated, -50.0, 5.0)
    return raster.with_data(calibrated.astype(np.float32))


# ---------------------------------------------------------------------------
# Step 3: resolution harmonization
# ---------------------------------------------------------------------------


def resolution_harmonization(
    raster: Raster, target_pixel_size: float, target_shape: tuple[int, int] | None = None
) -> Raster:
    """Resample to a common GSD on the raster's existing CRS.

    Nodata is preserved. Repeated calls with the same target are stable
    (no silent drift across calls).
    """
    if raster.pixel_size == target_pixel_size and target_shape in (None, _spatial_shape(raster.data)):
        return raster
    cur_shape = _spatial_shape(raster.data)
    if target_shape is None:
        scale = raster.pixel_size / target_pixel_size
        new_h = max(1, int(round(cur_shape[0] * scale)))
        new_w = max(1, int(round(cur_shape[1] * scale)))
    else:
        new_h, new_w = target_shape
    resampled = _nearest_resample(raster.data, (new_h, new_w))
    new_mask = (
        _nearest_resample(raster.mask.astype(np.uint8), (new_h, new_w)).astype(bool)
        if raster.mask is not None
        else None
    )
    return Raster(
        data=resampled,
        crs=raster.crs,
        bounds=raster.bounds,
        pixel_size=target_pixel_size,
        nodata=raster.nodata,
        mask=new_mask,
    )


# ---------------------------------------------------------------------------
# Step 4: tile extraction & spatial reprojection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    tile_id: str
    raster: Raster
    bounds: BoundingBox


def tile_extraction(
    raster: Raster, aoi_bounds: BoundingBox, tile_size: int
) -> list[Tile]:
    """Tile a scene into uniform tiles fully covering the AOI.

    Tiles are non-overlapping. The AOI is fully covered (the last tile may
    be padded with nodata if the scene size is not a multiple of `tile_size`).
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    h, w = _spatial_shape(raster.data)
    out: list[Tile] = []
    n_rows = (h + tile_size - 1) // tile_size
    n_cols = (w + tile_size - 1) // tile_size
    lon_step = (aoi_bounds.max_lon - aoi_bounds.min_lon) / max(n_cols, 1)
    lat_step = (aoi_bounds.max_lat - aoi_bounds.min_lat) / max(n_rows, 1)
    is_3d = raster.data.ndim == 3
    for r in range(n_rows):
        for c in range(n_cols):
            r0, c0 = r * tile_size, c * tile_size
            r1, c1 = r0 + tile_size, c0 + tile_size
            patch = _slice_with_pad(raster.data, r0, r1, c0, c1, tile_size, is_3d, raster.nodata)
            mask_patch = (
                _slice_with_pad(raster.mask, r0, r1, c0, c1, tile_size, False, True)
                if raster.mask is not None
                else None
            )
            tile_bounds = BoundingBox(
                min_lon=aoi_bounds.min_lon + c * lon_step,
                min_lat=aoi_bounds.max_lat - (r + 1) * lat_step,
                max_lon=aoi_bounds.min_lon + (c + 1) * lon_step,
                max_lat=aoi_bounds.max_lat - r * lat_step,
            )
            tile_raster = Raster(
                data=patch,
                crs=raster.crs,
                bounds=tile_bounds,
                pixel_size=raster.pixel_size,
                nodata=raster.nodata,
                mask=mask_patch,
            )
            tile_id = f"r{r:03d}c{c:03d}"
            out.append(Tile(tile_id=tile_id, raster=tile_raster, bounds=tile_bounds))
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _spatial_shape(arr: np.ndarray) -> tuple[int, int]:
    if arr.ndim == 2:
        return arr.shape  # type: ignore[return-value]
    return arr.shape[-2], arr.shape[-1]


def _nearest_resample(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = _spatial_shape(arr)
    tgt_h, tgt_w = target_shape
    row_idx = (np.linspace(0, src_h, tgt_h, endpoint=False)).astype(int)
    col_idx = (np.linspace(0, src_w, tgt_w, endpoint=False)).astype(int)
    if arr.ndim == 2:
        return arr[np.ix_(row_idx, col_idx)]
    return arr[..., row_idx[:, None], col_idx[None, :]]


def _slice_with_pad(
    arr: np.ndarray,
    r0: int, r1: int, c0: int, c1: int,
    tile_size: int, is_3d: bool, fill: float | bool,
) -> np.ndarray:
    h = _spatial_shape(arr)[0]
    w = _spatial_shape(arr)[1]
    if is_3d:
        b = arr.shape[0]
        out = np.full((b, tile_size, tile_size), fill, dtype=arr.dtype)
        rr1, cc1 = min(r1, h), min(c1, w)
        out[:, : rr1 - r0, : cc1 - c0] = arr[:, r0:rr1, c0:cc1]
    else:
        out = np.full((tile_size, tile_size), fill, dtype=arr.dtype)
        rr1, cc1 = min(r1, h), min(c1, w)
        out[: rr1 - r0, : cc1 - c0] = arr[r0:rr1, c0:cc1]
    return out
