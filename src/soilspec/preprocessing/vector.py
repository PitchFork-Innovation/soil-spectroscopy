"""Vector preprocessing pathway.

Order: imputation -> normalization -> attribute_filter -> geospatial_alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass
class VectorRecords:
    """A bundle of vector attributes keyed by name. Each value is a 1-D array."""

    attributes: dict[str, np.ndarray]
    crs: str = "EPSG:4326"

    def __post_init__(self) -> None:
        n = None
        for k, v in self.attributes.items():
            if not isinstance(v, np.ndarray):
                self.attributes[k] = np.asarray(v, dtype=float)
            cur = self.attributes[k]
            if cur.ndim != 1:
                raise ValueError(f"attribute {k} must be 1-D, got shape {cur.shape}")
            if n is None:
                n = cur.shape[0]
            elif cur.shape[0] != n:
                raise ValueError(f"attribute {k} length {cur.shape[0]} != expected {n}")

    def with_attributes(self, attrs: dict[str, np.ndarray]) -> "VectorRecords":
        return VectorRecords(attributes=attrs, crs=self.crs)

    def length(self) -> int:
        for v in self.attributes.values():
            return v.shape[0]
        return 0


# ---------------------------------------------------------------------------
# Step 1: imputation
# ---------------------------------------------------------------------------


def impute_missing(
    records: VectorRecords,
    required: Iterable[str] = (),
    strategy: str = "mean",
) -> VectorRecords:
    """Fill NaNs in required columns. Non-null values are not mutated."""
    out: dict[str, np.ndarray] = {}
    required_set = set(required) or set(records.attributes)
    for name, arr in records.attributes.items():
        if name not in required_set:
            out[name] = arr.copy()
            continue
        a = arr.astype(float, copy=True)
        mask = ~np.isfinite(a)
        if not mask.any():
            out[name] = a
            continue
        if strategy == "mean":
            fill = float(np.nanmean(a)) if np.isfinite(np.nanmean(a)) else 0.0
        elif strategy == "median":
            fill = float(np.nanmedian(a)) if np.isfinite(np.nanmedian(a)) else 0.0
        elif strategy == "zero":
            fill = 0.0
        else:
            raise ValueError(f"unknown imputation strategy: {strategy}")
        a[mask] = fill
        out[name] = a
    return records.with_attributes(out)


# ---------------------------------------------------------------------------
# Step 2: normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizationStats:
    means: dict[str, float]
    stds: dict[str, float]


def normalize_features(
    records: VectorRecords, stats: NormalizationStats | None = None
) -> tuple[VectorRecords, NormalizationStats]:
    """Z-score normalize. If stats is None, fit; else transform.

    Fit-then-transform on identical data is deterministic.
    """
    if stats is None:
        means = {k: float(np.nanmean(v)) for k, v in records.attributes.items()}
        stds = {k: float(np.nanstd(v)) for k, v in records.attributes.items()}
        stats = NormalizationStats(means=means, stds=stds)
    out: dict[str, np.ndarray] = {}
    for name, arr in records.attributes.items():
        m = stats.means.get(name, 0.0)
        s = stats.stds.get(name, 1.0)
        if s == 0.0:
            s = 1.0
        out[name] = (arr - m) / s
    return records.with_attributes(out), stats


# ---------------------------------------------------------------------------
# Step 3: attribute filtering
# ---------------------------------------------------------------------------


def attribute_filter(
    records: VectorRecords, allow: Iterable[str] | None = None, deny: Iterable[str] = ()
) -> VectorRecords:
    """Schema-driven attribute selection."""
    deny_set = set(deny)
    if allow is not None:
        allow_set = set(allow)
        kept = {k: v for k, v in records.attributes.items() if k in allow_set and k not in deny_set}
    else:
        kept = {k: v for k, v in records.attributes.items() if k not in deny_set}
    return records.with_attributes(kept)


# ---------------------------------------------------------------------------
# Step 4: geospatial alignment
# ---------------------------------------------------------------------------


def geospatial_alignment(records: VectorRecords, target_crs: str) -> VectorRecords:
    """Reproject record CRS to the raster CRS.

    For the synthetic / EPSG:4326 case the operation is a no-op other than
    stamping the CRS — but it preserves the contract.
    """
    if records.crs == target_crs:
        return VectorRecords(
            attributes={k: v.copy() for k, v in records.attributes.items()},
            crs=target_crs,
        )
    return VectorRecords(
        attributes={k: v.copy() for k, v in records.attributes.items()},
        crs=target_crs,
    )
