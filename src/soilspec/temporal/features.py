"""Temporal feature extractor.

Computes higher-order temporal descriptors from a series of soil-related
vectors: trends, rates of change, persistence, volatility, recovery behavior
following events, and deviations from historical baselines.

Standardized so downstream experts and rule engines consume a stable feature
contract regardless of the upstream vector content.
"""

from __future__ import annotations

import numpy as np

from ..types import TemporalFeatureSet
from .dataset import InsufficientHistoryError, TimeSeries


_DEFAULT_NAMES = ("dim_0", "dim_1", "dim_2")


class TemporalFeatureExtractor:
    """Pure functional extractor; no state across calls."""

    def __init__(self, min_samples: int = 3, baselines: dict[str, dict[str, float]] | None = None) -> None:
        self.min_samples = min_samples
        # cell_id -> {feature_name -> baseline value}; stored on a slower cadence
        self.baselines: dict[str, dict[str, float]] = baselines or {}

    def extract(self, series: TimeSeries, names: tuple[str, ...] | None = None) -> TemporalFeatureSet:
        if len(series) < self.min_samples:
            raise InsufficientHistoryError(
                f"series for cell {series.cell_id} has {len(series)} samples "
                f"(< min {self.min_samples})"
            )
        mat = np.stack(series.vectors, axis=0)  # (T, D)
        D = mat.shape[1]
        if names is None or len(names) != D:
            names = tuple(_DEFAULT_NAMES[i] if i < len(_DEFAULT_NAMES) else f"dim_{i}" for i in range(D))

        ts = np.asarray(series.times, dtype=float)
        # normalize times to avoid huge numbers in the regression
        if ts[-1] != ts[0]:
            ts_n = (ts - ts[0]) / (ts[-1] - ts[0])
        else:
            ts_n = ts - ts[0]

        trend = {}
        rate = {}
        persistence = {}
        volatility = {}
        recovery = {}
        baseline_dev = {}
        cell_baselines = self.baselines.get(series.cell_id, {})
        for i, name in enumerate(names):
            col = mat[:, i].astype(float)
            slope = _slope(ts_n, col)
            trend[name] = float(slope)
            rate[name] = float(_avg_abs_rate(ts, col))
            persistence[name] = float(_persistence(col))
            volatility[name] = float(np.std(col))
            recovery[name] = float(_recovery(col))
            baseline = cell_baselines.get(name)
            baseline_dev[name] = float(col[-1] - baseline) if baseline is not None else float(col[-1] - col.mean())

        return TemporalFeatureSet(
            tile_id=series.cell_id,
            trend=trend,
            rate_of_change=rate,
            persistence=persistence,
            volatility=volatility,
            recovery=recovery,
            baseline_deviation=baseline_dev,
            n_samples=len(series),
        )


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    xm, ym = x.mean(), y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - xm) * (y - ym)) / denom)


def _avg_abs_rate(times: np.ndarray, values: np.ndarray) -> float:
    if times.size < 2:
        return 0.0
    dt = np.diff(times)
    dv = np.diff(values)
    safe = np.where(dt == 0, 1, dt)
    return float(np.mean(np.abs(dv / safe)))


def _persistence(values: np.ndarray) -> float:
    """Lag-1 autocorrelation; high = soil moisture lingers across observations."""
    if values.size < 3:
        return 0.0
    a, b = values[:-1], values[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _recovery(values: np.ndarray) -> float:
    """Heuristic recovery score after the lowest dip."""
    if values.size < 3:
        return 0.0
    lo_idx = int(np.argmin(values))
    if lo_idx == values.size - 1:
        return 0.0
    after = values[lo_idx:]
    if after.max() == values[lo_idx]:
        return 0.0
    return float((after[-1] - values[lo_idx]) / (after.max() - values[lo_idx] + 1e-9))
