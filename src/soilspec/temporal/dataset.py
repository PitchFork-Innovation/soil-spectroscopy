"""Temporal dataset module.

A time-ordered store of fused representations (and inferred properties) keyed
by spatial cell. The dataset is the load-bearing artifact of the system: most
downstream value reads from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..storage import StorageTier, StorageTierManager, temporal_key


class InsufficientHistoryError(RuntimeError):
    """Raised by feature extractors when a series is below the minimum length."""


class PayloadConflictError(ValueError):
    """Raised when an idempotent insert receives a conflicting payload."""


@dataclass
class TimeSeries:
    """A time-ordered list of (time, vector) entries for a single cell."""

    cell_id: str
    times: list[int]
    vectors: list[np.ndarray]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.times)

    def window(self, start: int, end: int) -> "TimeSeries":
        idx = [i for i, t in enumerate(self.times) if start <= t <= end]
        return TimeSeries(
            cell_id=self.cell_id,
            times=[self.times[i] for i in idx],
            vectors=[self.vectors[i] for i in idx],
        )


@dataclass
class SufficiencyCriteria:
    """Predicate used by the scheduled loop's gate."""

    min_samples: int = 3
    min_window: int = 0  # seconds; 0 disables this check
    max_gap: int | None = None  # seconds between consecutive samples

    def evaluate(self, series: TimeSeries) -> bool:
        if len(series) < self.min_samples:
            return False
        if self.min_window > 0 and (series.times[-1] - series.times[0]) < self.min_window:
            return False
        if self.max_gap is not None:
            gaps = [b - a for a, b in zip(series.times, series.times[1:])]
            if any(g > self.max_gap for g in gaps):
                return False
        return True


class TemporalDataset:
    """In-memory time-ordered store keyed by cell. Backed by storage tier."""

    def __init__(self, storage: StorageTierManager) -> None:
        self._storage = storage

    # -------------------------- inserts -----------------------------------

    def append(self, cell_id: str, time: int, vector: np.ndarray) -> None:
        """Insert (time, vector). Idempotent: same key+payload is a no-op."""
        series = self._load(cell_id)
        if time in series.times:
            existing = series.vectors[series.times.index(time)]
            if existing.shape != vector.shape or not np.array_equal(existing, vector):
                raise PayloadConflictError(
                    f"conflicting payload for cell={cell_id} time={time}"
                )
            return  # no-op
        # insert preserving time-ascending order
        idx = self._bisect(series.times, time)
        series.times.insert(idx, int(time))
        series.vectors.insert(idx, np.asarray(vector))
        self._save(series)

    # -------------------------- queries -----------------------------------

    def series(self, cell_id: str, start: int | None = None, end: int | None = None) -> TimeSeries:
        s = self._load(cell_id)
        if start is None and end is None:
            return s
        return s.window(start if start is not None else s.times[0] if s.times else 0,
                        end if end is not None else s.times[-1] if s.times else 0)

    def query_by_tile_and_time(self, cell_id: str, time: int) -> np.ndarray:
        s = self._load(cell_id)
        if time not in s.times:
            raise KeyError(f"no entry for cell={cell_id} time={time}")
        return s.vectors[s.times.index(time)]

    def query_range(self, cell_id: str, start: int, end: int) -> TimeSeries:
        return self.series(cell_id, start, end)

    def cells(self) -> Iterator[str]:
        for k in self._storage.list(StorageTier.TEMPORAL, prefix="cell/"):
            yield k.split("/", 1)[1]

    def sufficient(self, cell_id: str, criteria: SufficiencyCriteria) -> bool:
        return criteria.evaluate(self._load(cell_id))

    # ---------------------------- internals -------------------------------

    def _load(self, cell_id: str) -> TimeSeries:
        key = temporal_key(cell_id)
        if not self._storage.exists(StorageTier.TEMPORAL, key):
            return TimeSeries(cell_id=cell_id, times=[], vectors=[])
        return self._storage.get(StorageTier.TEMPORAL, key)

    def _save(self, series: TimeSeries) -> None:
        self._storage.put(StorageTier.TEMPORAL, temporal_key(series.cell_id), series)

    @staticmethod
    def _bisect(times: list[int], t: int) -> int:
        lo, hi = 0, len(times)
        while lo < hi:
            mid = (lo + hi) // 2
            if times[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        return lo
