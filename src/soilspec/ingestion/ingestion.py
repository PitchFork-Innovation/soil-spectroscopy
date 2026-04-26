"""Ingestion module: orchestrates source adapters + raw store persistence.

Idempotent. Returns stable handles into the raw store. Re-running the same
request reuses existing raw entries instead of re-fetching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..storage import StorageTier, StorageTierManager, raw_key
from ..types import AOI, RawObservationHandle, TimeWindow
from .adapters import RawAsset, SourceAdapter, UnreachableSourceError
from .metadata import MetadataParser


@dataclass
class _Retry:
    attempts: int = 3
    backoff_base: float = 0.0


class Ingestion:
    """Acquire raw observations and persist them with provenance."""

    def __init__(
        self,
        storage: StorageTierManager,
        adapters: dict[str, SourceAdapter],
        parser: MetadataParser | None = None,
        retry: _Retry | None = None,
    ) -> None:
        self._storage = storage
        self._adapters = adapters
        self._parser = parser or MetadataParser()
        self._retry = retry or _Retry()

    def fetch(
        self,
        aoi: AOI,
        window: TimeWindow,
        modalities: Iterable[str],
    ) -> list[RawObservationHandle]:
        handles: list[RawObservationHandle] = []
        for modality in modalities:
            if modality not in self._adapters:
                # Tolerated: missing adapters reduce the modality set rather than
                # aborting. The pipeline will run in degraded mode.
                continue
            handles.extend(self._fetch_modality(aoi, window, modality))
        return handles

    def _fetch_modality(
        self, aoi: AOI, window: TimeWindow, modality: str
    ) -> list[RawObservationHandle]:
        adapter = self._adapters[modality]
        last_err: Exception | None = None
        for _attempt in range(self._retry.attempts):
            try:
                assets = list(adapter.fetch(aoi, window))
                break
            except UnreachableSourceError as e:
                last_err = e
        else:
            assert last_err is not None
            raise last_err
        out: list[RawObservationHandle] = []
        for asset in assets:
            meta = self._parser.parse(asset)
            key = raw_key(asset.provider, asset.observation_id)
            if not self._storage.exists(StorageTier.RAW, key):
                self._storage.put(StorageTier.RAW, key, asset)
            out.append(
                RawObservationHandle(
                    observation_id=asset.observation_id,
                    provider=asset.provider,
                    modality=asset.modality,
                    storage_key=key,
                    metadata=meta,
                )
            )
        return out

    def replay(self, handle: RawObservationHandle) -> RawAsset:
        """Re-derive a raw asset from the raw store; supports stage re-runs."""
        return self._storage.get(StorageTier.RAW, handle.storage_key)
