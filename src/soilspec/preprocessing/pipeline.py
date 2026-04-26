"""Preprocessor: routes records, runs each pathway, joins on (tile_id, time).

Both pathways are step-toggleable so a missing modality skips inapplicable
steps; the descriptor records which steps actually ran for downstream
confidence accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from ..ingestion.adapters import RawAsset
from ..storage import StorageTier, StorageTierManager, preprocessed_key
from ..types import (
    SENTINEL1, SENTINEL2, VECTOR, BoundingBox, PreprocessedRecord, RawObservationHandle,
)
from . import spatial, vector


class MisalignedSampleError(ValueError):
    """Raised when a spatial output and a vector output cannot be co-aligned."""


@dataclass
class PreprocessConfig:
    target_pixel_size: float = 1.0
    target_shape: tuple[int, int] = (32, 32)
    tile_size: int = 16
    target_crs: str = "EPSG:4326"
    enable_steps: Mapping[str, bool] = field(default_factory=dict)

    def is_enabled(self, name: str) -> bool:
        return self.enable_steps.get(name, True)


class Preprocessor:
    """Drives both pathways and emits co-aligned multimodal records."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    # ----------------------------- public API -----------------------------

    def preprocess(
        self,
        handles: Iterable[RawObservationHandle],
        storage: StorageTierManager,
    ) -> list[PreprocessedRecord]:
        # Group handles by modality and time.
        spatial_records: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
        spatial_meta: dict[tuple[str, str, int], tuple[str, BoundingBox, tuple[str, ...]]] = {}
        vector_records: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
        vector_meta: dict[tuple[str, str, int], tuple[str, BoundingBox, tuple[str, ...]]] = {}

        for handle in handles:
            asset: RawAsset = storage.get(StorageTier.RAW, handle.storage_key)
            if handle.modality in (SENTINEL1, SENTINEL2):
                self._process_spatial(asset, spatial_records, spatial_meta)
            elif handle.modality == VECTOR:
                self._process_vector(asset, vector_records, vector_meta)
            # other modalities (insitu, hyperspectral) tolerated as no-op for MVP

        out = co_align(
            spatial_records, spatial_meta, vector_records, vector_meta, self.config
        )

        # persist preprocessed records
        for record in out:
            for modality, arr in record.spatial.items():
                storage.put(
                    StorageTier.PREPROCESSED,
                    preprocessed_key(record.aoi_id, record.tile_id, record.time, modality),
                    arr,
                )
            if record.vector:
                storage.put(
                    StorageTier.PREPROCESSED,
                    preprocessed_key(record.aoi_id, record.tile_id, record.time, "vector"),
                    record.vector,
                )
        return out

    # ----------------------------- pathways -------------------------------

    def _process_spatial(self, asset: RawAsset, store, meta_store) -> None:
        cfg = self.config
        descriptor: list[str] = []
        if asset.modality == SENTINEL2:
            payload = asset.payload
            data = payload["reflectance"]
            qa = payload.get("qa")
            raster = spatial.Raster(
                data=data,
                crs=cfg.target_crs,
                bounds=asset.metadata.bbox,
                pixel_size=cfg.target_pixel_size,
            )
            if cfg.is_enabled("cloud_shadow_mask"):
                raster = spatial.cloud_shadow_mask(raster, qa_band=qa)
                descriptor.append("cloud_shadow_mask")
            # S2 does not pass through radar_calibration
        elif asset.modality == SENTINEL1:
            data = asset.payload
            raster = spatial.Raster(
                data=data,
                crs=cfg.target_crs,
                bounds=asset.metadata.bbox,
                pixel_size=cfg.target_pixel_size,
            )
            if cfg.is_enabled("cloud_shadow_mask"):
                raster = spatial.cloud_shadow_mask(raster, qa_band=None)
                descriptor.append("cloud_shadow_mask")
            if cfg.is_enabled("radar_calibration"):
                raster = spatial.radar_calibration(raster)
                descriptor.append("radar_calibration")
        else:  # pragma: no cover - guarded by caller
            return

        if cfg.is_enabled("resolution_harmonization"):
            raster = spatial.resolution_harmonization(
                raster, cfg.target_pixel_size, cfg.target_shape
            )
            descriptor.append("resolution_harmonization")

        if cfg.is_enabled("tile_extraction"):
            tiles = spatial.tile_extraction(raster, asset.metadata.bbox, cfg.tile_size)
            descriptor.append("tile_extraction")
            for tile in tiles:
                key = ("aoi", tile.tile_id, asset.metadata.timestamp)
                store.setdefault(key, {})[asset.modality] = tile.raster.data
                meta_store[key] = (cfg.target_crs, tile.bounds, tuple(descriptor))

    def _process_vector(self, asset: RawAsset, store, meta_store) -> None:
        cfg = self.config
        descriptor: list[str] = []
        attributes = {k: np.asarray(v, dtype=float) for k, v in asset.payload.items()}
        records = vector.VectorRecords(attributes=attributes)
        if cfg.is_enabled("imputation"):
            records = vector.impute_missing(records, required=tuple(records.attributes))
            descriptor.append("imputation")
        if cfg.is_enabled("normalization"):
            records, _ = vector.normalize_features(records)
            descriptor.append("normalization")
        if cfg.is_enabled("attribute_filter"):
            records = vector.attribute_filter(records)
            descriptor.append("attribute_filter")
        if cfg.is_enabled("geospatial_alignment"):
            records = vector.geospatial_alignment(records, cfg.target_crs)
            descriptor.append("geospatial_alignment")

        # broadcast vector to all tiles of the same AOI; tile_ids are deterministic
        # in spatial._extract — we don't know them yet, so we stash by `*`.
        key = ("aoi", "*", asset.metadata.timestamp)
        store[key] = records.attributes
        meta_store[key] = (cfg.target_crs, asset.metadata.bbox, tuple(descriptor))


def co_align(
    spatial_records: dict[tuple[str, str, int], dict[str, np.ndarray]],
    spatial_meta: dict[tuple[str, str, int], tuple[str, BoundingBox, tuple[str, ...]]],
    vector_records: dict[tuple[str, str, int], dict[str, np.ndarray]],
    vector_meta: dict[tuple[str, str, int], tuple[str, BoundingBox, tuple[str, ...]]],
    config: PreprocessConfig,
) -> list[PreprocessedRecord]:
    """Join spatial + vector outputs by (tile_id, time)."""
    out: list[PreprocessedRecord] = []
    seen: set[tuple[str, int]] = set()
    # spatial tiles drive the join; vector is broadcast across tiles within
    # the same time bucket. If only vector is present we synthesize a single
    # (tile_id="vector_only", time) record.
    for (aoi_id, tile_id, t), modalities in spatial_records.items():
        if (tile_id, t) in seen:
            continue
        seen.add((tile_id, t))
        crs, bounds, descriptor_s = spatial_meta[(aoi_id, tile_id, t)]
        v_attrs: dict[str, np.ndarray] = {}
        descriptor_v: tuple[str, ...] = ()
        for (vk_aoi, vk_tile, vk_t), v in vector_records.items():
            if vk_t == t and (vk_tile == tile_id or vk_tile == "*"):
                v_attrs = v
                descriptor_v = vector_meta[(vk_aoi, vk_tile, vk_t)][2]
                break
        if not v_attrs:
            for (vk_aoi, vk_tile, vk_t), v in vector_records.items():
                if vk_tile == "*":
                    v_attrs = v
                    descriptor_v = vector_meta[(vk_aoi, vk_tile, vk_t)][2]
                    break
        # validate join
        if v_attrs and not v_attrs:  # pragma: no cover - defensive
            raise MisalignedSampleError("empty vector attributes for matched tile")
        record = PreprocessedRecord(
            aoi_id=aoi_id,
            tile_id=tile_id,
            time=t,
            crs=crs,
            bounds=bounds,
            spatial=dict(modalities),
            vector=v_attrs,
            pathway_descriptor=descriptor_s + descriptor_v,
        )
        out.append(record)
    if not spatial_records and vector_records:
        for (aoi_id, _tile, t), v in vector_records.items():
            crs, bounds, descriptor_v = vector_meta[(aoi_id, _tile, t)]
            out.append(PreprocessedRecord(
                aoi_id=aoi_id,
                tile_id="vector_only",
                time=t,
                crs=crs,
                bounds=bounds,
                spatial={},
                vector=v,
                pathway_descriptor=descriptor_v,
            ))
    return out
