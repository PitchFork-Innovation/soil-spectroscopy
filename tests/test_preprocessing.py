import numpy as np
import pytest

from soilspec.preprocessing import (
    cloud_shadow_mask, radar_calibration, resolution_harmonization,
    tile_extraction, impute_missing, normalize_features, attribute_filter,
    geospatial_alignment, Preprocessor,
)
from soilspec.preprocessing.pipeline import PreprocessConfig
from soilspec.preprocessing.spatial import Raster
from soilspec.preprocessing.vector import VectorRecords
from soilspec.storage import StorageTier
from soilspec.types import BoundingBox, SENTINEL1, SENTINEL2, VECTOR


# ---------------------------- spatial pathway ------------------------------


def test_cloud_mask_shape_matches_raster(synthetic_s2_raster, synthetic_s2_qa):
    out = cloud_shadow_mask(synthetic_s2_raster, qa_band=synthetic_s2_qa)
    assert out.mask is not None
    assert out.mask.shape == (32, 32)
    assert out.mask.dtype == bool


def test_cloud_mask_is_idempotent(synthetic_s2_raster, synthetic_s2_qa):
    once = cloud_shadow_mask(synthetic_s2_raster, qa_band=synthetic_s2_qa)
    twice = cloud_shadow_mask(once, qa_band=synthetic_s2_qa)
    assert np.array_equal(once.mask, twice.mask)


def test_cloud_mask_qa_shape_mismatch_raises(synthetic_s2_raster):
    with pytest.raises(ValueError):
        cloud_shadow_mask(synthetic_s2_raster, qa_band=np.zeros((10, 10), dtype=np.uint8))


def test_radar_calibration_bounded_and_deterministic(synthetic_s1_raster):
    a = radar_calibration(synthetic_s1_raster)
    b = radar_calibration(synthetic_s1_raster)
    assert np.array_equal(a.data, b.data)
    assert a.data.min() >= -50.0
    assert a.data.max() <= 5.0


def test_resolution_harmonization_targets_shape(synthetic_s2_raster):
    out = resolution_harmonization(synthetic_s2_raster, target_pixel_size=0.5, target_shape=(64, 64))
    assert out.data.shape[-2:] == (64, 64)
    # repeat -> stable
    out2 = resolution_harmonization(out, target_pixel_size=0.5, target_shape=(64, 64))
    assert np.array_equal(out.data, out2.data)


def test_tile_extraction_covers_aoi(synthetic_s2_raster):
    aoi = BoundingBox(0, 0, 1, 1)
    tiles = tile_extraction(synthetic_s2_raster, aoi, tile_size=16)
    assert len(tiles) == 4  # 32/16 == 2 per axis
    ids = sorted(t.tile_id for t in tiles)
    assert ids == ["r000c000", "r000c001", "r001c000", "r001c001"]
    for t in tiles:
        assert t.raster.data.shape[-2:] == (16, 16)


# ---------------------------- vector pathway -------------------------------


def test_imputation_fills_only_required_nans(synthetic_vector_records):
    out = impute_missing(synthetic_vector_records, required=["slope", "soc"])
    assert not np.any(np.isnan(out.attributes["slope"]))
    assert not np.any(np.isnan(out.attributes["soc"]))
    # original non-null values should be preserved
    assert out.attributes["slope"][0] == 0.0
    assert out.attributes["elevation"][0] == 100.0  # untouched (not in required)


def test_normalize_is_deterministic(synthetic_vector_records):
    imputed = impute_missing(synthetic_vector_records, required=tuple(synthetic_vector_records.attributes))
    a, stats_a = normalize_features(imputed)
    b, stats_b = normalize_features(imputed)
    assert stats_a == stats_b
    for k in a.attributes:
        assert np.allclose(a.attributes[k], b.attributes[k])


def test_attribute_filter_uses_schema(synthetic_vector_records):
    out = attribute_filter(synthetic_vector_records, allow=["slope", "elevation"])
    assert set(out.attributes) == {"slope", "elevation"}


def test_geospatial_alignment_stamps_crs(synthetic_vector_records):
    out = geospatial_alignment(synthetic_vector_records, target_crs="EPSG:3857")
    assert out.crs == "EPSG:3857"


# ---------------------------- pipeline -------------------------------------


def test_preprocessor_produces_co_aligned_records(ingestion, storage, aoi, small_window):
    handles = ingestion.fetch(aoi, small_window, [SENTINEL1, SENTINEL2, VECTOR])
    pp = Preprocessor(PreprocessConfig(target_shape=(32, 32), tile_size=16))
    records = pp.preprocess(handles, storage)
    assert records, "expected at least one record"
    for rec in records:
        # tile-time key uniqueness within the run
        assert isinstance(rec.tile_id, str) and isinstance(rec.time, int)
        assert rec.crs == "EPSG:4326"
    keys = {(r.tile_id, r.time) for r in records}
    assert len(keys) == len(records)


def test_preprocessed_records_persisted(ingestion, storage, aoi, small_window):
    handles = ingestion.fetch(aoi, small_window, [SENTINEL2])
    pp = Preprocessor(PreprocessConfig(target_shape=(32, 32), tile_size=16))
    records = pp.preprocess(handles, storage)
    assert any(storage.list(StorageTier.PREPROCESSED, prefix=aoi.aoi_id)) or any(
        storage.list(StorageTier.PREPROCESSED, prefix="aoi/")
    )
    sample = records[0]
    keys = list(storage.list(StorageTier.PREPROCESSED))
    assert any(sample.tile_id in k for k in keys)
