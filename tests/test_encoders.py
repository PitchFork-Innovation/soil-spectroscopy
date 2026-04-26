import numpy as np
import pytest

from soilspec.encoders import (
    SpatialEncoderRegistry, SpectralEncoderRegistry,
    band_quality_filter, impute_missing_bands,
)


@pytest.fixture
def spectral_input():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, size=(6, 16, 16)).astype(np.float32)


@pytest.fixture
def raster_input():
    rng = np.random.default_rng(1)
    return rng.uniform(-25, -5, size=(2, 32, 32)).astype(np.float32)


# --------------------------- spectral encoders -----------------------------


@pytest.mark.parametrize("name", ["1d_cnn", "transformer", "autoencoder", "statistical"])
def test_spectral_encoder_output_shape(name, spectral_input):
    enc = SpectralEncoderRegistry.create(name, latent_dim=24, seed=0)
    emb = enc.encode("tile-1", 1000, spectral_input)
    assert emb.vector.shape == (24,)
    assert emb.tile_id == "tile-1"
    assert emb.time == 1000
    assert emb.backend == name
    assert emb.valid_bands == 6


@pytest.mark.parametrize("name", ["1d_cnn", "transformer", "autoencoder", "statistical"])
def test_spectral_encoder_is_deterministic(name, spectral_input):
    a = SpectralEncoderRegistry.create(name, latent_dim=16, seed=0).encode("t", 1, spectral_input)
    b = SpectralEncoderRegistry.create(name, latent_dim=16, seed=0).encode("t", 1, spectral_input)
    assert np.array_equal(a.vector, b.vector)


def test_spectral_encoder_handles_missing_band(spectral_input):
    nan_band = spectral_input.copy()
    nan_band[2] = np.nan
    enc = SpectralEncoderRegistry.create("1d_cnn", latent_dim=8, seed=0)
    emb = enc.encode("t", 1, nan_band)
    assert np.all(np.isfinite(emb.vector))
    assert emb.valid_bands == 5


def test_band_quality_filter_rejects_all_missing():
    arr = np.full((3, 4, 4), np.nan, dtype=np.float32)
    with pytest.raises(ValueError):
        band_quality_filter(arr, min_valid=1)


def test_impute_missing_bands_uses_valid_bands():
    arr = np.zeros((3, 2, 2), dtype=np.float32)
    arr[1] = np.nan
    arr[0] = 2.0
    arr[2] = 4.0
    valid = np.array([True, False, True])
    out = impute_missing_bands(arr, valid)
    assert np.all(np.isfinite(out))
    # imputed band 1 should equal mean of bands 0 & 2 = 3.0
    assert np.allclose(out[1], 3.0)


# --------------------------- spatial encoders ------------------------------


@pytest.mark.parametrize("name", ["cnn", "vit", "autoencoder"])
def test_spatial_encoder_output_shape(name, raster_input):
    enc = SpatialEncoderRegistry.create(name, latent_dim=16, patch_size=8, stride=4, context_patches=0, seed=0)
    emb = enc.encode("t", 0, raster_input)
    assert emb.vector.shape == (16,)
    assert emb.backend == name
    assert emb.patch_size == 8


@pytest.mark.parametrize("name", ["cnn", "vit", "autoencoder"])
def test_spatial_encoder_is_deterministic(name, raster_input):
    a = SpatialEncoderRegistry.create(name, latent_dim=16, patch_size=8, stride=4, context_patches=0, seed=0).encode("t", 0, raster_input)
    b = SpatialEncoderRegistry.create(name, latent_dim=16, patch_size=8, stride=4, context_patches=0, seed=0).encode("t", 0, raster_input)
    assert np.array_equal(a.vector, b.vector)


def test_spatial_encoder_context_patches_documented(raster_input):
    """Context patches are configurable; padding behavior is well-defined (edge padding)."""
    enc_no_ctx = SpatialEncoderRegistry.create("cnn", latent_dim=8, patch_size=8, stride=4, context_patches=0, seed=0)
    enc_ctx = SpatialEncoderRegistry.create("cnn", latent_dim=8, patch_size=8, stride=4, context_patches=1, seed=0)
    emb1 = enc_no_ctx.encode("t", 0, raster_input)
    emb2 = enc_ctx.encode("t", 0, raster_input)
    assert emb1.vector.shape == emb2.vector.shape
    # outputs differ because context changes the patch population
    assert not np.array_equal(emb1.vector, emb2.vector)


def test_spatial_encoder_accepts_2d_raster(raster_input):
    enc = SpatialEncoderRegistry.create("cnn", latent_dim=8, patch_size=8, stride=4, context_patches=0, seed=0)
    emb = enc.encode("t", 0, raster_input[0])
    assert emb.vector.shape == (8,)
