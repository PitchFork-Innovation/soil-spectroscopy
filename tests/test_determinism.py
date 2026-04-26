"""Determinism / reproducibility tests across encoders, fusion, ensemble, scoring."""

import subprocess
import sys

import numpy as np
import pytest

from soilspec.capability import CapabilityScoringEngine, RulesEngine
from soilspec.encoders import SpatialEncoderRegistry, SpectralEncoderRegistry
from soilspec.fusion import FusionConfig, FusionEngine
from soilspec.inference import EnsembleInferenceEngine, InferenceConfig
from soilspec.types import (
    CharacteristicScores, SpatialEmbedding, SpectralEmbedding,
)


@pytest.mark.parametrize("name", ["1d_cnn", "transformer", "autoencoder", "statistical"])
def test_spectral_determinism(name):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(6, 16, 16)).astype(np.float32)
    a = SpectralEncoderRegistry.create(name, latent_dim=16, seed=0).encode("t", 0, x)
    b = SpectralEncoderRegistry.create(name, latent_dim=16, seed=0).encode("t", 0, x)
    np.testing.assert_array_equal(a.vector, b.vector)


@pytest.mark.parametrize("name", ["cnn", "vit", "autoencoder"])
def test_spatial_determinism(name):
    rng = np.random.default_rng(0)
    x = rng.uniform(-25, -5, size=(2, 32, 32)).astype(np.float32)
    a = SpatialEncoderRegistry.create(name, latent_dim=16, patch_size=8, stride=4, context_patches=0, seed=0).encode("t", 0, x)
    b = SpatialEncoderRegistry.create(name, latent_dim=16, patch_size=8, stride=4, context_patches=0, seed=0).encode("t", 0, x)
    np.testing.assert_array_equal(a.vector, b.vector)


@pytest.mark.parametrize("strategy", ["concat", "attention", "gating", "deep"])
def test_fusion_determinism(strategy):
    rng = np.random.default_rng(0)
    spec = SpectralEmbedding("t", 0, rng.standard_normal(16).astype(np.float32), "x", 6)
    spat = SpatialEmbedding("t", 0, rng.standard_normal(16).astype(np.float32), "x", 8)
    cfg = FusionConfig(strategy=strategy)
    a = FusionEngine(cfg).fuse(spec, spat)
    b = FusionEngine(cfg).fuse(spec, spat)
    np.testing.assert_array_equal(a.vector, b.vector)


def test_inference_determinism():
    rng = np.random.default_rng(0)
    spec = SpectralEmbedding("t", 0, rng.standard_normal(16).astype(np.float32), "x", 6)
    spat = SpatialEmbedding("t", 0, rng.standard_normal(16).astype(np.float32), "x", 8)
    cfg = FusionConfig(strategy="concat")
    fused = FusionEngine(cfg).fuse(spec, spat)
    a = EnsembleInferenceEngine(fused_dim=cfg.output_dim, channels=fused.channels, config=InferenceConfig(seed=0)).infer(fused)
    b = EnsembleInferenceEngine(fused_dim=cfg.output_dim, channels=fused.channels, config=InferenceConfig(seed=0)).infer(fused)
    assert a.properties == b.properties


def test_capability_classification_determinism():
    rules = RulesEngine()
    a = rules.classify(CharacteristicScores(tile_id="t", scores={
        "moisture_capacity": 0.6, "infiltration_capacity": 0.5,
        "erosion_resistance": 0.7, "stability": 0.5, "resilience": 0.5,
    }))
    b = rules.classify(CharacteristicScores(tile_id="t", scores={
        "moisture_capacity": 0.6, "infiltration_capacity": 0.5,
        "erosion_resistance": 0.7, "stability": 0.5, "resilience": 0.5,
    }))
    assert a == b


def test_cross_process_determinism(tmp_path):
    """Same seed across fresh interpreters yields the same output."""
    script = tmp_path / "encode.py"
    script.write_text("""
import numpy as np, json
from soilspec.encoders import SpectralEncoderRegistry
rng = np.random.default_rng(0)
x = rng.uniform(0, 1, size=(6, 16, 16)).astype(np.float32)
e = SpectralEncoderRegistry.create('1d_cnn', latent_dim=8, seed=0).encode('t', 0, x)
print(json.dumps([float(v) for v in e.vector]))
""")
    out1 = subprocess.check_output([sys.executable, str(script)]).decode().strip()
    out2 = subprocess.check_output([sys.executable, str(script)]).decode().strip()
    assert out1 == out2
