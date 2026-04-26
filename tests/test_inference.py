import numpy as np
import pytest

from soilspec.fusion import FusionConfig, FusionEngine
from soilspec.inference import (
    ClassicMLMember, DeepLearningMember, EnsembleInferenceEngine,
    InferenceConfig, LiftingLayer, MathematicalInterpolationMember,
)
from soilspec.types import SOIL_PROPERTY_NAMES, SpatialEmbedding, SpectralEmbedding


def _fused():
    spec = SpectralEmbedding(tile_id="t", time=0, vector=np.ones(16, dtype=np.float32), backend="x", valid_bands=6)
    spat = SpatialEmbedding(tile_id="t", time=0, vector=-np.ones(16, dtype=np.float32), backend="x", patch_size=8)
    cfg = FusionConfig(strategy="concat")
    return FusionEngine(cfg).fuse(spec, spat), cfg


def test_lifting_layer_shape_and_determinism():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(48).astype(np.float32)
    a = LiftingLayer(in_dim=48, out_dim=64, seed=42)(x)
    b = LiftingLayer(in_dim=48, out_dim=64, seed=42)(x)
    assert a.shape == (64,)
    assert np.array_equal(a, b)


def test_each_member_predicts_all_property_names():
    lifted = np.zeros(48, dtype=np.float32)
    for member in (
        ClassicMLMember(in_dim=48, seed=0),
        DeepLearningMember(in_dim=48, seed=0),
        MathematicalInterpolationMember(),
    ):
        out = member.predict(lifted)
        assert set(out) == set(SOIL_PROPERTY_NAMES)
        assert all(0.0 <= v <= 1.0 for v in out.values())


def test_ensemble_meta_model_in_documented_bound():
    fused, cfg = _fused()
    eng = EnsembleInferenceEngine(
        fused_dim=cfg.output_dim,
        channels={k: v for k, v in fused.channels.items()},
        config=InferenceConfig(seed=0),
    )
    out = eng.infer(fused)
    for prop, val in out.properties.items():
        members = [m[prop] for m in out.member_outputs.values()]
        assert min(members) - 1e-9 <= val <= max(members) + 1e-9
        assert 0.0 <= val <= 1.0
        assert out.uncertainty[prop] >= 0.0


def test_ensemble_inference_is_deterministic():
    fused, cfg = _fused()
    eng_a = EnsembleInferenceEngine(fused_dim=cfg.output_dim, channels=fused.channels, config=InferenceConfig(seed=7))
    eng_b = EnsembleInferenceEngine(fused_dim=cfg.output_dim, channels=fused.channels, config=InferenceConfig(seed=7))
    a = eng_a.infer(fused)
    b = eng_b.infer(fused)
    assert a.properties == b.properties
