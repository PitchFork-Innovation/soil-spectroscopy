import numpy as np
import pytest

from soilspec.fusion import FusionConfig, FusionEngine, FusionStrategyRegistry
from soilspec.types import SpatialEmbedding, SpectralEmbedding


def _make_embeddings(spec_dim=16, spat_dim=16, tile_id="t", time=0):
    rng = np.random.default_rng(123)
    spec = SpectralEmbedding(
        tile_id=tile_id, time=time,
        vector=rng.standard_normal(spec_dim).astype(np.float32),
        backend="1d_cnn", valid_bands=6,
    )
    spat = SpatialEmbedding(
        tile_id=tile_id, time=time,
        vector=rng.standard_normal(spat_dim).astype(np.float32),
        backend="cnn", patch_size=8,
    )
    return spec, spat


@pytest.mark.parametrize("strategy", ["concat", "attention", "gating", "deep"])
def test_fusion_strategy_output_shape(strategy):
    cfg = FusionConfig(strategy=strategy)
    engine = FusionEngine(cfg)
    spec, spat = _make_embeddings()
    fused = engine.fuse(spec, spat)
    assert fused.vector.shape == (cfg.output_dim,)
    assert fused.strategy == strategy
    assert fused.degraded is False


@pytest.mark.parametrize("strategy", ["concat", "attention", "gating", "deep"])
def test_fusion_is_deterministic(strategy):
    cfg = FusionConfig(strategy=strategy)
    spec, spat = _make_embeddings()
    a = FusionEngine(cfg).fuse(spec, spat)
    b = FusionEngine(cfg).fuse(spec, spat)
    assert np.array_equal(a.vector, b.vector)


def test_fusion_capability_channels_partition_output_exactly():
    cfg = FusionConfig(strategy="concat")
    engine = FusionEngine(cfg)
    spec, spat = _make_embeddings()
    fused = engine.fuse(spec, spat)
    spans = sorted((s.start, s.stop) for s in fused.channels.values())
    cur = 0
    for start, stop in spans:
        assert start == cur
        cur = stop
    assert cur == fused.vector.shape[0]


def test_fusion_keys_must_match():
    cfg = FusionConfig()
    spec, spat = _make_embeddings()
    spat = SpatialEmbedding(tile_id="other", time=0, vector=spat.vector, backend="cnn", patch_size=8)
    with pytest.raises(ValueError):
        FusionEngine(cfg).fuse(spec, spat)


def test_fusion_degraded_when_one_modality_missing():
    cfg = FusionConfig(strategy="concat")
    engine = FusionEngine(cfg)
    spec, _ = _make_embeddings()
    fused = engine.fuse(spec, None)
    assert fused.degraded is True
    assert fused.missing_modalities == ("spatial",)
    assert fused.vector.shape == (cfg.output_dim,)


def test_fusion_requires_at_least_one_modality():
    cfg = FusionConfig()
    with pytest.raises(ValueError):
        FusionEngine(cfg).fuse(None, None)


def test_fusion_strategy_registry_returns_known_names():
    names = FusionStrategyRegistry.names()
    assert {"concat", "attention", "gating", "deep"} <= set(names)
