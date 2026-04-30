"""Tests for the training-data assembly + ridge baseline."""

from __future__ import annotations

import numpy as np
import pytest

from soilspec.groundtruth import (
    GroundTruthDataset,
    Measurement,
    TileGrid,
)
from soilspec.training import (
    InsufficientTrainingDataError,
    RidgeModel,
    assemble_training_examples,
    train_ridge,
)
from soilspec.types import (
    BoundingBox,
    FusedRepresentation,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _grid() -> TileGrid:
    return TileGrid.from_shape(
        BoundingBox(0.0, 0.0, 1.0, 1.0), raster_shape=(32, 32), tile_size=16
    )


def _fused(tile_id: str, time: int, vec: np.ndarray) -> FusedRepresentation:
    return FusedRepresentation(
        tile_id=tile_id, time=int(time), vector=vec.astype(np.float64),
        channels={"all": slice(0, vec.size)}, strategy="concat", degraded=False,
    )


def _add_sm(ds: GroundTruthDataset, lon: float, lat: float, t: int, v: float,
            source: str = "ismn") -> None:
    ds.add(Measurement(lon=lon, lat=lat, time=t,
                      properties={"soil_moisture": v}, source=source))


# ---------------------------------------------------------------------------
# Assembly: tile/time join
# ---------------------------------------------------------------------------


def test_join_aligns_fused_reps_with_gt_buckets():
    grid = _grid()
    ds = GroundTruthDataset(grid, time_bucket_seconds=86400)
    # Two GT samples in different (tile, day) buckets
    _add_sm(ds, lon=0.25, lat=0.75, t=3600, v=0.20)       # r000c000, day 0
    _add_sm(ds, lon=0.75, lat=0.25, t=90000, v=0.35)      # r001c001, day 1

    fused = [
        _fused("r000c000", time=43200, vec=np.array([1.0, 2.0])),   # day 0
        _fused("r001c001", time=129600, vec=np.array([3.0, 4.0])),  # day 1
        _fused("r000c001", time=200, vec=np.array([9.0, 9.0])),     # no GT
    ]
    ex = assemble_training_examples(fused, ds, property_names=("soil_moisture",))
    assert len(ex) == 2  # the unmatched fused rep is dropped
    assert ex.X.shape == (2, 2)
    np.testing.assert_allclose(np.sort(ex.y["soil_moisture"]), [0.20, 0.35])


def test_missing_property_yields_nan_label():
    grid = _grid()
    ds = GroundTruthDataset(grid)
    # Sample only carries soc, not soil_moisture
    ds.add(Measurement(lon=0.25, lat=0.75, time=0,
                      properties={"soc": 12.0}, source="lucas"))
    fused = [_fused("r000c000", time=0, vec=np.array([1.0, 2.0]))]
    ex = assemble_training_examples(
        fused, ds, property_names=("soil_moisture", "soc")
    )
    assert np.isnan(ex.y["soil_moisture"]).all()
    assert ex.y["soc"][0] == 12.0


def test_source_filter_limits_join():
    grid = _grid()
    ds = GroundTruthDataset(grid)
    _add_sm(ds, lon=0.25, lat=0.75, t=0, v=0.2, source="ismn")
    _add_sm(ds, lon=0.25, lat=0.75, t=0, v=0.5, source="smap")
    fused = [_fused("r000c000", time=0, vec=np.array([1.0, 2.0]))]
    ex = assemble_training_examples(
        fused, ds, property_names=("soil_moisture",), sources=("ismn",),
    )
    assert ex.y["soil_moisture"][0] == pytest.approx(0.2)
    assert ex.sources == ("ismn",)


def test_multiple_sources_are_inverse_variance_weighted():
    grid = _grid()
    ds = GroundTruthDataset(grid)
    # Two sources land on the same (tile, day): one tight (low unc), one loose.
    # The aggregate should pull toward the tight measurement.
    ds.add(Measurement(lon=0.25, lat=0.75, time=0,
                      properties={"soil_moisture": 0.10},
                      uncertainty={"soil_moisture": 0.01}, source="ismn"))
    ds.add(Measurement(lon=0.25, lat=0.75, time=0,
                      properties={"soil_moisture": 0.50},
                      uncertainty={"soil_moisture": 0.10}, source="smap"))
    fused = [_fused("r000c000", time=0, vec=np.array([1.0, 2.0]))]
    ex = assemble_training_examples(fused, ds, property_names=("soil_moisture",))
    # weights: 1/0.01² = 10000 vs 1/0.10² = 100 -> mean ≈ 0.104
    assert ex.y["soil_moisture"][0] == pytest.approx(0.104, abs=1e-3)


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------


def test_ridge_recovers_known_linear_relationship():
    rng = np.random.default_rng(0)
    n, d = 200, 4
    X = rng.normal(size=(n, d))
    true_w = np.array([0.5, -1.2, 0.0, 2.0])
    true_b = 0.3
    y = X @ true_w + true_b + rng.normal(scale=0.01, size=n)

    grid = _grid()
    ds = GroundTruthDataset(grid)
    fused = []
    # Drop synthetic measurements onto a single tile bucketed at day 0, so
    # all join on (r000c000, 0). Use distinct timestamps so the dataset
    # stores them as separate samples (different sources).
    for i in range(n):
        ds.add(Measurement(
            lon=0.25, lat=0.75, time=i,
            properties={"soil_moisture": float(y[i])},
            source=f"src{i}",  # unique per row -> no within-bucket aggregation
        ))
        fused.append(_fused("r000c000", time=i, vec=X[i]))

    ex = assemble_training_examples(fused, ds, property_names=("soil_moisture",))
    # Each fused rep gets joined to *all* GT samples in its bucket — so
    # the recovered relationship needs the labels to be paired one-to-one.
    # For this test we instead pre-aggregate per-fused-rep by using the GT
    # sample label that matches its index. Verify by training on a clean
    # X/y constructed directly:
    direct = _DirectExamples(X=X, y={"soil_moisture": y},
                             weights={"soil_moisture": np.ones(n)},
                             property_names=("soil_moisture",))
    models = train_ridge(direct, alpha=0.01)
    np.testing.assert_allclose(models["soil_moisture"].weights, true_w, atol=0.05)
    assert models["soil_moisture"].intercept == pytest.approx(true_b, abs=0.05)


class _DirectExamples:
    """Test double for TrainingExamples that bypasses the assembly step.

    The real assembler joins per (tile, bucket) — one row of X corresponds
    to potentially many GT samples. For the linear-recovery test we want a
    one-to-one (X, y) pairing without bucket entanglement.
    """
    def __init__(self, X, y, weights, property_names):
        self.X = X
        self.y = y
        self.weights = weights
        self.property_names = property_names
    def __len__(self):
        return self.X.shape[0]
    @property
    def n_features(self):
        return self.X.shape[1]


def test_ridge_predict_round_trip():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([1.0, -0.5, 2.0]) + 0.1
    ex = _DirectExamples(X=X, y={"p": y}, weights={"p": np.ones(50)},
                         property_names=("p",))
    [(_, m)] = train_ridge(ex, alpha=0.01).items()
    assert isinstance(m, RidgeModel)
    pred = m.predict(X)
    assert pred.shape == (50,)
    # Reasonable fit
    assert np.corrcoef(pred, y)[0, 1] > 0.99


def test_ridge_skips_property_with_no_labels():
    X = np.eye(10)
    y = {"a": np.full(10, np.nan), "b": np.linspace(0, 1, 10)}
    w = {"a": np.zeros(10), "b": np.ones(10)}
    ex = _DirectExamples(X=X, y=y, weights=w, property_names=("a", "b"))
    models = train_ridge(ex, alpha=0.1, min_samples=5)
    assert "a" not in models
    assert "b" in models


def test_ridge_raises_on_underdetermined_property():
    X = np.eye(3)  # 3 samples, 3 features
    y = {"p": np.array([1.0, 2.0, 3.0])}
    w = {"p": np.ones(3)}
    ex = _DirectExamples(X=X, y=y, weights=w, property_names=("p",))
    with pytest.raises(InsufficientTrainingDataError):
        train_ridge(ex, alpha=0.1)  # default min_samples = D+1 = 4


def test_ridge_predict_shape_check():
    X = np.eye(10)
    y = {"p": np.arange(10, dtype=float)}
    w = {"p": np.ones(10)}
    ex = _DirectExamples(X=X, y=y, weights=w, property_names=("p",))
    [m] = train_ridge(ex, alpha=0.1, min_samples=2).values()
    with pytest.raises(ValueError, match="feature dim mismatch"):
        m.predict(np.zeros((1, 5)))


def test_empty_examples_returns_empty_models():
    ex = _DirectExamples(
        X=np.zeros((0, 0)), y={"p": np.zeros(0)},
        weights={"p": np.zeros(0)}, property_names=("p",),
    )
    assert train_ridge(ex) == {}
