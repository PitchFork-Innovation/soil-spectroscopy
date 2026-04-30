"""Tests for the torch-based MeasuredPropertyEnsemble trainer + persistence."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from soilspec.groundtruth import GroundTruthDataset, Measurement, TileGrid
from soilspec.storage import StorageTier, StorageTierManager, model_key
from soilspec.training import (
    InsufficientTrainingDataError,
    MeasuredPropertyEnsemble,
    MeasuredPropertyEnsembleConfig,
    TrainingExamples,
    TrainingHistory,
    assemble_training_examples,
    load_model,
    save_model,
    train_ensemble,
)
from soilspec.types import BoundingBox, FusedRepresentation


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _grid() -> TileGrid:
    return TileGrid.from_shape(
        BoundingBox(0.0, 0.0, 1.0, 1.0), raster_shape=(32, 32), tile_size=16,
    )


def _direct_examples(
    X: np.ndarray, y_dict: dict[str, np.ndarray]
) -> TrainingExamples:
    """Build TrainingExamples directly from arrays for trainer tests."""
    n = X.shape[0]
    weights = {p: np.ones(n) for p in y_dict}
    return TrainingExamples(
        X=X.astype(np.float64),
        y={p: v.astype(np.float64) for p, v in y_dict.items()},
        weights=weights,
        tile_keys=tuple((f"r000c{i:03d}", 0) for i in range(n)),
        property_names=tuple(y_dict.keys()),
        sources=("synthetic",),
    )


# ---------------------------------------------------------------------------
# Training: learns a real signal
# ---------------------------------------------------------------------------


def test_train_ensemble_learns_linear_signal():
    rng = np.random.default_rng(0)
    n, d = 400, 8
    X = rng.normal(size=(n, d)).astype(np.float64)
    true_w = rng.normal(size=d) * 0.5
    y = X @ true_w + 0.05 * rng.normal(size=n)
    examples = _direct_examples(X, {"soil_moisture": y})

    cfg = MeasuredPropertyEnsembleConfig(
        properties=("soil_moisture",),
        epochs=80, batch_size=32, learning_rate=5e-3,
        val_fraction=0.0, seed=0,
    )
    history = TrainingHistory()
    model = train_ensemble(examples, cfg, history=history)

    pred = np.array([
        model.predict(X[i].astype(np.float32))["soil_moisture"]
        for i in range(n)
    ])
    # Loss should drop substantially during training.
    assert history.train_loss[-1] < history.train_loss[0] * 0.5
    # And predictions should correlate strongly with the true signal.
    assert np.corrcoef(pred, y)[0, 1] > 0.9


def test_train_ensemble_handles_missing_labels_per_row():
    """Each row supplies one of two properties; both heads still train."""
    rng = np.random.default_rng(7)
    n, d = 300, 6
    X = rng.normal(size=(n, d))
    w_a = rng.normal(size=d)
    w_b = rng.normal(size=d)
    y_a = X @ w_a
    y_b = X @ w_b

    # Mask: first half has only y_a, second half has only y_b
    half = n // 2
    y_a_masked = y_a.copy()
    y_a_masked[half:] = np.nan
    y_b_masked = y_b.copy()
    y_b_masked[:half] = np.nan
    examples = _direct_examples(X, {"soc": y_a_masked, "ph": y_b_masked})

    cfg = MeasuredPropertyEnsembleConfig(
        properties=("soc", "ph"),
        epochs=80, batch_size=32, learning_rate=5e-3,
        val_fraction=0.0, seed=1,
    )
    model = train_ensemble(examples, cfg)
    pred_a = np.array([model.predict(X[i].astype(np.float32))["soc"] for i in range(n)])
    pred_b = np.array([model.predict(X[i].astype(np.float32))["ph"] for i in range(n)])
    # Each property should track its true signal despite the half-half label split.
    assert np.corrcoef(pred_a, y_a)[0, 1] > 0.7
    assert np.corrcoef(pred_b, y_b)[0, 1] > 0.7


def test_train_ensemble_skips_property_with_no_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = {
        "soc": X @ np.array([1.0, 0.0, -0.5, 2.0]),
        "ph": np.full(60, np.nan),
    }
    examples = _direct_examples(X, y)
    cfg = MeasuredPropertyEnsembleConfig(
        properties=("soc", "ph"), epochs=10, batch_size=8,
        val_fraction=0.0, seed=0,
    )
    model = train_ensemble(examples, cfg)
    assert model.properties == ("soc",)


def test_train_ensemble_raises_when_no_usable_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = {"soc": np.full(20, np.nan)}
    examples = _direct_examples(X, y)
    cfg = MeasuredPropertyEnsembleConfig(
        properties=("soc",), epochs=2, val_fraction=0.0,
    )
    with pytest.raises(InsufficientTrainingDataError):
        train_ensemble(examples, cfg)


def test_train_ensemble_validation_loss_recorded():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = X @ np.array([0.5, -0.5, 1.0, 0.0])
    examples = _direct_examples(X, {"soc": y})
    cfg = MeasuredPropertyEnsembleConfig(
        properties=("soc",), epochs=20, batch_size=16,
        learning_rate=1e-2, val_fraction=0.25, seed=0,
    )
    history = TrainingHistory()
    train_ensemble(examples, cfg, history=history)
    assert len(history.val_loss) == 20
    assert all(np.isfinite(v) for v in history.val_loss)


# ---------------------------------------------------------------------------
# Predict shape + interface
# ---------------------------------------------------------------------------


def test_predict_accepts_fused_representation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    y = X @ np.array([1.0, -1.0, 0.5, 0.0, 0.2])
    examples = _direct_examples(X, {"soc": y})
    cfg = MeasuredPropertyEnsembleConfig(
        properties=("soc",), epochs=15, val_fraction=0.0, seed=0,
    )
    model = train_ensemble(examples, cfg)

    rep = FusedRepresentation(
        tile_id="r000c000", time=0,
        vector=X[0].astype(np.float32),
        channels={"all": slice(0, 5)},
        strategy="concat", degraded=False,
    )
    out = model.predict(rep)
    assert set(out.keys()) == {"soc"}
    assert isinstance(out["soc"], float)


def test_predict_shape_mismatch_raises():
    cfg = MeasuredPropertyEnsembleConfig(
        properties=("soc",), epochs=1, val_fraction=0.0,
    )
    model = MeasuredPropertyEnsemble(fused_dim=4, config=cfg)
    with pytest.raises(ValueError, match="expected vector of shape"):
        model.predict(np.zeros(7, dtype=np.float32))


# ---------------------------------------------------------------------------
# Persistence: save/load round-trip preserves predictions
# ---------------------------------------------------------------------------


def test_save_load_round_trip_preserves_predictions(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 6))
    y = X @ np.array([0.3, -0.7, 0.1, 0.5, -0.2, 0.4])
    examples = _direct_examples(X, {"soc": y})
    cfg = MeasuredPropertyEnsembleConfig(
        properties=("soc",), epochs=20, val_fraction=0.0, seed=42,
    )
    model = train_ensemble(examples, cfg)

    storage = StorageTierManager()
    save_model(storage, family="measured", version="0.1.0", model=model)
    assert storage.exists(StorageTier.MODEL, model_key("measured", "0.1.0"))

    loaded = load_model(storage, family="measured", version="0.1.0")
    assert loaded.properties == model.properties
    assert loaded.fused_dim == model.fused_dim

    # Predictions must match exactly (state_dict is float-precise).
    sample = X[0].astype(np.float32)
    a = model.predict(sample)["soc"]
    b = loaded.predict(sample)["soc"]
    assert a == pytest.approx(b, rel=1e-6, abs=1e-6)


def test_load_model_missing_raises(tmp_path):
    storage = StorageTierManager()
    with pytest.raises(KeyError):
        load_model(storage, family="measured", version="0.0.0")


# ---------------------------------------------------------------------------
# End-to-end: ISMN CSV -> dataset -> assemble -> train -> predict
# ---------------------------------------------------------------------------


def test_end_to_end_ismn_to_trained_ensemble(tmp_path):
    """Smoke-test the full path: real-shaped ISMN CSV through to a fitted
    model that can predict from a held-out fused representation."""
    from soilspec.groundtruth import ISMNAdapter
    from soilspec.types import AOI, TimeWindow

    aoi = AOI(aoi_id="t", bbox=BoundingBox(0.0, 0.0, 1.0, 1.0))
    window = TimeWindow(start=0, end=365 * 86400)
    grid = _grid()

    # Generate a synthetic dataset where a 4-d feature vector linearly
    # determines soil_moisture (true relationship the model should recover).
    rng = np.random.default_rng(123)
    n = 200
    X = rng.normal(size=(n, 4)).astype(np.float64)
    true_w = np.array([0.4, -0.2, 0.6, 0.1])
    sm = 0.3 + 0.05 * (X @ true_w)  # in a plausible 0-1 range

    # Write CSV that ISMNAdapter can read; one tile gets all measurements
    # (simplest aligned scenario).
    csv = tmp_path / "ismn.csv"
    header = ("network,station,lon,lat,timestamp,soil_moisture,"
              "depth_from,depth_to,qc_flag,soil_moisture_uncertainty")
    rows = [header]
    for i in range(n):
        rows.append(f"NET,STN{i},0.25,0.75,{i * 86400},{sm[i]:.6f},0,5,G,")
    csv.write_text("\n".join(rows) + "\n")

    ds = GroundTruthDataset(grid, time_bucket_seconds=86400)
    ds.extend(ISMNAdapter(csv_path=csv).fetch(aoi, window))

    fused_reps = [
        FusedRepresentation(
            tile_id="r000c000", time=i * 86400,
            vector=X[i], channels={"all": slice(0, 4)},
            strategy="concat", degraded=False,
        )
        for i in range(n)
    ]
    examples = assemble_training_examples(
        fused_reps, ds, property_names=("soil_moisture",), sources=("ismn",),
    )
    assert len(examples) == n

    cfg = MeasuredPropertyEnsembleConfig(
        properties=("soil_moisture",), epochs=120, batch_size=32,
        learning_rate=5e-3, val_fraction=0.0, seed=0,
    )
    model = train_ensemble(examples, cfg)

    # Predict for a fresh sample — should be close to the underlying linear
    # relationship that generated the labels.
    held = rng.normal(size=4).astype(np.float32)
    expected = 0.3 + 0.05 * float(held @ true_w)
    got = model.predict(held)["soil_moisture"]
    assert got == pytest.approx(expected, abs=0.05)

    # Persistence round-trip works inside the same flow.
    storage = StorageTierManager()
    save_model(storage, "measured", "smoke-1", model)
    loaded = load_model(storage, "measured", "smoke-1")
    assert loaded.predict(held)["soil_moisture"] == pytest.approx(got, rel=1e-6)
