"""Supervised training over fused multimodal representations.

Two trainers, sharing :class:`TrainingExamples`:

* :func:`train_ridge` — closed-form weighted ridge regression. Pure numpy,
  no torch dependency, fast, useful as a baseline.
* :func:`train_ensemble` — neural-net trainer (Lifting + MLP head) that
  fits :class:`MeasuredPropertyEnsemble` jointly across multiple
  measured properties via masked MSE loss.

Both consume the output of :func:`assemble_training_examples`, which joins
:class:`FusedRepresentation`s (features) with
:class:`GroundTruthSample`s (labels) on ``(tile_id, time_bucket)``.

Usage sketch::

    examples = assemble_training_examples(fused_reps, gt_dataset,
                                          sources=("ismn", "lucas"))

    # Linear baseline
    ridge_models = train_ridge(examples, alpha=1.0)

    # Neural ensemble
    ensemble = train_ensemble(examples, MeasuredPropertyEnsembleConfig(
        properties=("soil_moisture", "soc"), epochs=200,
    ))
    save_model(storage, "measured_v1", "0.1.0", ensemble)
    loaded = load_model(storage, "measured_v1", "0.1.0")
    pred = loaded.predict(new_fused_rep)  # -> {"soil_moisture": ..., "soc": ...}
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from .groundtruth import GroundTruthDataset
from .storage import StorageTier, StorageTierManager, model_key
from .types import (
    FusedRepresentation,
    GroundTruthSample,
    MEASURED_PROPERTY_NAMES,
)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingExamples:
    """Aligned feature/label rows for a set of (tile, bucket) joins.

    ``y[prop]`` is shape ``(N,)`` with NaN where that property was absent
    from the joining ground-truth sample — trainers drop NaN rows per
    property so a row missing one label can still supply others.

    ``weights[prop]`` carries 1/σ² inverse-variance weights when the
    ground-truth source reported per-sample uncertainty, else 1.0.
    """

    X: np.ndarray
    y: dict[str, np.ndarray]
    weights: dict[str, np.ndarray]
    tile_keys: tuple[tuple[str, int], ...]
    property_names: tuple[str, ...]
    sources: tuple[str, ...]

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def usable(self, prop: str) -> int:
        """Number of rows with a finite label for `prop`."""
        return int(np.sum(np.isfinite(self.y[prop])))


def assemble_training_examples(
    fused: Iterable[FusedRepresentation],
    gt: GroundTruthDataset,
    property_names: Iterable[str] = MEASURED_PROPERTY_NAMES,
    sources: Iterable[str] | None = None,
) -> TrainingExamples:
    """Join fused representations with ground-truth samples.

    Each fused rep's timestamp is bucketed using the dataset's own
    ``time_bucket_seconds`` so the join key matches the bucket-aligned
    ``time`` already on each :class:`GroundTruthSample`.

    Parameters
    ----------
    sources :
        Restrict to a subset of provider ids (e.g. ``("ismn", "lucas")``
        for label-only training; ``("soilgrids",)`` to extract covariates).
        ``None`` (default) keeps every source — note this will mix labels
        and static covariates, which is rarely what you want.
    """
    props = tuple(property_names)
    allowed_sources = set(sources) if sources is not None else None

    # Index GT samples by (tile_id, bucket_start). Multiple sources can land
    # on the same key — collapse by averaging across sources, weighted by
    # n_observations. (Ridge regression is fine with averaged labels; a
    # multi-source mixture model is a future concern.)
    bucket = gt.time_bucket_seconds
    by_key: dict[tuple[str, int], list[GroundTruthSample]] = {}
    sources_seen: set[str] = set()
    for s in gt.samples():
        if allowed_sources is not None and s.source not in allowed_sources:
            continue
        sources_seen.add(s.source)
        by_key.setdefault((s.tile_id, s.time), []).append(s)

    rows_X: list[np.ndarray] = []
    rows_y: dict[str, list[float]] = {p: [] for p in props}
    rows_w: dict[str, list[float]] = {p: [] for p in props}
    keys: list[tuple[str, int]] = []

    for rep in fused:
        bucket_start = (int(rep.time) // bucket) * bucket
        matched = by_key.get((rep.tile_id, bucket_start))
        if not matched:
            continue
        rows_X.append(np.asarray(rep.vector, dtype=np.float64))
        keys.append((rep.tile_id, bucket_start))
        for p in props:
            mean, weight = _aggregate_label(matched, p)
            rows_y[p].append(mean)
            rows_w[p].append(weight)

    if not rows_X:
        X = np.zeros((0, 0), dtype=np.float64)
    else:
        X = np.stack(rows_X, axis=0)
    y = {p: np.asarray(rows_y[p], dtype=np.float64) for p in props}
    w = {p: np.asarray(rows_w[p], dtype=np.float64) for p in props}
    return TrainingExamples(
        X=X,
        y=y,
        weights=w,
        tile_keys=tuple(keys),
        property_names=props,
        sources=tuple(sorted(sources_seen)),
    )


def _aggregate_label(
    samples: list[GroundTruthSample], prop: str
) -> tuple[float, float]:
    """Inverse-variance weighted mean across multiple samples for one prop.

    Returns ``(NaN, 0.0)`` if no sample carries this property.
    """
    weighted_sum = 0.0
    weight_sum = 0.0
    for s in samples:
        v = s.properties.get(prop)
        if v is None or not np.isfinite(v):
            continue
        unc = s.uncertainty.get(prop, 0.0)
        # Uncertainty=0 means "exact", which would dominate the weighting;
        # use n_observations as the weight in that case so denser tiles
        # carry more signal.
        if unc and np.isfinite(unc) and unc > 0:
            w = 1.0 / (unc * unc)
        else:
            w = float(max(s.n_observations, 1))
        weighted_sum += w * float(v)
        weight_sum += w
    if weight_sum == 0:
        return float("nan"), 0.0
    return weighted_sum / weight_sum, weight_sum


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RidgeModel:
    """Linear regression with L2 penalty. Pure numpy, no torch."""

    weights: np.ndarray   # (D,)
    intercept: float
    feature_mean: np.ndarray  # (D,) — for diagnostics / future scaling
    n_train: int
    alpha: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"feature dim mismatch: model expects {self.weights.shape[0]}, "
                f"got {X.shape[1]}"
            )
        return X @ self.weights + self.intercept


class InsufficientTrainingDataError(RuntimeError):
    """Raised when a property has fewer usable rows than features."""


def train_ridge(
    examples: TrainingExamples,
    alpha: float = 1.0,
    min_samples: int | None = None,
) -> dict[str, RidgeModel]:
    """Fit one ridge model per property, skipping properties with no labels.

    Closed-form solution::

        w = (Xᵀ W X + αI)⁻¹ Xᵀ W (y - ȳ)

    where ``W`` is the diagonal matrix of per-sample weights (inverse
    variance when available, n_observations otherwise) and the intercept
    absorbs the centering.

    Parameters
    ----------
    min_samples :
        Minimum usable rows required to fit a property. Defaults to
        ``n_features + 1`` to keep the system overdetermined.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if len(examples) == 0 or examples.n_features == 0:
        return {}
    D = examples.n_features
    floor = (D + 1) if min_samples is None else min_samples
    out: dict[str, RidgeModel] = {}
    for prop in examples.property_names:
        y = examples.y[prop]
        w = examples.weights[prop]
        mask = np.isfinite(y) & (w > 0)
        n = int(mask.sum())
        if n == 0:
            continue
        if n < floor:
            raise InsufficientTrainingDataError(
                f"{prop}: {n} usable samples < required {floor} "
                f"(features={D}). Either lower `min_samples`, add more "
                f"ground-truth coverage, or reduce feature dimensionality."
            )
        Xs = examples.X[mask]
        ys = y[mask]
        ws = w[mask]
        out[prop] = _fit_one(Xs, ys, ws, alpha)
    return out


def _fit_one(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float
) -> RidgeModel:
    # Weighted mean-centering — the intercept absorbs the means so we don't
    # need to penalize it.
    w = w / w.sum()
    x_mean = (w[:, None] * X).sum(axis=0)
    y_mean = float((w * y).sum())
    Xc = X - x_mean
    yc = y - y_mean
    W = np.diag(w)
    XtWX = Xc.T @ W @ Xc
    reg = alpha * np.eye(Xc.shape[1])
    weights = np.linalg.solve(XtWX + reg, Xc.T @ (w * yc))
    intercept = y_mean - float(x_mean @ weights)
    return RidgeModel(
        weights=weights,
        intercept=intercept,
        feature_mean=x_mean,
        n_train=int(X.shape[0]),
        alpha=float(alpha),
    )


# ---------------------------------------------------------------------------
# Neural ensemble (torch)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasuredPropertyEnsembleConfig:
    """Hyperparameters for :func:`train_ensemble`."""

    properties: tuple[str, ...] = MEASURED_PROPERTY_NAMES
    lifting_dim: int = 64
    hidden_dim: int = 64
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    seed: int = 0


@dataclass
class TrainingHistory:
    """Loss curves returned by :func:`train_ensemble`."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    epochs_trained: int = 0


class MeasuredPropertyEnsemble:
    """Trainable neural ensemble: Lifting (Tanh) + MLP head.

    Internal layout::

        fused (D_fused) -> Linear -> Tanh -> Linear -> Linear -> properties

    The head outputs *standardized* values; ``predict`` undoes the
    standardization using stats fit during training. Inputs and outputs
    are numpy at the API boundary — torch tensors never leak.

    Tied to a fixed property list at construction time; ``predict`` always
    emits exactly ``self.properties``.
    """

    def __init__(
        self,
        fused_dim: int,
        config: MeasuredPropertyEnsembleConfig,
        y_mean: np.ndarray | None = None,
        y_std: np.ndarray | None = None,
    ) -> None:
        import torch
        from torch import nn

        if not config.properties:
            raise ValueError("config.properties must be non-empty")
        self._torch = torch
        self.config = config
        self.fused_dim = int(fused_dim)
        self.properties = tuple(config.properties)
        self._n_props = len(self.properties)

        torch.manual_seed(int(config.seed))
        self._net = nn.Sequential(
            nn.Linear(self.fused_dim, config.lifting_dim),
            nn.Tanh(),
            nn.Linear(config.lifting_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, self._n_props),
        )

        # Output standardization stats — defaults are zero/one until fit.
        self._y_mean = (
            np.asarray(y_mean, dtype=np.float64) if y_mean is not None
            else np.zeros(self._n_props, dtype=np.float64)
        )
        self._y_std = (
            np.asarray(y_std, dtype=np.float64) if y_std is not None
            else np.ones(self._n_props, dtype=np.float64)
        )

    # ----------------------------- inference --------------------------------

    def predict(self, fused: FusedRepresentation | np.ndarray) -> dict[str, float]:
        """Predict measured properties for one fused representation."""
        if isinstance(fused, FusedRepresentation):
            vec = np.asarray(fused.vector, dtype=np.float32)
        else:
            vec = np.asarray(fused, dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != self.fused_dim:
            raise ValueError(
                f"expected vector of shape ({self.fused_dim},), got {vec.shape}"
            )
        torch = self._torch
        self._net.eval()
        with torch.no_grad():
            standardized = self._net(torch.from_numpy(vec[None, :])).cpu().numpy()[0]
        denorm = standardized * self._y_std + self._y_mean
        return {p: float(denorm[i]) for i, p in enumerate(self.properties)}

    def predict_batch(
        self, fused_reps: Iterable[FusedRepresentation]
    ) -> list[dict[str, float]]:
        return [self.predict(r) for r in fused_reps]

    # ----------------------------- persistence ------------------------------

    def to_state_dict(self) -> dict:
        """Serialize to a plain dict suitable for :func:`save_model`."""
        return {
            "schema_version": 1,
            "fused_dim": self.fused_dim,
            "properties": list(self.properties),
            "config": {
                "properties": list(self.config.properties),
                "lifting_dim": self.config.lifting_dim,
                "hidden_dim": self.config.hidden_dim,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "val_fraction": self.config.val_fraction,
                "seed": self.config.seed,
            },
            "y_mean": self._y_mean.tolist(),
            "y_std": self._y_std.tolist(),
            "net_state_dict": {k: v.detach().cpu()
                               for k, v in self._net.state_dict().items()},
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "MeasuredPropertyEnsemble":
        if d.get("schema_version") != 1:
            raise ValueError(
                f"unknown ensemble schema version: {d.get('schema_version')}"
            )
        cfg = MeasuredPropertyEnsembleConfig(
            properties=tuple(d["config"]["properties"]),
            lifting_dim=int(d["config"]["lifting_dim"]),
            hidden_dim=int(d["config"]["hidden_dim"]),
            epochs=int(d["config"]["epochs"]),
            batch_size=int(d["config"]["batch_size"]),
            learning_rate=float(d["config"]["learning_rate"]),
            weight_decay=float(d["config"]["weight_decay"]),
            val_fraction=float(d["config"]["val_fraction"]),
            seed=int(d["config"]["seed"]),
        )
        inst = cls(
            fused_dim=int(d["fused_dim"]),
            config=cfg,
            y_mean=np.asarray(d["y_mean"], dtype=np.float64),
            y_std=np.asarray(d["y_std"], dtype=np.float64),
        )
        inst._net.load_state_dict(d["net_state_dict"])
        return inst

    # ------------------------- internal training hooks ----------------------

    def _net_module(self):
        return self._net


def train_ensemble(
    examples: TrainingExamples,
    config: MeasuredPropertyEnsembleConfig | None = None,
    history: TrainingHistory | None = None,
) -> MeasuredPropertyEnsemble:
    """Fit a :class:`MeasuredPropertyEnsemble` on ``examples``.

    Loss: per-property weighted MSE in standardized output space, masked to
    rows where each label is finite. Properties are independent in the loss
    (no covariance term) but share the lifting + first hidden layers, so
    representations transfer across properties when many examples share
    only a subset of labels (typical in the ISMN+LUCAS mix).

    Caller can pass a :class:`TrainingHistory` to receive epoch-by-epoch
    train/val loss curves for diagnostics.
    """
    import torch

    cfg = config or MeasuredPropertyEnsembleConfig()
    if not cfg.properties:
        raise ValueError("config.properties must be non-empty")
    if examples.n_features == 0 or len(examples) == 0:
        raise InsufficientTrainingDataError("no training examples available")

    # Restrict to properties that exist in `examples` *and* have at least
    # some usable labels — otherwise we'd compute gradients against a
    # column with zero mask and propagate NaNs.
    target_props: list[str] = []
    for p in cfg.properties:
        if p not in examples.y:
            continue
        if examples.usable(p) == 0:
            continue
        target_props.append(p)
    if not target_props:
        raise InsufficientTrainingDataError(
            f"no usable labels for any of {cfg.properties}"
        )
    cfg = MeasuredPropertyEnsembleConfig(
        properties=tuple(target_props),
        lifting_dim=cfg.lifting_dim, hidden_dim=cfg.hidden_dim,
        epochs=cfg.epochs, batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay,
        val_fraction=cfg.val_fraction, seed=cfg.seed,
    )

    X = examples.X.astype(np.float64)
    y = np.stack([examples.y[p] for p in cfg.properties], axis=1)
    w = np.stack([examples.weights[p] for p in cfg.properties], axis=1)
    mask = np.isfinite(y) & (w > 0)

    # Standardize per-property on observed entries only.
    y_mean = np.zeros(len(cfg.properties))
    y_std = np.ones(len(cfg.properties))
    for j in range(y.shape[1]):
        col_mask = mask[:, j]
        vals = y[col_mask, j]
        if vals.size == 0:
            continue
        y_mean[j] = float(vals.mean())
        std = float(vals.std())
        y_std[j] = std if std > 1e-8 else 1.0

    y_std_safe = y_std.copy()
    y_std_safe[y_std_safe == 0] = 1.0
    y_norm = (y - y_mean[None, :]) / y_std_safe[None, :]
    y_norm = np.where(mask, y_norm, 0.0)  # masked rows zeroed out

    rng = np.random.default_rng(cfg.seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    val_n = int(round(n * cfg.val_fraction))
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]
    if tr_idx.size == 0:
        tr_idx = perm
        val_idx = np.empty(0, dtype=np.int64)

    model = MeasuredPropertyEnsemble(
        fused_dim=examples.n_features, config=cfg,
        y_mean=y_mean, y_std=y_std_safe,
    )
    net = model._net_module()
    net.train()
    opt = torch.optim.Adam(
        net.parameters(),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y_norm.astype(np.float32))
    w_t = torch.from_numpy(w.astype(np.float32))
    m_t = torch.from_numpy(mask.astype(np.float32))

    hist = history or TrainingHistory()
    batch = max(1, int(cfg.batch_size))
    for epoch in range(cfg.epochs):
        net.train()
        epoch_loss = 0.0
        epoch_count = 0
        # Mini-batch over training rows
        rng_epoch = np.random.default_rng(cfg.seed + 1 + epoch)
        order = rng_epoch.permutation(tr_idx.size)
        for start in range(0, tr_idx.size, batch):
            chunk = tr_idx[order[start:start + batch]]
            opt.zero_grad()
            pred = net(X_t[chunk])
            diff = pred - y_t[chunk]
            sq = diff * diff
            weighted = sq * w_t[chunk] * m_t[chunk]
            denom = (w_t[chunk] * m_t[chunk]).sum().clamp_min(1e-9)
            loss = weighted.sum() / denom
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * chunk.size
            epoch_count += int(chunk.size)
        train_loss = epoch_loss / max(epoch_count, 1)
        hist.train_loss.append(train_loss)

        if val_idx.size:
            net.eval()
            with torch.no_grad():
                pred = net(X_t[val_idx])
                diff = pred - y_t[val_idx]
                weighted = diff * diff * w_t[val_idx] * m_t[val_idx]
                denom = (w_t[val_idx] * m_t[val_idx]).sum().clamp_min(1e-9)
                hist.val_loss.append(float((weighted.sum() / denom).item()))
        else:
            hist.val_loss.append(float("nan"))
        hist.epochs_trained = epoch + 1

    net.eval()
    return model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(
    storage: StorageTierManager,
    family: str,
    version: str,
    model: MeasuredPropertyEnsemble,
) -> str:
    """Serialize ``model`` and put it on the MODEL tier. Returns the key."""
    import torch

    buf = io.BytesIO()
    torch.save(model.to_state_dict(), buf)
    key = model_key(family, version)
    storage.put(StorageTier.MODEL, key, buf.getvalue())
    return key


def load_model(
    storage: StorageTierManager, family: str, version: str,
) -> MeasuredPropertyEnsemble:
    """Inverse of :func:`save_model`."""
    import torch

    key = model_key(family, version)
    blob = storage.get(StorageTier.MODEL, key)
    if isinstance(blob, (bytes, bytearray)):
        buf = io.BytesIO(bytes(blob))
        # weights_only=False: we control both ends and need to restore the
        # numpy stats embedded in the dict.
        d = torch.load(buf, weights_only=False)
    else:
        # In-memory test backend can return the dict directly.
        d = blob
    return MeasuredPropertyEnsemble.from_state_dict(d)


# ===========================================================================
# Joint end-to-end pipeline training
# ===========================================================================
#
# The MeasuredPropertyEnsemble above trains a head ON TOP OF the encoder+
# fusion stack treated as a frozen random-init feature extractor. The
# `train_pipeline` flow below trains the WHOLE pipeline jointly: encoders,
# fusion, and head all backprop together.


from .encoders import SpatialEncoderRegistry, SpectralEncoderRegistry
from .encoders.spatial import SpatialEncoder
from .encoders.spectral import SpectralEncoder
from .fusion import FusionConfig, FusionStrategyRegistry
from .groundtruth import RasterTrainingExamples


@dataclass(frozen=True)
class PipelineTrainerConfig:
    """Hyperparameters for :func:`train_pipeline`."""

    properties: tuple[str, ...] = MEASURED_PROPERTY_NAMES
    spectral_backend: str = "1d_cnn"
    spatial_backend: str = "cnn"
    spectral_latent_dim: int = 32
    spatial_latent_dim: int = 32
    fusion_strategy: str = "concat"
    fusion_output_dim: int = 48
    head_hidden_dim: int = 64
    dropout: float = 0.1
    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    gradient_clip: float = 1.0
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    early_stopping_patience: int | None = 25
    warmup_epochs: int = 5
    seed: int = 0
    text_projection_dim: int = 32
    """Output width of the trainable Linear projection applied to the frozen
    text embedding. Ignored when no text features are supplied."""


@dataclass
class Metrics:
    """Per-property eval scores."""

    rmse: float
    mae: float
    r2: float
    n: int
    baseline_rmse: float


@dataclass
class _Splits:
    train: list[int]
    val: list[int]
    test: list[int]


def split_by_tile(
    examples: RasterTrainingExamples,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> _Splits:
    """Spatially-blocked split: every row of a given tile_id lands in one partition.

    Prevents leakage from temporal autocorrelation within a tile and from
    spatial autocorrelation across nearby pixels (which a true random row
    split would cheerfully feed into both train and val).
    """
    rng = np.random.default_rng(int(seed))
    unique_tiles = sorted({k[0] for k in examples.tile_keys})
    perm = list(unique_tiles)
    rng.shuffle(perm)
    n = len(perm)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test_tiles = set(perm[:n_test])
    val_tiles = set(perm[n_test : n_test + n_val])
    train_tiles = set(perm[n_test + n_val :])

    train, val, test = [], [], []
    for i, (tile_id, _) in enumerate(examples.tile_keys):
        if tile_id in train_tiles:
            train.append(i)
        elif tile_id in val_tiles:
            val.append(i)
        elif tile_id in test_tiles:
            test.append(i)
    return _Splits(train=train, val=val, test=test)


class _PipelineTrainable:
    """Wraps encoders + fusion + head as a single trainable graph.

    Not an ``nn.Module`` — the encoder/fusion classes don't inherit from one
    either; they expose ``parameters()``/``state_dict()``/``forward_torch``
    via duck typing. Treating this composite the same way keeps everything
    consistent.
    """

    def __init__(
        self,
        cfg: PipelineTrainerConfig,
        s1_bands: int,
        s2_bands: int,
        vector_dim: int,
        n_props: int,
        text_dim: int = 0,
    ) -> None:
        import torch
        from torch import nn

        self._torch = torch
        self.cfg = cfg
        self.n_props = n_props
        self.text_dim = int(text_dim)
        self.text_projection_dim = int(cfg.text_projection_dim)
        torch.manual_seed(int(cfg.seed))

        self.spatial_enc: SpatialEncoder = SpatialEncoderRegistry.create(
            cfg.spatial_backend,
            latent_dim=cfg.spatial_latent_dim,
            patch_size=8, stride=4, context_patches=0, seed=cfg.seed,
        )
        self.spectral_enc: SpectralEncoder = SpectralEncoderRegistry.create(
            cfg.spectral_backend,
            latent_dim=cfg.spectral_latent_dim,
            seed=cfg.seed,
        )
        # Force-build encoders so their parameters exist before we collect them.
        self.spatial_enc.build(s1_bands)  # type: ignore[attr-defined]
        self.spectral_enc.build(s2_bands)  # type: ignore[attr-defined]

        self.fusion = FusionStrategyRegistry.create(
            cfg.fusion_strategy, output_dim=cfg.fusion_output_dim,
        )
        # Force-build fusion: spec_dim and spat_dim are the encoder latents.
        self.fusion.build(  # type: ignore[attr-defined]
            cfg.spectral_latent_dim, cfg.spatial_latent_dim,
        )

        in_head = cfg.fusion_output_dim + int(vector_dim)
        # Trainable text projection: Linear(text_dim, P) + LayerNorm. The
        # upstream encoder is frozen — only this small head learns from
        # joint signal. Skipped entirely when text_dim == 0 so the trainer
        # behaves identically to the legacy two-modality model.
        if self.text_dim > 0:
            self.text_projection = nn.Sequential(
                nn.Linear(self.text_dim, self.text_projection_dim),
                nn.LayerNorm(self.text_projection_dim),
            )
            # +1 for the missingness scalar concatenated alongside the
            # projected vector so the head can learn to discount missing rows.
            in_head += self.text_projection_dim + 1
        else:
            self.text_projection = None
        self.head = nn.Sequential(
            nn.Linear(in_head, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, n_props),
        )

    # ---- composition ----

    def forward(self, s1_t, s2_t, vec_t, text_t=None, text_missing_t=None):
        spat = self.spatial_enc.forward_torch(s1_t)  # type: ignore[attr-defined]
        spec = self.spectral_enc.forward_torch(s2_t)  # type: ignore[attr-defined]
        fused = self.fusion.fuse_tensors(spec, spat)  # type: ignore[attr-defined]
        parts = [fused]
        if vec_t is not None and vec_t.shape[1] > 0:
            parts.append(vec_t)
        if self.text_projection is not None:
            if text_t is None:
                # Inference-time fallback: zero vector + "missing" flag set.
                text_t = self._torch.zeros(
                    fused.shape[0], self.text_dim, dtype=fused.dtype,
                )
                text_missing_t = self._torch.ones(
                    fused.shape[0], 1, dtype=fused.dtype,
                )
            projected = self.text_projection(text_t)
            if text_missing_t is None:
                text_missing_t = self._torch.zeros(
                    text_t.shape[0], 1, dtype=text_t.dtype,
                )
            elif text_missing_t.dim() == 1:
                text_missing_t = text_missing_t.view(-1, 1)
            parts.append(projected)
            parts.append(text_missing_t.to(projected.dtype))
        combined = self._torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.head(combined)

    def parameters(self) -> Iterable:
        from itertools import chain
        params = [
            self.spatial_enc.parameters(),  # type: ignore[attr-defined]
            self.spectral_enc.parameters(),  # type: ignore[attr-defined]
            self.fusion.parameters(),  # type: ignore[attr-defined]
            self.head.parameters(),
        ]
        if self.text_projection is not None:
            params.append(self.text_projection.parameters())
        return chain(*params)

    def train(self) -> None:
        self.head.train()
        if self.text_projection is not None:
            self.text_projection.train()
        for mod in self._sub_nn_modules():
            mod.train()

    def eval(self) -> None:
        self.head.eval()
        if self.text_projection is not None:
            self.text_projection.eval()
        for mod in self._sub_nn_modules():
            mod.eval()

    def _sub_nn_modules(self) -> list:
        out: list = []
        for owner in (self.spatial_enc, self.spectral_enc, self.fusion):
            sd = owner.state_dict()  # type: ignore[attr-defined]
            for k, v in sd.items():
                mod = getattr(owner, k, None)
                if mod is None:
                    # Some encoders use private attribute names like '_net'
                    mod = getattr(owner, "_" + k, None)
                if mod is not None and hasattr(mod, "train"):
                    out.append(mod)
        return out


class TrainedPipeline:
    """Public handle for a trained pipeline. Round-trippable via state_dict.

    Use :meth:`predict` (single example, numpy-friendly) or
    :meth:`predict_batch` for inference.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        cfg: PipelineTrainerConfig,
        s1_bands: int,
        s2_bands: int,
        vector_dim: int,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        vector_attr_names: tuple[str, ...] = (),
        text_dim: int = 0,
        text_encoder: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.s1_bands = int(s1_bands)
        self.s2_bands = int(s2_bands)
        self.vector_dim = int(vector_dim)
        self.vector_attr_names = tuple(vector_attr_names)
        self.properties = tuple(cfg.properties)
        self.text_dim = int(text_dim)
        self.text_encoder = text_encoder
        self._y_mean = np.asarray(y_mean, dtype=np.float64).copy()
        self._y_std = np.asarray(y_std, dtype=np.float64).copy()
        self._trainable = _PipelineTrainable(
            cfg, s1_bands, s2_bands, vector_dim, n_props=len(self.properties),
            text_dim=self.text_dim,
        )

    # ----------------------- inference -----------------------------------

    def predict(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        vector_features: np.ndarray | None = None,
        text_features: np.ndarray | None = None,
        text_missing: float | np.ndarray | None = None,
    ) -> dict[str, float]:
        """Predict measured properties for one tile's rasters.

        ``text_features`` is optional even on text-aware models — passing
        ``None`` sets the missingness flag and feeds a zero vector through
        the projection (graceful degradation matching training-time
        handling of unaligned rows).
        """
        import torch

        s1_t = torch.from_numpy(np.asarray(s1, dtype=np.float32))
        s2_t = torch.from_numpy(np.asarray(s2, dtype=np.float32))
        if vector_features is None:
            vec_t = torch.zeros(1, self.vector_dim, dtype=torch.float32)
        else:
            vec = np.asarray(vector_features, dtype=np.float32)
            if vec.ndim == 1:
                vec = vec[None, :]
            vec_t = torch.from_numpy(vec)
        # ensure batched
        if s1_t.dim() == 3:
            s1_t = s1_t.unsqueeze(0)
        if s2_t.dim() == 3:
            s2_t = s2_t.unsqueeze(0)
        text_t, missing_t = self._prepare_text_tensors(
            text_features, text_missing, batch_size=1,
        )
        self._trainable.eval()
        with torch.no_grad():
            standardized = self._trainable.forward(
                s1_t, s2_t, vec_t, text_t, missing_t,
            ).cpu().numpy()[0]
        denorm = standardized * self._y_std + self._y_mean
        return {p: float(denorm[i]) for i, p in enumerate(self.properties)}

    def predict_batch(
        self, s1: np.ndarray, s2: np.ndarray,
        vector_features: np.ndarray | None = None,
        text_features: np.ndarray | None = None,
        text_missing: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict for a batch — returns shape ``(N, n_props)`` in original units."""
        import torch

        s1_t = torch.from_numpy(np.asarray(s1, dtype=np.float32))
        s2_t = torch.from_numpy(np.asarray(s2, dtype=np.float32))
        n = int(s1_t.shape[0])
        if vector_features is None:
            vec_t = torch.zeros(n, self.vector_dim, dtype=torch.float32)
        else:
            vec_t = torch.from_numpy(np.asarray(vector_features, dtype=np.float32))
        text_t, missing_t = self._prepare_text_tensors(
            text_features, text_missing, batch_size=n,
        )
        self._trainable.eval()
        with torch.no_grad():
            std = self._trainable.forward(
                s1_t, s2_t, vec_t, text_t, missing_t,
            ).cpu().numpy()
        return std * self._y_std[None, :] + self._y_mean[None, :]

    def _prepare_text_tensors(
        self,
        text_features: np.ndarray | None,
        text_missing: float | np.ndarray | None,
        batch_size: int,
    ):
        """Build (text_t, missing_t) for forward(), or (None, None) for non-text models."""
        import torch
        if self.text_dim == 0:
            return None, None
        if text_features is None:
            text_t = torch.zeros(batch_size, self.text_dim, dtype=torch.float32)
            missing_t = torch.ones(batch_size, 1, dtype=torch.float32)
            return text_t, missing_t
        tx = np.asarray(text_features, dtype=np.float32)
        if tx.ndim == 1:
            tx = tx[None, :]
        if tx.shape[1] != self.text_dim:
            raise ValueError(
                f"text_features dim {tx.shape[1]} != model text_dim {self.text_dim}"
            )
        text_t = torch.from_numpy(tx)
        if text_missing is None:
            missing_arr = np.zeros((tx.shape[0],), dtype=np.float32)
        else:
            missing_arr = np.asarray(text_missing, dtype=np.float32).reshape(-1)
            if missing_arr.size == 1 and tx.shape[0] > 1:
                missing_arr = np.broadcast_to(missing_arr, (tx.shape[0],)).copy()
        missing_t = torch.from_numpy(missing_arr.reshape(-1, 1))
        return text_t, missing_t

    # ----------------------- persistence ---------------------------------

    def to_state_dict(self) -> dict:
        out = {
            "schema_version": self.SCHEMA_VERSION,
            "kind": "trained_pipeline",
            "cfg": _trainer_cfg_to_dict(self.cfg),
            "s1_bands": self.s1_bands,
            "s2_bands": self.s2_bands,
            "vector_dim": self.vector_dim,
            "vector_attr_names": list(self.vector_attr_names),
            "y_mean": self._y_mean.tolist(),
            "y_std": self._y_std.tolist(),
            "spatial": self._trainable.spatial_enc.state_dict(),  # type: ignore[attr-defined]
            "spectral": self._trainable.spectral_enc.state_dict(),  # type: ignore[attr-defined]
            "fusion": self._trainable.fusion.state_dict(),  # type: ignore[attr-defined]
            "head": self._trainable.head.state_dict(),
            "text_dim": self.text_dim,
            "text_encoder": self.text_encoder,
        }
        if self._trainable.text_projection is not None:
            out["text_projection"] = self._trainable.text_projection.state_dict()
        return out

    @classmethod
    def from_state_dict(cls, d: dict) -> "TrainedPipeline":
        if d.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unknown TrainedPipeline schema: {d.get('schema_version')}"
            )
        cfg = _trainer_cfg_from_dict(d["cfg"])
        inst = cls(
            cfg=cfg,
            s1_bands=int(d["s1_bands"]),
            s2_bands=int(d["s2_bands"]),
            vector_dim=int(d["vector_dim"]),
            y_mean=np.asarray(d["y_mean"], dtype=np.float64),
            y_std=np.asarray(d["y_std"], dtype=np.float64),
            vector_attr_names=tuple(d.get("vector_attr_names", ())),
            text_dim=int(d.get("text_dim", 0) or 0),
            text_encoder=d.get("text_encoder"),
        )
        inst._trainable.spatial_enc.load_state_dict(d["spatial"])  # type: ignore[attr-defined]
        inst._trainable.spectral_enc.load_state_dict(d["spectral"])  # type: ignore[attr-defined]
        inst._trainable.fusion.load_state_dict(d["fusion"])  # type: ignore[attr-defined]
        inst._trainable.head.load_state_dict(d["head"])
        if inst._trainable.text_projection is not None and "text_projection" in d:
            inst._trainable.text_projection.load_state_dict(d["text_projection"])
        return inst


def _trainer_cfg_to_dict(cfg: PipelineTrainerConfig) -> dict:
    return {
        "properties": list(cfg.properties),
        "spectral_backend": cfg.spectral_backend,
        "spatial_backend": cfg.spatial_backend,
        "spectral_latent_dim": cfg.spectral_latent_dim,
        "spatial_latent_dim": cfg.spatial_latent_dim,
        "fusion_strategy": cfg.fusion_strategy,
        "fusion_output_dim": cfg.fusion_output_dim,
        "head_hidden_dim": cfg.head_hidden_dim,
        "dropout": cfg.dropout,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "gradient_clip": cfg.gradient_clip,
        "val_fraction": cfg.val_fraction,
        "test_fraction": cfg.test_fraction,
        "early_stopping_patience": cfg.early_stopping_patience,
        "warmup_epochs": cfg.warmup_epochs,
        "seed": cfg.seed,
        "text_projection_dim": cfg.text_projection_dim,
    }


def _trainer_cfg_from_dict(d: dict) -> PipelineTrainerConfig:
    return PipelineTrainerConfig(
        properties=tuple(d["properties"]),
        spectral_backend=d["spectral_backend"],
        spatial_backend=d["spatial_backend"],
        spectral_latent_dim=int(d["spectral_latent_dim"]),
        spatial_latent_dim=int(d["spatial_latent_dim"]),
        fusion_strategy=d["fusion_strategy"],
        fusion_output_dim=int(d["fusion_output_dim"]),
        head_hidden_dim=int(d["head_hidden_dim"]),
        dropout=float(d["dropout"]),
        epochs=int(d["epochs"]),
        batch_size=int(d["batch_size"]),
        learning_rate=float(d["learning_rate"]),
        weight_decay=float(d["weight_decay"]),
        gradient_clip=float(d["gradient_clip"]),
        val_fraction=float(d["val_fraction"]),
        test_fraction=float(d["test_fraction"]),
        early_stopping_patience=(
            int(d["early_stopping_patience"])
            if d.get("early_stopping_patience") is not None else None
        ),
        warmup_epochs=int(d["warmup_epochs"]),
        seed=int(d["seed"]),
        text_projection_dim=int(d.get("text_projection_dim", 32)),
    )


def train_pipeline(
    examples: RasterTrainingExamples,
    cfg: PipelineTrainerConfig | None = None,
    history: TrainingHistory | None = None,
) -> tuple[TrainedPipeline, dict[str, Metrics], _Splits]:
    """Fit the full pipeline jointly. Returns (model, test_metrics, splits).

    Loss: per-property weighted MSE in standardized output space, masked to
    rows where each label is finite. Spatial-block split, AdamW, cosine
    LR with warmup, gradient clipping, early stopping on val loss.
    """
    import torch

    cfg = cfg or PipelineTrainerConfig()
    if len(examples) == 0:
        raise InsufficientTrainingDataError("no raster examples available")

    # Restrict to properties that have ANY usable label.
    target_props: list[str] = [
        p for p in cfg.properties
        if p in examples.y and examples.usable(p) > 0
    ]
    if not target_props:
        raise InsufficientTrainingDataError(
            f"no usable labels among configured properties: {cfg.properties}"
        )
    if list(target_props) != list(cfg.properties):
        cfg = PipelineTrainerConfig(
            **{**_trainer_cfg_to_dict(cfg), "properties": list(target_props)}
        )

    # ---- splits ----
    splits = split_by_tile(
        examples, cfg.val_fraction, cfg.test_fraction, cfg.seed,
    )
    if not splits.train:
        raise InsufficientTrainingDataError(
            "spatial split left zero training rows — reduce val/test fractions "
            "or add more tiles"
        )

    # ---- standardization stats (computed on train rows only) ----
    n_props = len(cfg.properties)
    y = np.stack([examples.y[p] for p in cfg.properties], axis=1)
    w = np.stack([examples.weights[p] for p in cfg.properties], axis=1)
    mask = np.isfinite(y) & (w > 0)
    y_mean = np.zeros(n_props)
    y_std = np.ones(n_props)
    train_idx_arr = np.asarray(splits.train, dtype=np.int64)
    for j in range(n_props):
        col_mask = mask[train_idx_arr, j]
        vals = y[train_idx_arr, j][col_mask]
        if vals.size == 0:
            continue
        y_mean[j] = float(vals.mean())
        std = float(vals.std())
        y_std[j] = std if std > 1e-8 else 1.0
    y_norm = (y - y_mean[None, :]) / y_std[None, :]
    y_norm = np.where(mask, y_norm, 0.0)

    # ---- model ----
    model = TrainedPipeline(
        cfg=cfg,
        s1_bands=int(examples.s1_bands),
        s2_bands=int(examples.s2_bands),
        vector_dim=int(examples.n_features_vector),
        y_mean=y_mean, y_std=y_std,
        vector_attr_names=examples.vector_attr_names,
        text_dim=int(examples.text_dim),
        text_encoder=examples.text_encoder,
    )
    trainable = model._trainable
    params = list(trainable.parameters())
    opt = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    # cosine schedule with linear warmup
    total_steps = max(1, cfg.epochs)
    def lr_at(epoch: int) -> float:
        if epoch < cfg.warmup_epochs and cfg.warmup_epochs > 0:
            return cfg.learning_rate * (epoch + 1) / cfg.warmup_epochs
        progress = (epoch - cfg.warmup_epochs) / max(
            1, total_steps - cfg.warmup_epochs
        )
        return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    # ---- tensors ----
    s1_t = torch.from_numpy(examples.s1.astype(np.float32))
    s2_t = torch.from_numpy(examples.s2.astype(np.float32))
    if examples.n_features_vector > 0:
        vec_t = torch.from_numpy(examples.vector_features.astype(np.float32))
    else:
        vec_t = torch.zeros(len(examples), 0, dtype=torch.float32)
    if examples.text_dim > 0:
        text_t = torch.from_numpy(examples.text_features.astype(np.float32))
        text_missing_t = torch.from_numpy(
            examples.text_missing.astype(np.float32)
        ).view(-1, 1)
    else:
        text_t = None
        text_missing_t = None
    y_t = torch.from_numpy(y_norm.astype(np.float32))
    w_t = torch.from_numpy(w.astype(np.float32))
    m_t = torch.from_numpy(mask.astype(np.float32))

    hist = history or TrainingHistory()
    best_val = float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0
    rng = np.random.default_rng(cfg.seed)

    for epoch in range(cfg.epochs):
        # set LR
        lr_now = lr_at(epoch)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        # train epoch
        trainable.train()
        order = list(splits.train)
        rng.shuffle(order)
        epoch_loss, epoch_count = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            chunk = order[start : start + cfg.batch_size]
            opt.zero_grad()
            text_chunk = text_t[chunk] if text_t is not None else None
            missing_chunk = (
                text_missing_t[chunk] if text_missing_t is not None else None
            )
            pred = trainable.forward(
                s1_t[chunk], s2_t[chunk], vec_t[chunk],
                text_chunk, missing_chunk,
            )
            diff = pred - y_t[chunk]
            sq = diff * diff
            weighted = sq * w_t[chunk] * m_t[chunk]
            denom = (w_t[chunk] * m_t[chunk]).sum().clamp_min(1e-9)
            loss = weighted.sum() / denom
            loss.backward()
            if cfg.gradient_clip and cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.gradient_clip)
            opt.step()
            epoch_loss += float(loss.item()) * len(chunk)
            epoch_count += len(chunk)
        train_loss = epoch_loss / max(epoch_count, 1)
        hist.train_loss.append(train_loss)

        # val epoch
        if splits.val:
            trainable.eval()
            with torch.no_grad():
                idx = splits.val
                text_val = text_t[idx] if text_t is not None else None
                missing_val = (
                    text_missing_t[idx] if text_missing_t is not None else None
                )
                pred = trainable.forward(
                    s1_t[idx], s2_t[idx], vec_t[idx], text_val, missing_val,
                )
                diff = pred - y_t[idx]
                weighted = diff * diff * w_t[idx] * m_t[idx]
                denom = (w_t[idx] * m_t[idx]).sum().clamp_min(1e-9)
                val_loss = float((weighted.sum() / denom).item())
            hist.val_loss.append(val_loss)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = model.to_state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if (cfg.early_stopping_patience is not None
                        and epochs_no_improve >= cfg.early_stopping_patience):
                    hist.epochs_trained = epoch + 1
                    break
        else:
            hist.val_loss.append(float("nan"))
        hist.epochs_trained = epoch + 1

    # restore best-val state if we tracked one
    if best_state is not None:
        model = TrainedPipeline.from_state_dict(best_state)

    # ---- test metrics ----
    metrics = evaluate(model, examples, indices=splits.test) if splits.test else {}
    return model, metrics, splits


def evaluate(
    model: TrainedPipeline,
    examples: RasterTrainingExamples,
    indices: list[int] | None = None,
) -> dict[str, Metrics]:
    """Per-property R²/RMSE/MAE on the given subset (defaults to all rows)."""
    idx = list(indices) if indices is not None else list(range(len(examples)))
    if not idx:
        return {}
    s1 = examples.s1[idx]
    s2 = examples.s2[idx]
    vec = (
        examples.vector_features[idx]
        if examples.n_features_vector > 0 else None
    )
    if examples.text_dim > 0:
        text = examples.text_features[idx]
        missing = examples.text_missing[idx]
    else:
        text = None
        missing = None
    pred = model.predict_batch(s1, s2, vec, text, missing)  # (N, n_props)
    out: dict[str, Metrics] = {}
    for i, p in enumerate(model.properties):
        y = examples.y[p][idx]
        m = np.isfinite(y)
        if not m.any():
            continue
        yt = y[m]
        yp = pred[m, i]
        residual = yp - yt
        rmse = float(np.sqrt(np.mean(residual * residual)))
        mae = float(np.mean(np.abs(residual)))
        baseline_pred = float(np.mean(yt))
        baseline_residual = baseline_pred - yt
        baseline_rmse = float(np.sqrt(np.mean(baseline_residual * baseline_residual)))
        ss_res = float(np.sum(residual * residual))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        out[p] = Metrics(
            rmse=rmse, mae=mae, r2=r2, n=int(m.sum()),
            baseline_rmse=baseline_rmse,
        )
    return out


# Add math import for cosine schedule.
import math  # noqa: E402  (used by lr_at above)


# ---------------------------------------------------------------------------
# Pipeline persistence (separate functions to keep return types unambiguous)
# ---------------------------------------------------------------------------


def save_pipeline(
    storage: StorageTierManager,
    family: str,
    version: str,
    model: TrainedPipeline,
) -> str:
    """Serialize a :class:`TrainedPipeline` to ``StorageTier.MODEL``."""
    import torch

    buf = io.BytesIO()
    torch.save(model.to_state_dict(), buf)
    key = model_key(family, version)
    storage.put(StorageTier.MODEL, key, buf.getvalue())
    return key


def load_pipeline(
    storage: StorageTierManager, family: str, version: str,
) -> TrainedPipeline:
    """Inverse of :func:`save_pipeline`."""
    import torch

    key = model_key(family, version)
    blob = storage.get(StorageTier.MODEL, key)
    if isinstance(blob, (bytes, bytearray)):
        buf = io.BytesIO(bytes(blob))
        d = torch.load(buf, weights_only=False)
    else:
        d = blob
    if d.get("kind") != "trained_pipeline":
        raise ValueError(
            f"key {family}/{version} is not a trained pipeline "
            f"(got kind={d.get('kind')!r})"
        )
    return TrainedPipeline.from_state_dict(d)


__all__ = [
    "TrainingExamples",
    "assemble_training_examples",
    "RidgeModel",
    "InsufficientTrainingDataError",
    "train_ridge",
    "MeasuredPropertyEnsemble",
    "MeasuredPropertyEnsembleConfig",
    "TrainingHistory",
    "train_ensemble",
    "save_model",
    "load_model",
    "PipelineTrainerConfig",
    "TrainedPipeline",
    "Metrics",
    "split_by_tile",
    "train_pipeline",
    "evaluate",
    "save_pipeline",
    "load_pipeline",
]
