"""Train the joint pipeline (encoders + fusion + head) on a built dataset.

Loads a :class:`RasterTrainingExamples` pickle (produced by
:mod:`soilspec.datasets.build`), runs :func:`soilspec.training.train_pipeline`
with a config tuned for the small initial dataset, persists the resulting
:class:`TrainedPipeline` to disk, and prints per-property test metrics.

The training config here is deliberately small / regularised — 292 rows
spread across ~24 AOIs is not enough to support a big model. Goal of this
first cut: verify the joint pipeline learns *anything* on real data;
publish numbers as a baseline for the retry/expansion follow-on.

PRODUCTION NOTE — chemistry heads are training-only auxiliaries
----------------------------------------------------------------
On the v2 dataset (521 rows, 45 AOIs) only the **soil_moisture** head beats
baseline on a spatially-blocked test split (R²=+0.082). SOC / N / P / K /
ph heads do not generalise across AOIs at this dataset size; S1+S2 alone
appears to carry too little chemistry signal for spatial-block extrapolation
to work.

That said, training with all six properties produces a *better*
soil_moisture head than training with soil_moisture alone (R²=+0.082 vs
−0.014 on the same held-out tiles, same seed). The chemistry rows act as
auxiliary multi-task regularisation on the shared encoders — they expand
the feature distribution the encoders see and prevent overfitting on the
93-row soil_moisture subset. Don't drop them from training.

For inference, treat only soil_moisture as a production output. The
chemistry heads can be left in the model artifact but should not be
surfaced to downstream consumers until per-source dataset coverage is
large enough to make those heads generalise.

Run::

    python -m soilspec.datasets.train_initial \\
        --dataset data/trainsets/initial_v2.pkl \\
        --output data/models/initial_v2.pkl
"""

from __future__ import annotations

import argparse
import io
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from .. import training as _training
from ..training import (
    PipelineTrainerConfig,
    TrainingHistory,
    train_pipeline,
)
from ..groundtruth import TextRecord


def _split_random(examples, val_fraction: float, test_fraction: float,
                  seed: int):
    """Random row-level split (DIAGNOSTIC ONLY — leaks spatial autocorrelation).

    Drop-in for :func:`soilspec.training.split_by_tile`. Use to verify the
    model can learn signal when train/test distributions match; production
    runs should always use the spatial-block split.
    """
    from ..training import _Splits

    rng = np.random.default_rng(int(seed))
    n = len(examples.tile_keys)
    perm = rng.permutation(n).tolist()
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test = perm[:n_test]
    val = perm[n_test : n_test + n_val]
    train = perm[n_test + n_val :]
    return _Splits(train=train, val=val, test=test)

log = logging.getLogger(__name__)


def _load_text_npz(path: Path) -> tuple[list[TextRecord], str]:
    """Read the .npz produced by :mod:`soilspec.datasets.build_text`."""
    data = np.load(path, allow_pickle=False)
    tile_ids = data["tile_ids"]
    times = data["times"]
    embeddings = data["embeddings"]
    doc_ids = data["doc_ids"] if "doc_ids" in data.files else [""] * len(tile_ids)
    encoder = str(data["encoder"]) if "encoder" in data.files else ""
    records = [
        TextRecord(
            tile_id=str(tile_ids[i]), time=int(times[i]),
            embedding=np.asarray(embeddings[i], dtype=np.float32),
            doc_id=str(doc_ids[i]), encoder=encoder,
        )
        for i in range(len(tile_ids))
    ]
    return records, encoder


def _merge_text_into_examples(examples, text_records, encoder_name):
    """Broadcast static per-tile text across every (tile, time) row.

    Text describing a sensor / sampling location is time-invariant, so
    we attach a single embedding per ``tile_id`` and re-use it for every
    bucketed time at that tile. Rows whose tile has no matching text get
    a zero vector and ``text_missing == 1``.
    """
    import dataclasses

    per_tile: dict[str, tuple[np.ndarray, str]] = {}
    text_dim = 0
    for tr in text_records:
        if tr.tile_id in per_tile:
            continue
        emb = np.asarray(tr.embedding, dtype=np.float32).reshape(-1)
        if text_dim == 0:
            text_dim = emb.shape[0]
        elif emb.shape[0] != text_dim:
            raise ValueError(
                f"text_dim mismatch: {emb.shape[0]} vs {text_dim} "
                f"for tile {tr.tile_id}"
            )
        per_tile[tr.tile_id] = (emb, tr.doc_id)

    n = len(examples)
    text_features = np.zeros((n, text_dim), dtype=np.float32)
    text_missing = np.ones(n, dtype=np.float32)
    doc_ids: list[str] = []
    matched_tiles: set[str] = set()
    for i, (tid, _) in enumerate(examples.tile_keys):
        match = per_tile.get(tid)
        if match is None:
            doc_ids.append("")
        else:
            emb, did = match
            text_features[i] = emb
            text_missing[i] = 0.0
            doc_ids.append(did)
            matched_tiles.add(tid)

    log.info(
        "text alignment: %d/%d rows matched, %d/%d unique tiles matched (dim=%d)",
        int((text_missing == 0).sum()), n,
        len(matched_tiles), len({t for t, _ in examples.tile_keys}),
        text_dim,
    )
    return dataclasses.replace(
        examples,
        text_features=text_features,
        text_missing=text_missing,
        text_doc_ids=tuple(doc_ids),
        text_dim=text_dim,
        text_encoder=encoder_name,
    )


def _save_trained(model, history, splits, metrics, output: Path) -> None:
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "schema_version": 1,
        "model_state": model.to_state_dict(),
        "history": {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "epochs_trained": history.epochs_trained,
        },
        "splits": {
            "train": list(splits.train),
            "val": list(splits.val),
            "test": list(splits.test),
        },
        "metrics": {
            p: {
                "rmse": m.rmse, "mae": m.mae, "r2": m.r2, "n": m.n,
                "baseline_rmse": m.baseline_rmse,
            }
            for p, m in metrics.items()
        },
    }
    buf = io.BytesIO()
    torch.save(blob, buf)
    output.write_bytes(buf.getvalue())
    log.info("wrote %d bytes to %s", output.stat().st_size, output)


def _format_metrics(metrics, label_n: dict[str, int]) -> str:
    """Render a per-property metrics table."""
    lines = [
        f"  {'property':14s}  {'n_test':>6s}  {'n_train_total':>13s}  "
        f"{'baseline_rmse':>13s}  {'trained_rmse':>13s}  {'improvement':>11s}  "
        f"{'mae':>8s}  {'R²':>7s}",
    ]
    for prop, m in metrics.items():
        improvement = (1 - m.rmse / m.baseline_rmse) * 100 if m.baseline_rmse > 0 else 0.0
        lines.append(
            f"  {prop:14s}  {m.n:6d}  {label_n.get(prop, 0):13d}  "
            f"{m.baseline_rmse:13.4f}  {m.rmse:13.4f}  {improvement:10.1f}%  "
            f"{m.mae:8.4f}  {m.r2:7.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True,
                   help="Path to RasterTrainingExamples pickle "
                        "(from soilspec.datasets.build).")
    p.add_argument("--output", type=Path,
                   default=Path("data/models/initial_v1.pkl"),
                   help="Destination for the trained model blob.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2,
                   help="Bumped vs the trainer default (0.1) given the small "
                        "292-row dataset.")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--early-stopping-patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--properties", type=str, default=None,
        help="Comma-separated list of properties to train on (default: all "
             "properties present in the dataset). Use 'soil_moisture' to "
             "train a soil-moisture-only baseline.",
    )
    p.add_argument(
        "--random-split", action="store_true",
        help="DIAGNOSTIC: use a random row-level split instead of the "
             "spatial-block split. Leaks spatial autocorrelation; use only "
             "to verify the model can learn signal at all.",
    )
    p.add_argument(
        "--text-npz", type=Path, default=None,
        help="Optional .npz produced by `python -m soilspec.datasets.build_text` "
             "carrying precomputed text embeddings. Adds a trainable text "
             "projection branch on top of the existing fused representation.",
    )
    p.add_argument(
        "--text-projection-dim", type=int, default=32,
        help="Output width of the trainable Linear projection applied to the "
             "frozen text embedding (default: 32).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    log.info("loading %s", args.dataset)
    with args.dataset.open("rb") as fh:
        examples = pickle.load(fh)
    log.info(
        "loaded %d rows, s1=%s, s2=%s, sources=%s",
        len(examples), examples.s1.shape, examples.s2.shape, examples.sources,
    )

    if args.text_npz is not None:
        log.info("loading text embeddings from %s", args.text_npz)
        text_records, encoder_name = _load_text_npz(args.text_npz)
        log.info(
            "loaded %d text records, encoder=%s",
            len(text_records), encoder_name,
        )
        examples = _merge_text_into_examples(examples, text_records, encoder_name)

    label_n = {p: examples.usable(p) for p in examples.property_names}
    log.info("per-property usable label counts: %s", label_n)

    if args.properties:
        target_props = tuple(p.strip() for p in args.properties.split(",") if p.strip())
        unknown = [p for p in target_props if p not in examples.property_names]
        if unknown:
            log.error("unknown properties: %s (available: %s)",
                      unknown, examples.property_names)
            return 2
        log.info("restricting training to properties=%s", target_props)
    else:
        target_props = examples.property_names

    cfg = PipelineTrainerConfig(
        properties=target_props,
        spectral_backend="1d_cnn",
        spatial_backend="cnn",
        spectral_latent_dim=32,
        spatial_latent_dim=32,
        fusion_strategy="concat",
        fusion_output_dim=48,
        head_hidden_dim=64,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=1.0,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        early_stopping_patience=args.early_stopping_patience,
        warmup_epochs=5,
        seed=args.seed,
        text_projection_dim=args.text_projection_dim,
    )

    if args.random_split:
        log.warning(
            "DIAGNOSTIC MODE: monkeypatching split_by_tile -> random row split"
        )
        _training.split_by_tile = _split_random

    history = TrainingHistory()
    log.info("starting train_pipeline with cfg=%s", cfg)
    model, metrics, splits = train_pipeline(examples, cfg, history=history)

    log.info(
        "training finished. epochs=%d, train_rows=%d, val_rows=%d, test_rows=%d",
        history.epochs_trained, len(splits.train), len(splits.val), len(splits.test),
    )
    if history.train_loss:
        log.info(
            "final train_loss=%.4f val_loss=%.4f",
            history.train_loss[-1],
            history.val_loss[-1] if history.val_loss else float("nan"),
        )

    print()
    print("=" * 100)
    print(f"per-property test metrics  (test_rows={len(splits.test)})")
    print("=" * 100)
    print(_format_metrics(metrics, label_n))
    print()

    _save_trained(model, history, splits, metrics, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
