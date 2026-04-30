"""Command-line trainer: ingest -> preprocess -> assemble -> fit -> save.

Usage::

    python -m soilspec.train \\
        --aoi "<min_lon>,<min_lat>,<max_lon>,<max_lat>" \\
        --start <epoch_seconds> --end <epoch_seconds> \\
        --ismn <path_to_ismn_archive> \\
        [--soilgrids-csv <path>] [--lucas-csv <path>] \\
        --out <family>:<version> \\
        [--epochs N] [--batch-size N] [--lr 1e-3] [--seed 0] \\
        [--properties soil_moisture,soc,...]

The CLI uses synthetic source adapters by default for the satellite data
(matching the rest of the codebase). Real Sentinel adapters can be wired in
via :mod:`soilspec.ingestion.planetary` once the ``planetary`` extra is
configured at the call site.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .groundtruth import (
    GroundTruthDataset, ISMNAdapter, ISMNArchiveAdapter,
    LUCASAdapter, SoilGridsAdapter, TileGrid,
    assemble_raster_examples,
)
from .ingestion import AdapterRegistry, Ingestion, MetadataParser
from .orchestrator import OrchestratorConfig, PipelineOrchestrator
from .preprocessing import Preprocessor
from .storage import StorageTierManager
from .training import (
    PipelineTrainerConfig, save_pipeline, train_pipeline,
)
from .types import (
    SENTINEL1, SENTINEL2, VECTOR, AnalysisRequest, AOI, BoundingBox,
    MEASURED_PROPERTY_NAMES, TimeWindow,
)


def _parse_aoi(s: str) -> BoundingBox:
    parts = [float(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--aoi expects 'min_lon,min_lat,max_lon,max_lat'; got {s!r}"
        )
    return BoundingBox(*parts)


def _parse_out(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"--out expects 'family:version'; got {s!r}")
    family, version = s.split(":", 1)
    if not family or not version:
        raise argparse.ArgumentTypeError(f"empty family or version in {s!r}")
    return family, version


def _parse_props(s: str | None) -> tuple[str, ...]:
    if not s:
        return MEASURED_PROPERTY_NAMES
    return tuple(p.strip() for p in s.split(",") if p.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m soilspec.train",
        description="Train an end-to-end soil property pipeline from real ground truth.",
    )
    p.add_argument("--aoi", required=True, type=_parse_aoi,
                   help="bbox 'min_lon,min_lat,max_lon,max_lat'")
    p.add_argument("--aoi-id", default="train_aoi",
                   help="stable AOI identifier (default: train_aoi)")
    p.add_argument("--start", required=True, type=int,
                   help="time window start, epoch seconds")
    p.add_argument("--end", required=True, type=int,
                   help="time window end, epoch seconds")
    p.add_argument("--ismn", type=Path, default=None,
                   help="path to ISMN archive (zip or unpacked dir) OR a CSV "
                        "fixture in the ISMNAdapter format")
    p.add_argument("--soilgrids-csv", type=Path, default=None,
                   help="optional CSV of SoilGrids covariates "
                        "(SoilGridsAdapter fixture format)")
    p.add_argument("--lucas-csv", type=Path, default=None,
                   help="optional LUCAS CSV (already normalized via "
                        "normalize_lucas_esdac_csv)")
    p.add_argument("--out", required=True, type=_parse_out,
                   help="output model key 'family:version'")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--properties", type=_parse_props, default=None,
                   help="comma-separated measured property names "
                        "(default: full vocabulary)")
    return p


def _build_gt_dataset(
    args: argparse.Namespace, grid: TileGrid, aoi: AOI, window: TimeWindow,
) -> GroundTruthDataset:
    ds = GroundTruthDataset(grid, time_bucket_seconds=86400)
    if args.ismn is not None:
        path = Path(args.ismn)
        if path.suffix.lower() == ".csv":
            ds.extend(ISMNAdapter(csv_path=path).fetch(aoi, window))
        else:
            ds.extend(ISMNArchiveAdapter(archive_path=path).fetch(aoi, window))
    if args.soilgrids_csv is not None:
        ds.extend(SoilGridsAdapter(csv_path=args.soilgrids_csv).fetch(aoi, window))
    if args.lucas_csv is not None:
        ds.extend(LUCASAdapter(csv_path=args.lucas_csv).fetch(aoi, window))
    return ds


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    aoi = AOI(aoi_id=args.aoi_id, bbox=args.aoi)
    window = TimeWindow(start=args.start, end=args.end)

    print(f"[train] aoi={aoi.aoi_id} bbox={aoi.bbox} "
          f"window=[{window.start}, {window.end}]")

    storage = StorageTierManager()
    adapters = {
        m: AdapterRegistry.create(m)
        for m in (SENTINEL1, SENTINEL2, VECTOR)
    }
    ingestion = Ingestion(storage, adapters, MetadataParser())
    cfg = OrchestratorConfig()  # default preprocessor + encoders + fusion
    print(f"[train] ingesting {list(adapters.keys())}...")
    handles = ingestion.fetch(aoi, window, list(adapters.keys()))
    print(f"[train] {len(handles)} raw observations.")

    preprocessor = Preprocessor(cfg.preprocess)
    print(f"[train] preprocessing...")
    records = preprocessor.preprocess(handles, storage)
    print(f"[train] {len(records)} preprocessed (tile, time) records.")

    grid = TileGrid.from_shape(
        aoi.bbox, raster_shape=cfg.preprocess.target_shape,
        tile_size=cfg.preprocess.tile_size,
    )
    gt = _build_gt_dataset(args, grid, aoi, window)
    n_samples = sum(1 for _ in gt.samples())
    print(f"[train] {n_samples} aggregated GT samples (dropped={gt.dropped} out of AOI).")

    examples = assemble_raster_examples(
        records, gt, property_names=args.properties or MEASURED_PROPERTY_NAMES,
    )
    print(f"[train] {len(examples)} (raster, label) training rows; "
          f"sources={examples.sources}; vector_attrs={examples.vector_attr_names}")
    if len(examples) == 0:
        print("[train] no usable rows — check that GT samples align with "
              "tiles produced by the preprocessor.")
        return 2
    for p in examples.property_names:
        n = examples.usable(p)
        if n > 0:
            print(f"[train]   labels for {p:>20}: {n}")

    trainer_cfg = PipelineTrainerConfig(
        properties=tuple(args.properties or MEASURED_PROPERTY_NAMES),
        spectral_backend=cfg.spectral_backend,
        spatial_backend=cfg.spatial_backend,
        spectral_latent_dim=cfg.spectral_latent_dim,
        spatial_latent_dim=cfg.spatial_latent_dim,
        fusion_strategy=cfg.fusion.strategy,
        fusion_output_dim=cfg.fusion.output_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(f"[train] fitting (epochs={trainer_cfg.epochs}, "
          f"batch={trainer_cfg.batch_size}, lr={trainer_cfg.learning_rate})...")
    model, metrics, splits = train_pipeline(examples, trainer_cfg)
    print(f"[train] split: train={len(splits.train)} val={len(splits.val)} "
          f"test={len(splits.test)} (tiles)")

    if metrics:
        print("[train] held-out test metrics:")
        for prop, m in sorted(metrics.items()):
            beat = "yes" if m.rmse < m.baseline_rmse else "no"
            print(f"[train]   {prop:>20}: R²={m.r2:+.3f} RMSE={m.rmse:.4f} "
                  f"MAE={m.mae:.4f} (n={m.n}, beat-baseline={beat})")
    else:
        print("[train] no held-out rows — skipping metrics. Increase AOI "
              "size or reduce val/test fractions.")

    family, version = args.out
    save_pipeline(storage, family, version, model)
    print(f"[train] saved to StorageTier.MODEL key={family}/{version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
