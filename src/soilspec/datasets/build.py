"""Driver: turn :mod:`initial_aois` + user-supplied GT CSVs into a
materialized :class:`RasterTrainingExamples` ready for
:func:`soilspec.training.train_pipeline`.

Per AOI it:

1. Builds a 1×1 :class:`TileGrid` over ``aoi.bbox`` matching the PC adapter
   output shape (``tile_size × tile_size``). With ``tile_size=32`` and the
   0.05° AOIs from :mod:`initial_aois`, the grid is a single tile per AOI;
   each AOI thus contributes ``len(s1_times) × len(s2_times)`` paired rows.
2. Runs the matching GT adapter (ISMN or LUCAS) over the AOI bbox + window,
   feeds the :class:`Measurement`\\s into a :class:`GroundTruthDataset`.
3. Fetches S1 (sentinel-1-rtc) and S2 (sentinel-2-l2a) from Planetary
   Computer for the same AOI + window, pairs them by ``time_bucket`` so
   each :class:`PreprocessedRecord` has both modalities (the assembler
   drops single-modality rows).
4. Calls :func:`assemble_raster_examples` to produce a per-AOI
   :class:`RasterTrainingExamples`.

All per-AOI results are then concatenated into one global examples object,
with ``tile_keys`` namespaced by ``aoi_id`` so the downstream spatial-block
split won't collapse different AOIs into the same logical tile.

Failures are isolated to a single AOI: an unreachable PC item, an empty
GT CSV slice, etc. are logged and the AOI is skipped — we don't want one
flaky network call to discard the other 63 AOIs of work.

CLI::

    python -m soilspec.datasets.build \\
        --ismn-csv path/to/ismn_2023.csv \\
        --lucas-csv path/to/lucas_2018_topsoil.csv \\
        --output data/trainsets/initial_v1.pkl
"""

from __future__ import annotations

import argparse
import io
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..groundtruth import (
    GroundTruthDataset,
    ISMNAdapter,
    LUCASAdapter,
    RasterTrainingExamples,
    TileGrid,
    assemble_raster_examples,
)
from ..ingestion.adapters import RawAsset, UnreachableSourceError
from ..ingestion.planetary import (
    PlanetaryComputerSentinel1Adapter,
    PlanetaryComputerSentinel2Adapter,
)
from ..types import (
    AOI,
    MEASURED_PROPERTY_NAMES,
    PreprocessedRecord,
    TimeWindow,
)
from .initial_aois import INITIAL_AOIS, AOIWindow


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-AOI build
# ---------------------------------------------------------------------------


# Default Planetary Computer tile size: must match what we pass the adapters
# so the TileGrid we build for GT-join lines up with the raster shape.
DEFAULT_TILE_SIZE = 32
DEFAULT_TIME_BUCKET_S = 86400  # 1 day — same as test fixtures


@dataclass(frozen=True)
class _AOIBuildResult:
    aoi_id: str
    examples: RasterTrainingExamples
    n_s1_items: int
    n_s2_items: int
    n_gt_samples: int


def _pair_assets_by_bucket(
    s1_assets: list[RawAsset],
    s2_assets: list[RawAsset],
    bucket_seconds: int,
) -> list[tuple[int, RawAsset, RawAsset]]:
    """Pair S1 + S2 acquisitions that fall in the same time bucket.

    PC returns S1 (6-day revisit) and S2 (5-day revisit) on independent
    cadences. We collapse both to ``floor(t / bucket_seconds) * bucket_seconds``
    and emit one pair per bucket where both modalities are present, taking
    the *latest* asset in the bucket for each modality (no compositing —
    a single dated acquisition is the unit of supervision against the
    aligned-day GT measurements).
    """
    s1_by_bucket: dict[int, RawAsset] = {}
    s2_by_bucket: dict[int, RawAsset] = {}
    for asset in s1_assets:
        b = (asset.metadata.timestamp // bucket_seconds) * bucket_seconds
        prev = s1_by_bucket.get(b)
        if prev is None or asset.metadata.timestamp > prev.metadata.timestamp:
            s1_by_bucket[b] = asset
    for asset in s2_assets:
        b = (asset.metadata.timestamp // bucket_seconds) * bucket_seconds
        prev = s2_by_bucket.get(b)
        if prev is None or asset.metadata.timestamp > prev.metadata.timestamp:
            s2_by_bucket[b] = asset
    pairs: list[tuple[int, RawAsset, RawAsset]] = []
    for b in sorted(s1_by_bucket.keys() & s2_by_bucket.keys()):
        pairs.append((b, s1_by_bucket[b], s2_by_bucket[b]))
    return pairs


def _payload_to_array(asset: RawAsset, modality: str) -> np.ndarray:
    """Pull the (C, H, W) float32 raster out of a PC adapter payload.

    S1 payloads are already arrays; S2 payloads are dicts with ``reflectance``
    plus a ``qa`` mask. The QA channel is dropped here because the encoder
    doesn't consume it — the cloud-masking decision is encoded in
    ``S2_CLOUD_CLASSES`` upstream.
    """
    if modality == "s1":
        arr = asset.payload  # np.ndarray (2, H, W)
    else:
        arr = asset.payload["reflectance"]  # (6, H, W)
    return np.asarray(arr, dtype=np.float32)


def _build_one_aoi(
    item: AOIWindow,
    ismn_csv: Path | None,
    lucas_csv: Path | None,
    tile_size: int,
    bucket_seconds: int,
) -> _AOIBuildResult | None:
    """Fetch + assemble examples for a single AOI. Returns None on failure."""
    aoi: AOI = item.aoi
    window: TimeWindow = item.window

    # ---- GT side ----
    csv_path: Path | None
    if item.label_source == "ismn":
        csv_path = ismn_csv
        if csv_path is None:
            log.warning("%s: ismn-csv not provided, skipping", aoi.aoi_id)
            return None
        gt_adapter = ISMNAdapter(csv_path=csv_path)
    elif item.label_source == "lucas":
        csv_path = lucas_csv
        if csv_path is None:
            log.warning("%s: lucas-csv not provided, skipping", aoi.aoi_id)
            return None
        gt_adapter = LUCASAdapter(csv_path=csv_path)
    else:
        log.warning("%s: unknown label_source %r", aoi.aoi_id, item.label_source)
        return None

    grid = TileGrid.from_shape(
        aoi.bbox, raster_shape=(tile_size, tile_size), tile_size=tile_size,
    )
    # Per-source GT bucket size. ISMN measurements vary day-to-day so daily
    # buckets preserve that signal; LUCAS gives one annual value per point
    # so we need a yearly bucket to let the (rec.time // bucket) join match
    # a Jan-1 GT timestamp against any 2018-dated S1+S2 acquisition.
    # (S1+S2 pairing still uses ``bucket_seconds`` (1 day) — each daily
    # pair becomes its own training row regardless of GT bucket size.)
    gt_bucket_seconds = (
        365 * 86400 if item.label_source == "lucas" else bucket_seconds
    )
    ds = GroundTruthDataset(grid, time_bucket_seconds=gt_bucket_seconds)
    try:
        ds.extend(gt_adapter.fetch(aoi, window))
    except FileNotFoundError as e:
        log.warning("%s: GT CSV missing (%s); skipping", aoi.aoi_id, e)
        return None
    n_gt = len(ds)
    if n_gt == 0:
        log.info("%s: no GT samples in window; skipping", aoi.aoi_id)
        return None

    # ---- S1 side ----
    try:
        s1_assets = list(PlanetaryComputerSentinel1Adapter(
            tile_size=tile_size,
        ).fetch(aoi, window))
    except UnreachableSourceError as e:
        log.warning("%s: S1 fetch failed: %s", aoi.aoi_id, e)
        return None

    # ---- S2 side ----
    try:
        s2_assets = list(PlanetaryComputerSentinel2Adapter(
            tile_size=tile_size,
        ).fetch(aoi, window))
    except UnreachableSourceError as e:
        log.warning("%s: S2 fetch failed: %s", aoi.aoi_id, e)
        return None

    pairs = _pair_assets_by_bucket(s1_assets, s2_assets, bucket_seconds)
    if not pairs:
        log.info(
            "%s: no overlapping S1+S2 buckets (s1=%d, s2=%d); skipping",
            aoi.aoi_id, len(s1_assets), len(s2_assets),
        )
        return None

    # All AOIs share the single-tile grid → tile_id is constant per AOI.
    # Namespacing across AOIs happens at concatenation time.
    only_tile_id = grid.locate(
        (aoi.bbox.min_lon + aoi.bbox.max_lon) / 2,
        (aoi.bbox.min_lat + aoi.bbox.max_lat) / 2,
    )
    assert only_tile_id is not None  # centroid is by construction inside bbox

    records: list[PreprocessedRecord] = []
    for bucket_start, s1_asset, s2_asset in pairs:
        records.append(PreprocessedRecord(
            aoi_id=aoi.aoi_id,
            tile_id=only_tile_id,
            time=int(bucket_start),  # align record time onto the same bucket
            crs="EPSG:4326",
            bounds=aoi.bbox,
            spatial={
                "s1": _payload_to_array(s1_asset, "s1"),
                "s2": _payload_to_array(s2_asset, "s2"),
            },
            vector={},
        ))

    sources = ("ismn",) if item.label_source == "ismn" else ("lucas",)
    examples = assemble_raster_examples(
        records, ds,
        property_names=MEASURED_PROPERTY_NAMES,
        sources=sources,
    )
    if len(examples) == 0:
        log.info(
            "%s: assembled 0 paired rows from %d GT samples / %d S1+S2 buckets",
            aoi.aoi_id, n_gt, len(pairs),
        )
        return None
    return _AOIBuildResult(
        aoi_id=aoi.aoi_id,
        examples=examples,
        n_s1_items=len(s1_assets),
        n_s2_items=len(s2_assets),
        n_gt_samples=n_gt,
    )


# ---------------------------------------------------------------------------
# Cross-AOI concatenation
# ---------------------------------------------------------------------------


def _concat_examples(
    per_aoi: list[_AOIBuildResult],
) -> RasterTrainingExamples:
    """Concatenate per-AOI examples into one global RasterTrainingExamples.

    Namespaces ``tile_keys`` with ``aoi_id`` so the spatial-block splitter
    in :func:`split_by_tile` treats each AOI as an independent tile group
    (rather than collapsing every AOI's local ``r000c000`` into one).
    """
    if not per_aoi:
        raise RuntimeError(
            "no per-AOI examples to concatenate — every AOI was skipped"
        )

    # Validate shape compatibility across AOIs. PC adapters give us a fixed
    # (B, H, W); a mismatch means a config drift that would silently break
    # train_pipeline, better to fail loudly here.
    first = per_aoi[0].examples
    for r in per_aoi[1:]:
        if r.examples.s1.shape[1:] != first.s1.shape[1:]:
            raise RuntimeError(
                f"{r.aoi_id}: S1 shape {r.examples.s1.shape[1:]} "
                f"!= {first.s1.shape[1:]} from {per_aoi[0].aoi_id}"
            )
        if r.examples.s2.shape[1:] != first.s2.shape[1:]:
            raise RuntimeError(
                f"{r.aoi_id}: S2 shape {r.examples.s2.shape[1:]} "
                f"!= {first.s2.shape[1:]} from {per_aoi[0].aoi_id}"
            )

    s1 = np.concatenate([r.examples.s1 for r in per_aoi], axis=0)
    s2 = np.concatenate([r.examples.s2 for r in per_aoi], axis=0)
    # All AOIs currently produce zero vector features (rec.vector={}), so
    # vector_features is (N, 0). Concatenation along axis 0 still works.
    vec = np.concatenate(
        [r.examples.vector_features for r in per_aoi], axis=0,
    )

    props = first.property_names
    y = {
        p: np.concatenate([r.examples.y[p] for r in per_aoi], axis=0)
        for p in props
    }
    w = {
        p: np.concatenate([r.examples.weights[p] for r in per_aoi], axis=0)
        for p in props
    }

    # Namespace tile_keys: ``r000c000`` becomes ``{aoi_id}/r000c000``.
    tile_keys: list[tuple[str, int]] = []
    for r in per_aoi:
        for (tid, bucket) in r.examples.tile_keys:
            tile_keys.append((f"{r.aoi_id}/{tid}", bucket))

    sources: set[str] = set()
    for r in per_aoi:
        sources.update(r.examples.sources)

    return RasterTrainingExamples(
        s1=s1, s2=s2, vector_features=vec,
        y=y, weights=w,
        tile_keys=tuple(tile_keys),
        property_names=props,
        sources=tuple(sorted(sources)),
        s1_bands=int(s1.shape[1]),
        s2_bands=int(s2.shape[1]),
        vector_attr_names=first.vector_attr_names,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def build_initial_trainset(
    ismn_csv: Path | None,
    lucas_csv: Path | None,
    aois: Iterable[AOIWindow] = INITIAL_AOIS,
    tile_size: int = DEFAULT_TILE_SIZE,
    bucket_seconds: int = DEFAULT_TIME_BUCKET_S,
) -> RasterTrainingExamples:
    """Build the global :class:`RasterTrainingExamples` for the initial cut.

    Failures per AOI are logged and skipped; a hard error is only raised if
    *every* AOI fails (so the caller doesn't silently end up with an empty
    training set).
    """
    per_aoi: list[_AOIBuildResult] = []
    for item in aois:
        log.info(
            "building %s (%s, %s)", item.aoi.aoi_id, item.label_source, item.network,
        )
        result = _build_one_aoi(
            item, ismn_csv, lucas_csv, tile_size, bucket_seconds,
        )
        if result is None:
            continue
        log.info(
            "  ok: %d rows (s1=%d s2=%d gt=%d)",
            len(result.examples), result.n_s1_items,
            result.n_s2_items, result.n_gt_samples,
        )
        per_aoi.append(result)

    return _concat_examples(per_aoi)


def save_examples(examples: RasterTrainingExamples, path: Path) -> None:
    """Persist as pickle. Arrays are numpy so this round-trips cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    pickle.dump(examples, buf, protocol=pickle.HIGHEST_PROTOCOL)
    path.write_bytes(buf.getvalue())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ismn-csv", type=Path, default=None,
                   help="Path to ISMN measurements CSV (see ISMNAdapter docs).")
    p.add_argument("--lucas-csv", type=Path, default=None,
                   help="Path to LUCAS 2018 topsoil CSV.")
    p.add_argument("--output", type=Path,
                   default=Path("data/trainsets/initial_v1.pkl"),
                   help="Destination pickle for the assembled examples.")
    p.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    p.add_argument("--bucket-seconds", type=int, default=DEFAULT_TIME_BUCKET_S)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.ismn_csv is None and args.lucas_csv is None:
        log.error(
            "at least one of --ismn-csv / --lucas-csv must be provided "
            "(otherwise every AOI is skipped)"
        )
        return 2

    examples = build_initial_trainset(
        ismn_csv=args.ismn_csv,
        lucas_csv=args.lucas_csv,
        tile_size=args.tile_size,
        bucket_seconds=args.bucket_seconds,
    )
    save_examples(examples, args.output)
    log.info(
        "wrote %d examples (s1=%dx%dx%d, s2=%dx%dx%d) to %s",
        len(examples),
        examples.s1_bands, examples.s1.shape[2], examples.s1.shape[3],
        examples.s2_bands, examples.s2.shape[2], examples.s2.shape[3],
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "build_initial_trainset",
    "save_examples",
    "main",
]
