"""CLI driver: build a text-embedding corpus aligned to the training AOIs.

Reads ISMN and/or LUCAS CSVs, generates one descriptive snippet per
sensor/sample, projects each onto every AOI from
:mod:`soilspec.datasets.initial_aois` whose bbox contains it, and writes
a ``.npz`` consumable by ``python -m soilspec.train --text-npz``.

Examples
--------

Quick offline sanity check (no extra deps, deterministic 64-d hash
embeddings)::

    python -m soilspec.datasets.build_text \\
        --ismn-csv data/groundtruth/ismn_2023.csv \\
        --lucas-csv data/groundtruth/lucas_2018_topsoil.csv \\
        --encoder hash \\
        --out data/text_corpora/initial_v2_hash.npz

Real semantic embeddings via sentence-transformers (slow, network on
first run)::

    pip install sentence-transformers
    python -m soilspec.datasets.build_text \\
        --ismn-csv data/groundtruth/ismn_2023.csv \\
        --lucas-csv data/groundtruth/lucas_2018_topsoil.csv \\
        --encoder st:all-MiniLM-L6-v2 \\
        --time-strategy daily \\
        --out data/text_corpora/initial_v2_minilm.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .initial_aois import INITIAL_AOIS
from .text_corpus import (
    AOIBinding, build_text_corpus, encoder_from_spec,
    ismn_station_snippets, lucas_point_snippets, save_text_corpus,
)


DEFAULT_TILE_SIZE = 32
DEFAULT_BUCKET_S = 86400


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m soilspec.datasets.build_text",
        description="Build a text-embedding corpus aligned to the training AOIs.",
    )
    p.add_argument("--ismn-csv", type=Path, default=None,
                   help="ISMNAdapter-format CSV — emits one snippet per station")
    p.add_argument("--lucas-csv", type=Path, default=None,
                   help="LUCASAdapter-format CSV — emits one snippet per point")
    p.add_argument("--encoder", type=str, default="hash",
                   help="encoder spec: 'hash', 'hash:<dim>', or 'st:<model>' "
                        "(default: hash)")
    p.add_argument("--time-strategy", type=str, default="window_start",
                   choices=("window_start", "daily"),
                   help="how to broadcast a snippet across its AOI window; "
                        "'daily' maximises raster alignment at the cost of "
                        "corpus size (default: window_start)")
    p.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE,
                   help=f"must match training tile size (default: {DEFAULT_TILE_SIZE})")
    p.add_argument("--bucket-seconds", type=int, default=DEFAULT_BUCKET_S,
                   help=f"time bucket size (default: {DEFAULT_BUCKET_S})")
    p.add_argument("--out", type=Path, required=True,
                   help="output .npz path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.ismn_csv is None and args.lucas_csv is None:
        print("[build_text] error: pass at least one of --ismn-csv or --lucas-csv",
              file=sys.stderr)
        return 2

    snippets = []
    if args.ismn_csv is not None:
        ismn = list(ismn_station_snippets(args.ismn_csv))
        print(f"[build_text] ISMN: {len(ismn)} unique stations from {args.ismn_csv}")
        snippets.extend(ismn)
    if args.lucas_csv is not None:
        lucas = list(lucas_point_snippets(args.lucas_csv))
        print(f"[build_text] LUCAS: {len(lucas)} unique points from {args.lucas_csv}")
        snippets.extend(lucas)
    if not snippets:
        print("[build_text] no snippets extracted — nothing to do", file=sys.stderr)
        return 2

    aoi_bindings = [
        AOIBinding(aoi=w.aoi, window=w.window) for w in INITIAL_AOIS
    ]
    print(f"[build_text] aligning against {len(aoi_bindings)} AOIs "
          f"(tile_size={args.tile_size}, strategy={args.time_strategy})")

    encoder = encoder_from_spec(args.encoder)
    print(f"[build_text] encoder: {encoder.name}")

    records, enc_name = build_text_corpus(
        snippets=snippets,
        aois=aoi_bindings,
        encoder=encoder,
        tile_size=args.tile_size,
        target_shape=(args.tile_size, args.tile_size),
        bucket_seconds=args.bucket_seconds,
        time_strategy=args.time_strategy,
    )
    print(f"[build_text] emitted {len(records)} aligned text records; "
          f"unique docs={len({r.doc_id for r in records})}; "
          f"unique tiles={len({r.tile_id for r in records})}")

    out_path = save_text_corpus(records, enc_name, args.out)
    print(f"[build_text] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
