"""Derive AOI registry from actual GT CSV contents via farthest-point sampling.

Replaces hand-picked coordinates (which can miss the real station/point
location by tens of kilometres if guessed wrong) with coordinates pulled
directly from the converted CSVs. Each AOI is guaranteed to contain at
least one GT measurement.

Selection strategy: greedy farthest-point sampling within each network /
across the LUCAS point set. The first pick is the one closest to the
group centroid; each subsequent pick maximises the minimum distance to
already-picked stations. The result spreads selections over the
network's geographic footprint without any manual climate-zone tagging.

Picks per ISMN network (totalling 21):

- USCRN: 8       SCAN: 4       REMEDHUS: 4
- TERENO: 3      SMOSMANIA: 2

LUCAS picks: 32, sampled across all 18,744 EU points.

Run::

    python -m soilspec.datasets.regenerate_aois \\
        --ismn-csv data/groundtruth/ismn_2023.csv \\
        --lucas-csv data/groundtruth/lucas_2018_topsoil.csv \\
        --output data/registry/initial_aois.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)


ISMN_PICKS_PER_NETWORK: dict[str, int] = {
    "USCRN":     8,
    "SCAN":      4,
    "REMEDHUS":  4,
    "TERENO":    3,
    "SMOSMANIA": 2,
}
LUCAS_PICKS: int = 32


# ---------------------------------------------------------------------------
# Farthest-point sampling
# ---------------------------------------------------------------------------


def _farthest_point_sample(
    coords: list[tuple[float, float]], n: int,
) -> list[int]:
    """Return ``n`` indices of ``coords`` spread far apart in lon/lat space.

    Distance metric is squared Euclidean on (lon, lat) — for AOIs spread
    across continents, geodesic-vs-Euclidean doesn't change which points
    are picked, only the absolute distances. Fast and dependency-free.
    """
    k = min(n, len(coords))
    if k <= 0:
        return []
    if k == len(coords):
        return list(range(k))

    cx = sum(p[0] for p in coords) / len(coords)
    cy = sum(p[1] for p in coords) / len(coords)
    first = min(
        range(len(coords)),
        key=lambda i: (coords[i][0] - cx) ** 2 + (coords[i][1] - cy) ** 2,
    )
    picked = [first]
    # Maintain per-point min-distance to the picked set incrementally —
    # O(N·k) instead of O(N²·k).
    min_d = [
        (coords[i][0] - coords[first][0]) ** 2
        + (coords[i][1] - coords[first][1]) ** 2
        for i in range(len(coords))
    ]
    min_d[first] = -1.0  # mark as picked

    while len(picked) < k:
        next_i = max(range(len(coords)), key=lambda i: min_d[i])
        picked.append(next_i)
        x, y = coords[next_i]
        for i in range(len(coords)):
            if min_d[i] < 0:
                continue
            d = (coords[i][0] - x) ** 2 + (coords[i][1] - y) ** 2
            if d < min_d[i]:
                min_d[i] = d
        min_d[next_i] = -1.0

    return picked


# ---------------------------------------------------------------------------
# ISMN station selection
# ---------------------------------------------------------------------------


def _read_ismn_stations(
    csv_path: Path,
) -> dict[str, list[tuple[str, float, float]]]:
    """Network → list of (station, lat, lon) — one entry per unique station."""
    seen: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            nw = row["network"]
            st = row["station"]
            if st in seen[nw]:
                continue
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
            except (TypeError, ValueError):
                continue
            seen[nw][st] = (lat, lon)
    return {
        nw: [(st, lat, lon) for st, (lat, lon) in d.items()]
        for nw, d in seen.items()
    }


def _pick_ismn_aois(
    stations_by_nw: dict[str, list[tuple[str, float, float]]],
) -> list[dict]:
    """Apply farthest-point sampling per network. Returns AOI dicts."""
    aois: list[dict] = []
    for network, n_picks in ISMN_PICKS_PER_NETWORK.items():
        stations = stations_by_nw.get(network, [])
        if not stations:
            log.warning(
                "network %s has no stations in CSV; skipping %d picks",
                network, n_picks,
            )
            continue
        coords = [(s[2], s[1]) for s in stations]  # (lon, lat) for FPS
        picked = _farthest_point_sample(coords, n_picks)
        log.info("%-10s picked %d of %d stations", network, len(picked), len(stations))
        for idx in picked:
            station, lat, lon = stations[idx]
            slug = (
                station.lower()
                .replace(" ", "_").replace("-", "_").replace("/", "_")
                .replace(".", "").replace("'", "")
            )
            aois.append({
                "aoi_id": f"ismn_{network.lower()}_{slug}",
                "label_source": "ismn",
                "network": network,
                "station": station,
                "lat": lat,
                "lon": lon,
                "hemisphere": "NH" if lat >= 0 else "SH",
            })
    return aois


# ---------------------------------------------------------------------------
# LUCAS point selection
# ---------------------------------------------------------------------------


def _read_lucas_points(csv_path: Path) -> list[tuple[str, float, float]]:
    """List of (point_id, lat, lon) for every LUCAS row."""
    out: list[tuple[str, float, float]] = []
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
            except (TypeError, ValueError):
                continue
            out.append((row["point_id"], lat, lon))
    return out


def _pick_lucas_aois(points: list[tuple[str, float, float]]) -> list[dict]:
    coords = [(p[2], p[1]) for p in points]  # (lon, lat)
    picked = _farthest_point_sample(coords, LUCAS_PICKS)
    log.info("LUCAS      picked %d of %d points", len(picked), len(points))
    aois: list[dict] = []
    for idx in picked:
        point_id, lat, lon = points[idx]
        aois.append({
            "aoi_id": f"lucas_{point_id}",
            "label_source": "lucas",
            "network": "LUCAS-2018",
            "point_id": point_id,
            "lat": lat,
            "lon": lon,
            "hemisphere": "NH",  # all LUCAS sites in N. Hemisphere
        })
    return aois


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ismn-csv", type=Path, required=True)
    p.add_argument("--lucas-csv", type=Path, required=True)
    p.add_argument(
        "--output", type=Path,
        default=Path("data/registry/initial_aois.json"),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.ismn_csv.exists():
        log.error("ISMN CSV not found: %s", args.ismn_csv)
        return 2
    if not args.lucas_csv.exists():
        log.error("LUCAS CSV not found: %s", args.lucas_csv)
        return 2

    stations_by_nw = _read_ismn_stations(args.ismn_csv)
    ismn_aois = _pick_ismn_aois(stations_by_nw)

    lucas_points = _read_lucas_points(args.lucas_csv)
    lucas_aois = _pick_lucas_aois(lucas_points)

    payload = {
        "ismn": ismn_aois,
        "lucas": lucas_aois,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(payload, fh, indent=2)
    log.info(
        "wrote %d ISMN + %d LUCAS AOIs to %s",
        len(ismn_aois), len(lucas_aois), args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
