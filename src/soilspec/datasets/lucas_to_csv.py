"""Convert the ESDAC LUCAS 2018 release into the flat CSV
:class:`LUCASAdapter` consumes.

The 2018 release at ESDAC (``LUCAS-SOIL-2018-v2``) ships per-point soil
property measurements with the theoretical sampling coordinates already
in the same CSV (``TH_LAT`` / ``TH_LONG`` columns, in WGS84 — the
sibling shapefile's ``.prj`` confirms ``GCS_WGS_1984``). So unlike earlier
LUCAS releases, no shapefile join is required for this one — we just
read the CSV and rename a few columns.

Texture (clay/sand/silt) is **not** in this release per ESDAC's notes;
the converter therefore emits only ``soc, nitrogen, phosphorus,
potassium, ph``. :class:`LUCASAdapter` tolerates missing property
columns.

Run::

    python -m soilspec.datasets.lucas_to_csv \\
        --csv ~/Downloads/LUCAS-SOIL-2018/LUCAS-SOIL-2018-v2/LUCAS-SOIL-2018.csv \\
        --output data/groundtruth/lucas_2018_topsoil.csv

By default the converter keeps only the canonical ``0-20 cm`` topsoil
rows (>99% of the file). The other depths exist as occasional follow-up
samples and would muddle the per-tile label aggregation.

For older LUCAS releases where the CSV does *not* carry ``TH_LAT`` /
``TH_LONG``, pass ``--shapefile`` to join on ``POINTID`` instead. Doing
so requires ``pip install pyshp pyproj``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


# Output column order — LUCASAdapter required columns first, then the
# property columns that this 2018 release contains.
OUTPUT_BASE: tuple[str, ...] = ("point_id", "lon", "lat", "year")
OUTPUT_PROPS: tuple[str, ...] = (
    "soc", "nitrogen", "phosphorus", "potassium", "ph",
)
OUTPUT_COLUMNS: tuple[str, ...] = OUTPUT_BASE + OUTPUT_PROPS


# Source-column candidates per canonical name. Order = preference (we take
# the first that exists in the CSV header).
_PROP_CANDIDATES: dict[str, tuple[str, ...]] = {
    "soc": ("OC", "SOC", "OC_2018", "oc"),
    "nitrogen": ("N", "Nitrogen", "N_2018", "n"),
    "phosphorus": ("P", "Phosphorus", "P_2018", "p"),
    "potassium": ("K", "Potassium", "K_2018", "k"),
    # CaCl2-extracted pH is the LUCAS canonical (matches what
    # LUCAS_COLUMN_MAP names "ph"). Fall back to H2O only if CaCl2
    # is absent.
    "ph": ("pH_CaCl2", "pH_CaCl", "pH_in_CaCl2", "PH_CACL2",
           "pH_H2O", "PH_H2O", "pH"),
}

_POINT_ID_CANDIDATES: tuple[str, ...] = (
    "POINTID", "POINT_ID", "POINT_NUMBER", "Point_ID", "PointID", "point_id",
)

_LAT_CANDIDATES: tuple[str, ...] = ("TH_LAT", "lat", "LAT", "latitude", "Latitude")
_LON_CANDIDATES: tuple[str, ...] = ("TH_LONG", "TH_LON", "lon", "LON", "longitude", "Longitude")

_DEPTH_CANDIDATES: tuple[str, ...] = ("Depth", "DEPTH", "depth")


def _resolve_column(header: list[str], candidates: Iterable[str]) -> str | None:
    """Return the first header column matching any candidate name."""
    header_set = set(header)
    for c in candidates:
        if c in header_set:
            return c
    return None


def _sniff_delimiter(path: Path) -> str:
    with path.open("r", newline="") as fh:
        head = fh.readline()
    if head.count(";") > head.count(","):
        return ";"
    return ","


def _safe_float(s):
    """Permissive numeric parser: blanks / 'n/a' / '<0.01' / '< LOD' → None.

    LUCAS encodes below-detection as ``< LOD`` (with space) or ``<0.01``;
    treat both as missing rather than zero so we don't fabricate precision
    the lab didn't actually report.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s) if s == s else None
    s = str(s).strip()
    if not s or s.lower() in ("n/a", "na", "nan", "null", "<lod", "bdl", "< lod"):
        return None
    if s.startswith("<") or s.startswith(">"):
        return None
    s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v == v else None


# ---------------------------------------------------------------------------
# Optional shapefile path (legacy / older LUCAS releases)
# ---------------------------------------------------------------------------


def _load_point_coords_from_shapefile(
    shapefile_path: Path,
) -> dict[str, tuple[float, float]]:
    """Read POINTID → (lon, lat) from an EPSG:3035 shapefile, reprojecting to WGS84.

    Used only if the source CSV is missing coordinates (older LUCAS
    releases). The 2018-v2 release this module targets carries
    ``TH_LAT`` / ``TH_LONG`` in the main CSV, so this path is normally
    unused.
    """
    try:
        import shapefile  # pyshp
    except ImportError as e:
        raise RuntimeError(
            "the `pyshp` package is required for shapefile-based joins: "
            "pip install pyshp"
        ) from e
    try:
        import pyproj
    except ImportError as e:
        raise RuntimeError(
            "the `pyproj` package is required for shapefile-based joins: "
            "pip install pyproj"
        ) from e

    if not shapefile_path.exists():
        raise FileNotFoundError(shapefile_path)

    prj_path = shapefile_path.with_suffix(".prj")
    src_crs: object = "EPSG:3035"
    if prj_path.exists():
        try:
            src_crs = pyproj.CRS.from_wkt(prj_path.read_text())
        except Exception:
            log.warning("could not parse %s; assuming EPSG:3035", prj_path.name)
    transformer = pyproj.Transformer.from_crs(
        src_crs, "EPSG:4326", always_xy=True,
    )

    coords: dict[str, tuple[float, float]] = {}
    with shapefile.Reader(str(shapefile_path)) as sf:
        field_names = [f[0] for f in sf.fields[1:]]
        point_id_col = _resolve_column(field_names, _POINT_ID_CANDIDATES)
        if point_id_col is None:
            raise ValueError(
                f"shapefile {shapefile_path.name} has no POINT_ID-like column. "
                f"Known fields: {field_names}"
            )
        pid_idx = field_names.index(point_id_col)
        for sr in sf.iterShapeRecords():
            point_id = str(sr.record[pid_idx])
            geom = sr.shape
            if not getattr(geom, "points", None):
                continue
            x, y = geom.points[0]
            lon, lat = transformer.transform(x, y)
            coords[point_id] = (float(lon), float(lat))
    log.info(
        "loaded %d coordinates from shapefile %s",
        len(coords), shapefile_path.name,
    )
    return coords


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(
    csv_in: Path,
    output: Path,
    shapefile_path: Path | None = None,
    year: int = 2018,
    depth_filter: str | None = "0-20 cm",
) -> int:
    """Join properties + coordinates and write the adapter CSV.

    If ``shapefile_path`` is ``None`` (the normal case for 2018-v2), the
    converter reads ``TH_LAT`` / ``TH_LONG`` from the CSV directly. If
    those columns are absent, it raises an error suggesting that the user
    pass a shapefile.

    Returns the number of output rows written.
    """
    if not csv_in.exists():
        raise FileNotFoundError(csv_in)

    delimiter = _sniff_delimiter(csv_in)
    log.info("reading %s (delimiter=%r)", csv_in.name, delimiter)

    output.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_dropped_depth = 0
    n_dropped_coords = 0
    n_dropped_no_props = 0

    with csv_in.open("r", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        header = reader.fieldnames or []
        if not header:
            raise ValueError(f"{csv_in} has no header row")

        point_id_col = _resolve_column(header, _POINT_ID_CANDIDATES)
        if point_id_col is None:
            raise ValueError(
                f"{csv_in.name} has no POINT_ID-like column. "
                f"Header: {header[:20]}..."
            )

        lat_col = _resolve_column(header, _LAT_CANDIDATES)
        lon_col = _resolve_column(header, _LON_CANDIDATES)
        depth_col = _resolve_column(header, _DEPTH_CANDIDATES)

        # Coordinate source: CSV columns preferred; shapefile only as
        # fallback when the CSV lacks them.
        shapefile_coords: dict[str, tuple[float, float]] | None = None
        if lat_col and lon_col:
            log.info(
                "using in-CSV coordinates: lat=%s lon=%s", lat_col, lon_col,
            )
        elif shapefile_path is not None:
            shapefile_coords = _load_point_coords_from_shapefile(shapefile_path)
        else:
            raise ValueError(
                f"{csv_in.name} has no lat/lon columns and no --shapefile "
                f"was provided. Try passing the LUCAS-SOIL-2018.shp via "
                f"--shapefile to do a POINTID-based join."
            )

        # Resolve which canonical prop maps to which source column.
        prop_resolution: dict[str, str] = {}
        for canonical, candidates in _PROP_CANDIDATES.items():
            src = _resolve_column(header, candidates)
            if src is not None:
                prop_resolution[canonical] = src
        if not prop_resolution:
            raise ValueError(
                f"{csv_in.name} has none of the expected property columns. "
                f"Header: {header[:20]}..."
            )
        log.info(
            "resolved properties: %s",
            ", ".join(f"{k}<-{v}" for k, v in prop_resolution.items()),
        )
        if depth_col is None and depth_filter is not None:
            log.warning(
                "no Depth column found; skipping the depth filter %r",
                depth_filter,
            )

        out_props = tuple(p for p in OUTPUT_PROPS if p in prop_resolution)
        out_cols = OUTPUT_BASE + out_props

        with output.open("w", newline="") as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=out_cols)
            writer.writeheader()

            for row in reader:
                # Depth filter (LUCAS-SOIL-2018 has 99% rows at "0-20 cm",
                # plus a few 0-10/10-20/20-30 cm follow-ups).
                if depth_col is not None and depth_filter is not None:
                    if str(row.get(depth_col, "")).strip() != depth_filter:
                        n_dropped_depth += 1
                        continue

                point_id = str(row.get(point_id_col, "")).strip()
                if not point_id:
                    continue

                # Resolve coordinates.
                if shapefile_coords is not None:
                    if point_id not in shapefile_coords:
                        n_dropped_coords += 1
                        continue
                    lon, lat = shapefile_coords[point_id]
                else:
                    lat = _safe_float(row.get(lat_col))
                    lon = _safe_float(row.get(lon_col))
                    if lat is None or lon is None:
                        n_dropped_coords += 1
                        continue

                out_row: dict[str, str] = {
                    "point_id": point_id,
                    "lon": f"{lon:.6f}",
                    "lat": f"{lat:.6f}",
                    "year": str(int(year)),
                }
                n_props_filled = 0
                for canonical in out_props:
                    src = prop_resolution[canonical]
                    v = _safe_float(row.get(src))
                    if v is None:
                        out_row[canonical] = ""
                    else:
                        out_row[canonical] = f"{v:.6f}"
                        n_props_filled += 1
                if n_props_filled == 0:
                    n_dropped_no_props += 1
                    continue
                writer.writerow(out_row)
                n_rows += 1

    log.info(
        "wrote %d rows to %s (dropped: depth=%d, coords=%d, no_props=%d)",
        n_rows, output, n_dropped_depth, n_dropped_coords, n_dropped_no_props,
    )
    return n_rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True,
                   help="ESDAC LUCAS 2018 properties CSV.")
    p.add_argument("--shapefile", type=Path, default=None,
                   help="Optional. Only needed if --csv lacks TH_LAT/TH_LONG "
                        "columns (older LUCAS releases). Requires "
                        "`pip install pyshp pyproj`.")
    p.add_argument("--output", type=Path, required=True,
                   help="Destination CSV path.")
    p.add_argument("--year", type=int, default=2018,
                   help="Survey year written into every row's `year` column.")
    p.add_argument("--depth-filter", type=str, default="0-20 cm",
                   help="Keep only rows whose Depth column equals this "
                        "(default: '0-20 cm', the canonical LUCAS topsoil "
                        "depth). Pass empty string to disable.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    n = convert(
        csv_in=args.csv,
        shapefile_path=args.shapefile,
        output=args.output,
        year=args.year,
        depth_filter=args.depth_filter if args.depth_filter else None,
    )
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "OUTPUT_COLUMNS",
    "convert",
    "main",
]
