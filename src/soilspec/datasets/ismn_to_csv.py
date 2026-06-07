"""Convert one or more ISMN archives into the flat CSV that :class:`ISMNAdapter`
reads.

ISMN ships per-station ``.stm`` (Header+Values) files inside a hierarchical
zip. Our :class:`soilspec.groundtruth.ISMNAdapter` instead consumes a single
flat CSV with this header::

    network, station, lon, lat, timestamp, soil_moisture,
    depth_from, depth_to, qc_flag

This module bridges the two. Given one or more archives (passed via
``--archive`` multiple times), it walks every soil-moisture sensor using
the official ``ismn`` Python package, applies depth / QC / time filters,
deduplicates rows across archives, and writes the merged CSV.

Setup::

    pip install ismn

Run::

    python -m soilspec.datasets.ismn_to_csv \\
        --archive ~/Downloads/Data_2023_first.zip \\
        --archive ~/Downloads/Data_2023_second.zip \\
        --output data/groundtruth/ismn_2023.csv \\
        --max-depth-cm 30

The output CSV is then passed to ``build.py --ismn-csv``.

Why two ``--archive`` flags? The recommended workflow downloaded ISMN data
in two requests (first batch with strict 0-10 cm depth, second batch with
relaxed 0-30 cm depth + wider time range to catch COSMOS-UK / OzNet /
TWENTE). Both archives feed in here; identical sensor-timestamps are
deduplicated automatically.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


# Column order must match :class:`ISMNAdapter._REQUIRED` so the adapter's
# header check passes without surprises. ``soil_moisture_uncertainty`` is
# omitted because per-measurement uncertainty is not reported in ISMN's
# standard sensor files — the adapter falls back to ``n_observations``
# weighting when this column is absent.
CSV_COLUMNS: tuple[str, ...] = (
    "network",
    "station",
    "lon",
    "lat",
    "timestamp",
    "soil_moisture",
    "depth_from",
    "depth_to",
    "qc_flag",
)

# Match :data:`ISMNAdapter.ISMN_GOOD_QC` — anything outside this set is
# filtered out at conversion time so the adapter never sees questionable
# rows.
GOOD_QC: frozenset[str] = frozenset({"G", "G_M"})


def _open_archive(archive_path: Path):
    """Open one ISMN archive via the official ismn package."""
    try:
        from ismn.interface import ISMN_Interface
    except ImportError as e:
        raise RuntimeError(
            "the `ismn` package is required for this converter: "
            "pip install ismn"
        ) from e
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    return ISMN_Interface(str(archive_path))


def _metadata_value(metadata, key: str, default=None):
    """Pull a value out of an ismn metadata object, tolerating layout drift.

    ``ismn`` has gone through multiple metadata-container shapes
    (``MetadataDict``, ``Metadata``, plain dict). Treating it as duck-typed
    keeps us robust across recent versions.
    """
    try:
        entry = metadata[key]
    except (KeyError, TypeError):
        return default
    val = getattr(entry, "val", entry)
    return val if val is not None else default


def iter_archive_rows(
    archive_path: Path,
    max_depth_cm: float,
    start_epoch: int | None,
    end_epoch: int | None,
    qc_filter: frozenset[str],
) -> Iterator[dict[str, str]]:
    """Yield one CSV row per (sensor, timestamp) that passes every filter."""
    iface = _open_archive(archive_path)

    # The ismn API has two iteration styles depending on version. Prefer
    # ``iter_sensors``; fall back to the network->station->sensor walk if
    # it's not available.
    if hasattr(iface, "collection") and hasattr(iface.collection, "iter_sensors"):
        sensor_iter = iface.collection.iter_sensors(variable="soil_moisture")
    elif hasattr(iface, "iter_sensors"):
        sensor_iter = iface.iter_sensors(variable="soil_moisture")
    else:
        sensor_iter = _legacy_sensor_walk(iface)

    for item in sensor_iter:
        # iter_sensors yields either a Sensor or a (network, station, sensor)
        # tuple depending on version — normalize.
        if isinstance(item, tuple):
            _, _, sensor = item
        else:
            sensor = item

        md = sensor.metadata
        depth_to = _metadata_value(md, "depth_to", default=0.0)
        try:
            depth_to_f = float(depth_to)
        except (TypeError, ValueError):
            continue
        if depth_to_f > max_depth_cm:
            continue

        depth_from = _metadata_value(md, "depth_from", default=0.0)
        try:
            depth_from_f = float(depth_from)
        except (TypeError, ValueError):
            depth_from_f = 0.0

        lon = _metadata_value(md, "longitude")
        lat = _metadata_value(md, "latitude")
        if lon is None or lat is None:
            continue
        try:
            lon_f = float(lon)
            lat_f = float(lat)
        except (TypeError, ValueError):
            continue

        network_name = _metadata_value(md, "network", default="")
        station_name = _metadata_value(md, "station", default="")

        try:
            ts = sensor.read_data()
        except Exception as e:
            log.warning(
                "skipping sensor %s/%s (depth %s-%s): read_data failed: %s",
                network_name, station_name, depth_from_f, depth_to_f, e,
            )
            continue
        if ts is None or len(ts) == 0:
            continue

        # The DataFrame schema varies; we want the soil_moisture value
        # column and the matching flag column. Find them by suffix.
        sm_cols = [c for c in ts.columns if c == "soil_moisture"]
        if not sm_cols:
            sm_cols = [c for c in ts.columns if "soil_moisture" in c and "flag" not in c]
        flag_cols = [c for c in ts.columns if "soil_moisture" in c and "flag" in c]
        if not sm_cols:
            continue
        sm_col = sm_cols[0]
        flag_col = flag_cols[0] if flag_cols else None

        for idx, row in ts.iterrows():
            sm = row[sm_col]
            # NaN check that avoids importing numpy/pandas here.
            if sm is None or sm != sm:
                continue

            flag_raw = row.get(flag_col) if flag_col else "G"
            flag = str(flag_raw).strip() if flag_raw is not None else ""
            if flag not in qc_filter:
                continue

            epoch = int(idx.timestamp())
            if start_epoch is not None and epoch < start_epoch:
                continue
            if end_epoch is not None and epoch > end_epoch:
                continue

            yield {
                "network": str(network_name),
                "station": str(station_name),
                "lon": f"{lon_f:.6f}",
                "lat": f"{lat_f:.6f}",
                "timestamp": str(epoch),
                "soil_moisture": f"{float(sm):.6f}",
                "depth_from": f"{depth_from_f:.2f}",
                "depth_to": f"{depth_to_f:.2f}",
                "qc_flag": flag,
            }


def _legacy_sensor_walk(iface):
    """Fallback iteration over networks/stations/sensors for older ismn."""
    for nw in iface.networks.values():
        for st in nw.stations.values():
            sensors = getattr(st, "sensors", None)
            if sensors is not None:
                for sensor in sensors.values():
                    if "soil_moisture" in getattr(sensor, "variable", ""):
                        yield sensor


def convert(
    archives: list[Path],
    output: Path,
    max_depth_cm: float = 30.0,
    start_epoch: int | None = None,
    end_epoch: int | None = None,
    qc_filter: frozenset[str] = GOOD_QC,
) -> int:
    """Drive the conversion. Returns the number of rows written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    # Dedup key: (network, station, depth_from, depth_to, timestamp).
    # We deliberately *don't* key on sensor name — two of the same physical
    # sensor reading at the same time will be in only one archive.
    seen: set[tuple[str, str, str, str, str]] = set()
    n_total = 0
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for archive in archives:
            log.info("reading archive: %s", archive)
            n_archive = 0
            n_dup = 0
            for row in iter_archive_rows(
                archive,
                max_depth_cm=max_depth_cm,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                qc_filter=qc_filter,
            ):
                key = (
                    row["network"], row["station"],
                    row["depth_from"], row["depth_to"], row["timestamp"],
                )
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                writer.writerow(row)
                n_archive += 1
            n_total += n_archive
            log.info(
                "  archive %s: wrote %d rows (%d duplicates skipped)",
                archive.name, n_archive, n_dup,
            )
    log.info("wrote %d rows to %s", n_total, output)
    return n_total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--archive", type=Path, action="append", required=True,
        help="Path to an ISMN .zip (or extracted directory). "
             "Pass multiple times to merge.",
    )
    p.add_argument("--output", type=Path, required=True,
                   help="Destination CSV path.")
    p.add_argument(
        "--max-depth-cm", type=float, default=30.0,
        help="Drop sensors with depth_to > this. Default 30 matches the "
             "build driver's max_depth_cm so COSMOS-UK passes through.",
    )
    p.add_argument("--start-epoch", type=int, default=None,
                   help="Optional: drop rows before this UTC epoch second.")
    p.add_argument("--end-epoch", type=int, default=None,
                   help="Optional: drop rows after this UTC epoch second.")
    p.add_argument("--allow-qc", action="append", default=None,
                   help="QC flag to accept (repeatable). "
                        f"Default: {sorted(GOOD_QC)}")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    qc_filter = frozenset(args.allow_qc) if args.allow_qc else GOOD_QC
    n = convert(
        archives=list(args.archive),
        output=args.output,
        max_depth_cm=args.max_depth_cm,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        qc_filter=qc_filter,
    )
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "CSV_COLUMNS",
    "GOOD_QC",
    "convert",
    "iter_archive_rows",
    "main",
]
