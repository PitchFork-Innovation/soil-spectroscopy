"""Ground-truth ingestion: point-to-tile spatial join and aggregation.

Real ground-truth data (ISMN, LUCAS, SMAP, ...) arrives as point measurements
in EPSG:4326. To use it as supervision for the per-tile inference pipeline,
each point has to land on the same `tile_id` the Sentinel preprocessing
pathway emits in `tile_extraction`. This module provides:

* ``TileGrid`` — encodes the AOI's tile layout and inverts the same lon/lat
  math used by :func:`soilspec.preprocessing.spatial.tile_extraction`.
* ``Measurement`` — a single raw point observation.
* ``GroundTruthDataset`` — bucket measurements by (tile_id, time_bucket) and
  emit aggregated :class:`GroundTruthSample` records with mean + std-error
  uncertainty.

The aggregation is provider-agnostic. Source adapters (added in later steps)
construct ``Measurement`` records and feed them in.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import numpy as np

from .types import (
    AOI, BoundingBox, GroundTruthSample, MEASURED_PROPERTY_NAMES,
    PreprocessedRecord, TimeWindow,
)


@dataclass(frozen=True)
class TileGrid:
    """The tile layout for a given AOI + preprocessing config.

    Mirrors the partition produced by ``tile_extraction``: an ``n_rows`` ×
    ``n_cols`` grid spanning ``aoi_bbox`` with row 0 at the *top* (max_lat).
    Construct via :meth:`from_shape` to derive the row/col counts the same
    way the preprocessor does.
    """

    aoi_bbox: BoundingBox
    n_rows: int
    n_cols: int

    @classmethod
    def from_shape(
        cls, aoi_bbox: BoundingBox, raster_shape: tuple[int, int], tile_size: int
    ) -> "TileGrid":
        """Derive the grid from raster (H, W) and tile_size.

        Matches the ceiling-division used in
        :func:`soilspec.preprocessing.spatial.tile_extraction`.
        """
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        h, w = raster_shape
        n_rows = (h + tile_size - 1) // tile_size
        n_cols = (w + tile_size - 1) // tile_size
        return cls(aoi_bbox=aoi_bbox, n_rows=max(n_rows, 1), n_cols=max(n_cols, 1))

    @property
    def lon_step(self) -> float:
        return (self.aoi_bbox.max_lon - self.aoi_bbox.min_lon) / self.n_cols

    @property
    def lat_step(self) -> float:
        return (self.aoi_bbox.max_lat - self.aoi_bbox.min_lat) / self.n_rows

    def locate(self, lon: float, lat: float) -> str | None:
        """Return the ``tile_id`` containing (lon, lat), or None if outside.

        Boundary convention: tiles own their min edges and the grid's outer
        max edges, matching how tile_extraction would assign a pixel that
        falls exactly on the AOI's max_lon/min_lat to the last tile.
        """
        bb = self.aoi_bbox
        if lon < bb.min_lon or lon > bb.max_lon:
            return None
        if lat < bb.min_lat or lat > bb.max_lat:
            return None
        c = int(math.floor((lon - bb.min_lon) / self.lon_step))
        r = int(math.floor((bb.max_lat - lat) / self.lat_step))
        # Clamp the upper boundary (lon == max_lon, lat == min_lat) onto the
        # last tile — without this, points exactly on the AOI edge get
        # dropped.
        c = min(c, self.n_cols - 1)
        r = min(r, self.n_rows - 1)
        return tile_id(r, c)


def tile_id(row: int, col: int) -> str:
    """Format a (row, col) pair as the canonical tile id."""
    return f"r{row:03d}c{col:03d}"


@dataclass(frozen=True)
class Measurement:
    """A single raw ground-truth point measurement.

    ``properties`` keys should be drawn from
    :data:`soilspec.types.MEASURED_PROPERTY_NAMES`; unknown keys are passed
    through but won't be consumed by the training loop.
    """

    lon: float
    lat: float
    time: int
    properties: dict[str, float]
    source: str
    uncertainty: dict[str, float] = field(default_factory=dict)


class GroundTruthDataset:
    """In-memory aggregator: bucket measurements by (tile_id, time_bucket).

    Each (tile, bucket) emits one :class:`GroundTruthSample` with per-property
    mean and standard-error uncertainty. Out-of-AOI points are silently
    dropped (callers can check :attr:`dropped` for a count).
    """

    def __init__(self, grid: TileGrid, time_bucket_seconds: int = 86400) -> None:
        if time_bucket_seconds <= 0:
            raise ValueError("time_bucket_seconds must be positive")
        self.grid = grid
        self.time_bucket_seconds = time_bucket_seconds
        # (tile_id, bucket_time, source) -> property -> list[value]
        self._buckets: dict[
            tuple[str, int, str], dict[str, list[float]]
        ] = defaultdict(lambda: defaultdict(list))
        # Track per-measurement uncertainty values, used as a fallback when
        # only one observation lands in a bucket.
        self._reported_unc: dict[
            tuple[str, int, str], dict[str, list[float]]
        ] = defaultdict(lambda: defaultdict(list))
        self.dropped: int = 0

    # ----------------------------- ingestion --------------------------------

    def add(self, measurement: Measurement) -> None:
        tid = self.grid.locate(measurement.lon, measurement.lat)
        if tid is None:
            self.dropped += 1
            return
        bucket = self._bucket_time(measurement.time)
        key = (tid, bucket, measurement.source)
        for prop, val in measurement.properties.items():
            if val is None or not _is_finite(val):
                continue
            self._buckets[key][prop].append(float(val))
            unc = measurement.uncertainty.get(prop)
            if unc is not None and _is_finite(unc):
                self._reported_unc[key][prop].append(float(unc))

    def extend(self, measurements: Iterable[Measurement]) -> None:
        for m in measurements:
            self.add(m)

    # ----------------------------- output -----------------------------------

    def samples(self) -> Iterator[GroundTruthSample]:
        for (tid, bucket, source), props in sorted(self._buckets.items()):
            mean: dict[str, float] = {}
            unc: dict[str, float] = {}
            n_obs = 0
            for prop, vals in props.items():
                n = len(vals)
                if n == 0:
                    continue
                m = sum(vals) / n
                mean[prop] = m
                if n >= 2:
                    var = sum((v - m) ** 2 for v in vals) / (n - 1)
                    unc[prop] = math.sqrt(var / n)  # std-error of the mean
                else:
                    reported = self._reported_unc[(tid, bucket, source)].get(prop, [])
                    unc[prop] = reported[0] if reported else 0.0
                n_obs = max(n_obs, n)
            if not mean:
                continue
            yield GroundTruthSample(
                tile_id=tid,
                time=bucket,
                properties=mean,
                uncertainty=unc,
                n_observations=n_obs,
                source=source,
            )

    def __len__(self) -> int:
        return sum(1 for _ in self.samples())

    # ----------------------------- internals --------------------------------

    def _bucket_time(self, t: int) -> int:
        return (int(t) // self.time_bucket_seconds) * self.time_bucket_seconds


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def grid_for_request(
    aoi: AOI, target_shape: tuple[int, int], tile_size: int
) -> TileGrid:
    """Convenience: build the TileGrid the orchestrator would use."""
    return TileGrid.from_shape(aoi.bbox, target_shape, tile_size)


# ---------------------------------------------------------------------------
# Raster-level training examples (for end-to-end joint training)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RasterTrainingExamples:
    """Aligned (raster, label) rows for end-to-end pipeline training.

    Each row pairs a `PreprocessedRecord`'s S1 + S2 rasters and vector
    covariates with the per-property ground-truth label aggregated for that
    ``(tile_id, time_bucket)``. The trainer feeds raw rasters straight into
    the encoders so gradients flow all the way back from labels to encoder
    weights.

    Shape contract:
      - ``s1``: ``(N, B_s1, H, W)`` float32. Channel order matches the
        ingestion adapter's ``bands`` tuple.
      - ``s2``: ``(N, B_s2, H, W)`` float32.
      - ``vector_features``: ``(N, D_vec)`` float32 — one scalar per vector
        attribute (mean over the attribute's array).
      - ``y[prop]``: ``(N,)`` with NaN where the row had no label for that
        property. Trainers mask NaN per-property.
      - ``weights[prop]``: ``(N,)``; 1/σ² when uncertainty was reported,
        else ``n_observations``.
      - ``tile_keys``: ``((tile_id, bucket_start), ...)`` for spatial split.
    """

    s1: np.ndarray
    s2: np.ndarray
    vector_features: np.ndarray
    y: dict[str, np.ndarray]
    weights: dict[str, np.ndarray]
    tile_keys: tuple[tuple[str, int], ...]
    property_names: tuple[str, ...]
    sources: tuple[str, ...]
    s1_bands: int
    s2_bands: int
    vector_attr_names: tuple[str, ...]

    def __len__(self) -> int:
        return self.s1.shape[0]

    @property
    def n_features_vector(self) -> int:
        return self.vector_features.shape[1]

    def usable(self, prop: str) -> int:
        return int(np.sum(np.isfinite(self.y[prop])))


def assemble_raster_examples(
    records: Iterable[PreprocessedRecord],
    gt: "GroundTruthDataset",
    property_names: Iterable[str] = MEASURED_PROPERTY_NAMES,
    sources: Iterable[str] | None = None,
    s1_modality: str = "s1",
    s2_modality: str = "s2",
) -> RasterTrainingExamples:
    """Join `PreprocessedRecord`s (raw rasters per tile) with `GroundTruthDataset`
    samples on ``(tile_id, time_bucket)``.

    Filters out records that lack S1 or S2 (we need both to feed the dual
    encoders) or lack any finite label. The result feeds straight into the
    end-to-end :func:`soilspec.training.train_pipeline`.
    """
    props = tuple(property_names)
    allowed_sources = set(sources) if sources is not None else None
    bucket = gt.time_bucket_seconds

    # Index GT samples by bucket key.
    by_key: dict[tuple[str, int], list[GroundTruthSample]] = {}
    sources_seen: set[str] = set()
    for s in gt.samples():
        if allowed_sources is not None and s.source not in allowed_sources:
            continue
        sources_seen.add(s.source)
        by_key.setdefault((s.tile_id, s.time), []).append(s)

    rows_s1: list[np.ndarray] = []
    rows_s2: list[np.ndarray] = []
    rows_vec: list[np.ndarray] = []
    rows_y: dict[str, list[float]] = {p: [] for p in props}
    rows_w: dict[str, list[float]] = {p: [] for p in props}
    keys: list[tuple[str, int]] = []
    vector_keys: list[str] | None = None

    for rec in records:
        if s1_modality not in rec.spatial or s2_modality not in rec.spatial:
            continue
        bucket_start = (int(rec.time) // bucket) * bucket
        matched = by_key.get((rec.tile_id, bucket_start))
        if not matched:
            continue
        # build per-property labels first; skip rows with no usable labels
        any_label = False
        y_row: dict[str, float] = {}
        w_row: dict[str, float] = {}
        for p in props:
            mean, weight = _aggregate_label_for_raster(matched, p)
            y_row[p] = mean
            w_row[p] = weight
            if np.isfinite(mean) and weight > 0:
                any_label = True
        if not any_label:
            continue

        # Vector features: stable key order for cross-row alignment.
        if vector_keys is None:
            vector_keys = sorted(rec.vector.keys()) if rec.vector else []
        feat = np.array([
            float(np.nanmean(rec.vector[k])) if k in rec.vector and np.size(rec.vector[k])
            else 0.0
            for k in vector_keys
        ], dtype=np.float32)

        rows_s1.append(np.asarray(rec.spatial[s1_modality], dtype=np.float32))
        rows_s2.append(np.asarray(rec.spatial[s2_modality], dtype=np.float32))
        rows_vec.append(feat)
        keys.append((rec.tile_id, bucket_start))
        for p in props:
            rows_y[p].append(y_row[p])
            rows_w[p].append(w_row[p])

    if not rows_s1:
        return RasterTrainingExamples(
            s1=np.zeros((0, 0, 0, 0), dtype=np.float32),
            s2=np.zeros((0, 0, 0, 0), dtype=np.float32),
            vector_features=np.zeros((0, 0), dtype=np.float32),
            y={p: np.zeros(0, dtype=np.float64) for p in props},
            weights={p: np.zeros(0, dtype=np.float64) for p in props},
            tile_keys=(),
            property_names=props,
            sources=tuple(sorted(sources_seen)),
            s1_bands=0,
            s2_bands=0,
            vector_attr_names=tuple(vector_keys or ()),
        )

    s1 = np.stack(rows_s1, axis=0)
    s2 = np.stack(rows_s2, axis=0)
    vec = np.stack(rows_vec, axis=0) if vector_keys else np.zeros(
        (len(rows_s1), 0), dtype=np.float32
    )
    y = {p: np.asarray(rows_y[p], dtype=np.float64) for p in props}
    w = {p: np.asarray(rows_w[p], dtype=np.float64) for p in props}
    return RasterTrainingExamples(
        s1=s1, s2=s2, vector_features=vec,
        y=y, weights=w,
        tile_keys=tuple(keys),
        property_names=props,
        sources=tuple(sorted(sources_seen)),
        s1_bands=int(s1.shape[1]),
        s2_bands=int(s2.shape[1]),
        vector_attr_names=tuple(vector_keys or ()),
    )


def _aggregate_label_for_raster(
    samples: list[GroundTruthSample], prop: str,
) -> tuple[float, float]:
    """Inverse-variance weighted mean over samples; mirrors the helper in
    :mod:`soilspec.training` so raster assembly doesn't drag a dep cycle."""
    weighted_sum = 0.0
    weight_sum = 0.0
    for s in samples:
        v = s.properties.get(prop)
        if v is None or not np.isfinite(v):
            continue
        unc = s.uncertainty.get(prop, 0.0)
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
# Adapter protocol
# ---------------------------------------------------------------------------


class GroundTruthAdapter(Protocol):
    """Yields :class:`Measurement` records for a given AOI + time window.

    Adapters are responsible for: source-specific I/O (CSV, NetCDF, HTTP),
    AOI bbox + time-window filtering, unit conversion, and quality control.
    They produce Measurements in EPSG:4326 with epoch-second timestamps and
    property keys drawn from
    :data:`soilspec.types.MEASURED_PROPERTY_NAMES`.

    The point→tile join and aggregation happen downstream in
    :class:`GroundTruthDataset` — adapters never see ``tile_id``.
    """

    provider: str
    properties: tuple[str, ...]

    def fetch(self, aoi: AOI, window: TimeWindow) -> Iterable[Measurement]: ...


# ---------------------------------------------------------------------------
# ISMN adapter
# ---------------------------------------------------------------------------


# Quality-control flags from ISMN that we treat as usable. ISMN docs:
#   G = good, M = missing, D/C/* = various failure modes.
# Keeping only "G" matches the ISMN team's recommendation for ML training.
ISMN_GOOD_QC = frozenset({"G", "G_M", "good"})


@dataclass(frozen=True)
class ISMNAdapter:
    """Adapter for International Soil Moisture Network point time-series.

    Reads from a CSV file with the following columns (header required):

    ``network, station, lon, lat, timestamp, soil_moisture,
    depth_from, depth_to, qc_flag, [soil_moisture_uncertainty]``

    * ``timestamp`` is epoch seconds (int).
    * ``soil_moisture`` is volumetric water content in m³/m³.
    * Depth is centimetres.
    * ``soil_moisture_uncertainty`` is optional and may be empty.

    This shape matches what the upstream ``ismn`` Python package can be
    persisted to; a real-network mode (downloading + parsing the official
    ISMN archive) plugs in by producing a CSV in this layout.

    Surface measurements only by default (``max_depth_cm=10``) — Sentinel-1
    backscatter responds to the top few centimetres, so deeper sensors
    aren't useful supervision for the S1 encoder.
    """

    csv_path: str | Path
    provider: str = "ismn"
    properties: tuple[str, ...] = ("soil_moisture",)
    max_depth_cm: float = 10.0
    allowed_qc: frozenset[str] = ISMN_GOOD_QC

    def fetch(self, aoi: AOI, window: TimeWindow) -> Iterator[Measurement]:
        path = Path(self.csv_path)
        if not path.exists():
            raise FileNotFoundError(f"ISMN fixture not found: {path}")
        with path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            self._require_columns(reader.fieldnames, path)
            for row in reader:
                m = self._row_to_measurement(row, aoi, window)
                if m is not None:
                    yield m

    # ----------------------------- internals --------------------------------

    _REQUIRED = ("lon", "lat", "timestamp", "soil_moisture", "qc_flag")

    def _require_columns(self, fields: list[str] | None, path: Path) -> None:
        if not fields:
            raise ValueError(f"ISMN fixture {path} has no header row")
        missing = [c for c in self._REQUIRED if c not in fields]
        if missing:
            raise ValueError(
                f"ISMN fixture {path} missing required columns: {missing}"
            )

    def _row_to_measurement(
        self, row: dict[str, str], aoi: AOI, window: TimeWindow
    ) -> Measurement | None:
        qc = (row.get("qc_flag") or "").strip()
        if qc and qc not in self.allowed_qc:
            return None
        try:
            lon = float(row["lon"])
            lat = float(row["lat"])
            t = int(float(row["timestamp"]))  # tolerate "1.6e9"-style values
            sm = float(row["soil_moisture"])
        except (KeyError, TypeError, ValueError):
            return None
        if not _is_finite(sm):
            return None
        if not aoi.bbox.contains(lon, lat):
            return None
        if not window.contains(t):
            return None
        depth_to = _safe_float(row.get("depth_to"))
        if depth_to is not None and depth_to > self.max_depth_cm:
            return None
        unc_val = _safe_float(row.get("soil_moisture_uncertainty"))
        return Measurement(
            lon=lon,
            lat=lat,
            time=t,
            properties={"soil_moisture": sm},
            uncertainty={"soil_moisture": unc_val} if unc_val is not None else {},
            source=self.provider,
        )


def _safe_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------------
# SoilGrids adapter (static covariates from ISRIC)
# ---------------------------------------------------------------------------


# Map adapter property names → SoilGrids layer names. The right-hand side is
# what the real ISRIC REST API expects (e.g.
# rest.isric.org/soilgrids/v2.0/properties/query?property=clay&depth=0-5cm).
SOILGRIDS_LAYER_MAP: dict[str, str] = {
    "clay_pct": "clay",
    "sand_pct": "sand",
    "soc": "soc",
    "bulk_density": "bdod",
    "ph": "phh2o",
    "nitrogen": "nitrogen",
}


@dataclass(frozen=True)
class SoilGridsAdapter:
    """Adapter for SoilGrids 250m global soil property maps (ISRIC).

    SoilGrids is *static* — the layers don't vary in time — so each row's
    timestamp is set to ``window.start``. Downstream aggregation in
    :class:`GroundTruthDataset` treats SoilGrids samples as covariates by
    setting ``source="soilgrids"``: the trainer keeps them separate from
    label sources.

    Fixture format: a CSV with a ``lon`` and ``lat`` column plus one column
    per measured property (any subset of
    :data:`soilspec.types.MEASURED_PROPERTY_NAMES`). Optional
    ``depth_from``/``depth_to`` are passed through for downstream filtering;
    surface (0–5cm) is the default in real-mode queries.

    Real-mode: replace the CSV with a client that hits the ISRIC REST
    endpoint per tile centroid. The output shape (one Measurement per
    point) is identical, so this adapter can swap in transparently.
    """

    csv_path: str | Path
    provider: str = "soilgrids"
    properties: tuple[str, ...] = ("clay_pct", "sand_pct", "soc", "bulk_density", "ph")
    max_depth_cm: float = 5.0

    def fetch(self, aoi: AOI, window: TimeWindow) -> Iterator[Measurement]:
        path = Path(self.csv_path)
        if not path.exists():
            raise FileNotFoundError(f"SoilGrids fixture not found: {path}")
        with path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError(f"SoilGrids fixture {path} has no header row")
            for col in ("lon", "lat"):
                if col not in reader.fieldnames:
                    raise ValueError(
                        f"SoilGrids fixture {path} missing required column: {col!r}"
                    )
            present = [p for p in self.properties if p in reader.fieldnames]
            if not present:
                raise ValueError(
                    f"SoilGrids fixture {path} has none of the configured "
                    f"properties {self.properties}"
                )
            for row in reader:
                m = self._row_to_measurement(row, present, aoi, window)
                if m is not None:
                    yield m

    def _row_to_measurement(
        self,
        row: dict[str, str],
        present: list[str],
        aoi: AOI,
        window: TimeWindow,
    ) -> Measurement | None:
        lon = _safe_float(row.get("lon"))
        lat = _safe_float(row.get("lat"))
        if lon is None or lat is None:
            return None
        if not aoi.bbox.contains(lon, lat):
            return None
        depth_to = _safe_float(row.get("depth_to"))
        if depth_to is not None and depth_to > self.max_depth_cm:
            return None
        props: dict[str, float] = {}
        for p in present:
            v = _safe_float(row.get(p))
            if v is not None:
                props[p] = v
        if not props:
            return None
        return Measurement(
            lon=lon,
            lat=lat,
            time=int(window.start),
            properties=props,
            source=self.provider,
        )


# ---------------------------------------------------------------------------
# LUCAS adapter (EU soil sampling — N/P/K/pH/SOC/texture lab measurements)
# ---------------------------------------------------------------------------


# LUCAS units in the ESDAC release vs. the canonical units we use:
#   nitrogen:    g/kg     (matches)
#   phosphorus:  mg/kg    (matches)
#   potassium:   mg/kg    (matches)
#   ph_in_caCl2: pH       (matches)
#   ec:          mS/m     (not used)
#   oc:          g/kg     -> we call it "soc"
#   clay/sand/silt: %     (matches "clay_pct"/"sand_pct")
LUCAS_COLUMN_MAP: dict[str, str] = {
    "soc": "soc",
    "nitrogen": "nitrogen",
    "phosphorus": "phosphorus",
    "potassium": "potassium",
    "ph": "ph",
    "clay_pct": "clay_pct",
    "sand_pct": "sand_pct",
}


@dataclass(frozen=True)
class LUCASAdapter:
    """Adapter for the LUCAS Soil Survey topsoil lab measurements.

    LUCAS samples are point measurements with a single survey year; we
    convert ``year`` → epoch seconds at Jan 1 UTC so they bucket cleanly
    against satellite acquisitions in the same survey year.

    Fixture format: CSV with columns ``point_id, lon, lat, year`` plus any
    subset of the canonical measured-property columns. Missing or blank
    cells are skipped per-property (a row producing zero usable properties
    is dropped entirely).
    """

    csv_path: str | Path
    provider: str = "lucas"
    properties: tuple[str, ...] = (
        "soc", "nitrogen", "phosphorus", "potassium", "ph", "clay_pct", "sand_pct",
    )

    def fetch(self, aoi: AOI, window: TimeWindow) -> Iterator[Measurement]:
        path = Path(self.csv_path)
        if not path.exists():
            raise FileNotFoundError(f"LUCAS fixture not found: {path}")
        with path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError(f"LUCAS fixture {path} has no header row")
            for col in ("lon", "lat", "year"):
                if col not in reader.fieldnames:
                    raise ValueError(
                        f"LUCAS fixture {path} missing required column: {col!r}"
                    )
            present = [p for p in self.properties if p in reader.fieldnames]
            if not present:
                raise ValueError(
                    f"LUCAS fixture {path} has none of the configured "
                    f"properties {self.properties}"
                )
            for row in reader:
                m = self._row_to_measurement(row, present, aoi, window)
                if m is not None:
                    yield m

    def _row_to_measurement(
        self,
        row: dict[str, str],
        present: list[str],
        aoi: AOI,
        window: TimeWindow,
    ) -> Measurement | None:
        lon = _safe_float(row.get("lon"))
        lat = _safe_float(row.get("lat"))
        if lon is None or lat is None:
            return None
        if not aoi.bbox.contains(lon, lat):
            return None
        year_raw = _safe_float(row.get("year"))
        if year_raw is None:
            return None
        t = _year_to_epoch(int(year_raw))
        if not window.contains(t):
            return None
        props: dict[str, float] = {}
        for p in present:
            v = _safe_float(row.get(p))
            if v is not None:
                props[p] = v
        if not props:
            return None
        return Measurement(
            lon=lon,
            lat=lat,
            time=t,
            properties=props,
            source=self.provider,
        )


def _year_to_epoch(year: int) -> int:
    """Jan 1 of `year` UTC, in epoch seconds. Avoids a datetime import path."""
    # Days from 1970-01-01 to Jan 1 of `year`. Leap years: every /4 except
    # /100 not /400. Pre-computed via the proleptic Gregorian calendar.
    from datetime import datetime, timezone

    return int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())


# ---------------------------------------------------------------------------
# ISMN archive adapter (real network download mode)
# ---------------------------------------------------------------------------


class GroundTruthSourceError(RuntimeError):
    """Raised when a real-mode adapter cannot reach or read its source."""


@dataclass(frozen=True)
class ISMNArchiveAdapter:
    """Real-mode ISMN adapter that reads a downloaded ISMN archive.

    Users download the archive themselves from the ISMN portal
    (https://ismn.geo.tuwien.ac.at/ — registration required, account is
    free) and point this adapter at the resulting zip file or unpacked
    directory.

    Implementation uses the upstream ``ismn`` package
    (``pip install ismn``) which understands the ISMN header-values format
    natively, including QC flags and per-sensor depth metadata.

    Filtering applied:
      * surface sensors only (``min_depth=0, max_depth=max_depth_m``)
      * variable = ``soil_moisture``
      * QC flag is in :data:`ISMN_GOOD_QC` (default: "G" only)
      * point inside ``aoi.bbox`` and time inside ``window``
    """

    archive_path: str | Path
    provider: str = "ismn"
    properties: tuple[str, ...] = ("soil_moisture",)
    max_depth_m: float = 0.10
    allowed_qc: frozenset[str] = ISMN_GOOD_QC

    def fetch(self, aoi: AOI, window: TimeWindow) -> Iterator[Measurement]:
        path = Path(self.archive_path)
        if not path.exists():
            raise FileNotFoundError(f"ISMN archive not found: {path}")
        try:
            from ismn.interface import ISMN_Interface  # type: ignore[import-not-found]
        except ImportError as e:
            raise GroundTruthSourceError(
                "ISMNArchiveAdapter requires the `ismn` package. "
                "Install with: pip install ismn"
            ) from e
        try:
            interface = ISMN_Interface(str(path))
        except Exception as e:
            raise GroundTruthSourceError(
                f"failed to open ISMN archive {path}: {e}"
            ) from e
        yield from self._iter_readings(interface, aoi, window)

    def _iter_readings(
        self, interface, aoi: AOI, window: TimeWindow
    ) -> Iterator[Measurement]:
        """Drive the interface; isolated so tests can mock it."""
        from datetime import datetime, timezone

        ids = interface.get_dataset_ids(
            variable="soil_moisture", min_depth=0.0, max_depth=self.max_depth_m,
        )
        for idx in ids:
            ts, meta = interface.read_ts(idx, return_meta=True)
            lon = float(meta.get("longitude") if hasattr(meta, "get") else meta["longitude"])
            lat = float(meta.get("latitude") if hasattr(meta, "get") else meta["latitude"])
            if not aoi.bbox.contains(lon, lat):
                continue
            # ts is a pandas DataFrame indexed by timestamp; columns include
            # "soil_moisture" and "soil_moisture_flag". Iterate row-by-row
            # without pulling pandas into our import path.
            for row_idx, row in ts.iterrows():
                qc = str(row.get("soil_moisture_flag", "")).strip()
                if qc and qc not in self.allowed_qc:
                    continue
                val = row.get("soil_moisture")
                if val is None or not _is_finite(val):
                    continue
                # row index is a pandas Timestamp; convert to epoch seconds
                if hasattr(row_idx, "to_pydatetime"):
                    dt = row_idx.to_pydatetime()
                else:
                    dt = row_idx
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                t = int(dt.timestamp())
                if not window.contains(t):
                    continue
                yield Measurement(
                    lon=lon, lat=lat, time=t,
                    properties={"soil_moisture": float(val)},
                    source=self.provider,
                )


# ---------------------------------------------------------------------------
# SoilGrids REST adapter (ISRIC v2.0 — anonymous, no API key)
# ---------------------------------------------------------------------------


SOILGRIDS_REST_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# SoilGrids returns scaled integers; divide by these to get our canonical
# units. Source: https://www.isric.org/explore/soilgrids/faq-soilgrids
SOILGRIDS_UNIT_DIVISORS: dict[str, float] = {
    "clay": 10.0,        # dg/kg -> %
    "sand": 10.0,        # dg/kg -> %
    "silt": 10.0,        # dg/kg -> %
    "soc": 10.0,         # dg/kg -> g/kg
    "bdod": 100.0,       # cg/cm3 -> g/cm3
    "phh2o": 10.0,       # pH*10 -> pH
    "nitrogen": 100.0,   # cg/kg -> g/kg
}

# Inverse of SOILGRIDS_LAYER_MAP.
_PROP_FROM_LAYER: dict[str, str] = {v: k for k, v in SOILGRIDS_LAYER_MAP.items()}


@dataclass(frozen=True)
class SoilGridsRESTAdapter:
    """Real-mode SoilGrids adapter querying the ISRIC REST API.

    Samples soil property values at the centroids of an ``n_rows × n_cols``
    grid spanning ``aoi.bbox`` (defaults to a 4×4 sample grid — keep this
    small, the API rate-limits aggressive callers). Each sample becomes a
    :class:`Measurement` tagged ``source="soilgrids"``.

    Static layers — the timestamp on every Measurement is ``window.start``.

    No auth required. For reproducibility, real-mode users should cache
    responses to disk (see :meth:`fetch_with_cache`).
    """

    provider: str = "soilgrids"
    properties: tuple[str, ...] = ("clay_pct", "sand_pct", "soc", "bulk_density", "ph")
    depth: str = "0-5cm"
    sample_rows: int = 4
    sample_cols: int = 4
    timeout_seconds: float = 30.0

    def fetch(self, aoi: AOI, window: TimeWindow) -> Iterator[Measurement]:
        layers = self._layers_for_properties()
        if not layers:
            raise ValueError(
                f"No SoilGrids layer mapping for properties={self.properties}"
            )
        for lon, lat in self._sample_points(aoi):
            payload = self._query(lon, lat, layers)
            props = self._parse_properties(payload)
            if props:
                yield Measurement(
                    lon=lon, lat=lat, time=int(window.start),
                    properties=props, source=self.provider,
                )

    # --- pure helpers (test directly) --------------------------------------

    def _layers_for_properties(self) -> list[str]:
        return [SOILGRIDS_LAYER_MAP[p] for p in self.properties
                if p in SOILGRIDS_LAYER_MAP]

    def _sample_points(self, aoi: AOI) -> Iterator[tuple[float, float]]:
        """Centroids of an ``sample_rows × sample_cols`` grid over the AOI."""
        bb = aoi.bbox
        for r in range(self.sample_rows):
            for c in range(self.sample_cols):
                lon = bb.min_lon + (c + 0.5) * (bb.max_lon - bb.min_lon) / self.sample_cols
                lat = bb.min_lat + (r + 0.5) * (bb.max_lat - bb.min_lat) / self.sample_rows
                yield lon, lat

    def _query(self, lon: float, lat: float, layers: list[str]) -> dict:
        import json
        from urllib.parse import urlencode
        from urllib.request import Request, urlopen
        from urllib.error import URLError

        params = [("lon", f"{lon:.6f}"), ("lat", f"{lat:.6f}"),
                  ("depth", self.depth), ("value", "mean")]
        params.extend(("property", layer) for layer in layers)
        url = f"{SOILGRIDS_REST_URL}?{urlencode(params)}"
        req = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, OSError) as e:
            raise GroundTruthSourceError(
                f"SoilGrids query failed for ({lon}, {lat}): {e}"
            ) from e

    def _parse_properties(self, payload: dict) -> dict[str, float]:
        """Extract canonical-unit values from a SoilGrids JSON response."""
        out: dict[str, float] = {}
        layers = (payload.get("properties") or {}).get("layers") or []
        for layer in layers:
            name = layer.get("name")
            prop = _PROP_FROM_LAYER.get(name)
            if prop is None:
                continue
            depths = layer.get("depths") or []
            target = next(
                (d for d in depths if d.get("label") == self.depth), None
            )
            if target is None:
                continue
            mean = (target.get("values") or {}).get("mean")
            if mean is None:
                continue
            divisor = SOILGRIDS_UNIT_DIVISORS.get(name, 1.0)
            try:
                out[prop] = float(mean) / divisor
            except (TypeError, ValueError):
                continue
        return out


# ---------------------------------------------------------------------------
# LUCAS — ESDAC official-CSV column mapping
# ---------------------------------------------------------------------------


# Maps canonical (our) property name → list of candidate column names found
# in real ESDAC LUCAS Topsoil distributions across years (2009/2015/2018).
# The first column present in the actual CSV wins. ESDAC has shifted naming
# between releases so we tolerate the spread.
LUCAS_ESDAC_COLUMNS: dict[str, tuple[str, ...]] = {
    "soc": ("OC", "OC_(g/kg)", "oc"),
    "nitrogen": ("N", "N_(g/kg)", "nitrogen"),
    "phosphorus": ("P", "P_(mg/kg)", "phosphorus"),
    "potassium": ("K", "K_(mg/kg)", "potassium"),
    "ph": ("pH_CaCl2", "pH(CaCl2)", "pH_in_CaCl2", "ph"),
    "clay_pct": ("Clay", "Clay_(%)", "clay"),
    "sand_pct": ("Sand", "Sand_(%)", "sand"),
}

LUCAS_ESDAC_LON_COLS: tuple[str, ...] = ("GPS_LONG", "GPS_LON", "lon", "Longitude")
LUCAS_ESDAC_LAT_COLS: tuple[str, ...] = ("GPS_LAT", "lat", "Latitude")
LUCAS_ESDAC_YEAR_COLS: tuple[str, ...] = ("SURVEY_YEAR", "Year", "year")


def normalize_lucas_esdac_csv(
    src_path: str | Path, dst_path: str | Path,
) -> Path:
    """Rewrite an ESDAC LUCAS CSV into the schema :class:`LUCASAdapter` expects.

    The official ESDAC release uses different column names per survey year
    and sometimes uses commas as decimal separators. This helper auto-detects
    columns from :data:`LUCAS_ESDAC_COLUMNS` and emits a normalized CSV that
    plugs straight into ``LUCASAdapter(csv_path=normalized)``.

    Returns the destination path.
    """
    src = Path(src_path)
    dst = Path(dst_path)
    if not src.exists():
        raise FileNotFoundError(f"LUCAS ESDAC CSV not found: {src}")
    with src.open("r", newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        # Decide separator: ESDAC files are usually ',' but some are ';'
        sep = ";" if sample.count(";") > sample.count(",") else ","
        reader = csv.DictReader(fh, delimiter=sep)
        if not reader.fieldnames:
            raise ValueError(f"empty header in {src}")
        lon_col = _first_present(reader.fieldnames, LUCAS_ESDAC_LON_COLS)
        lat_col = _first_present(reader.fieldnames, LUCAS_ESDAC_LAT_COLS)
        year_col = _first_present(reader.fieldnames, LUCAS_ESDAC_YEAR_COLS)
        if not (lon_col and lat_col and year_col):
            raise ValueError(
                f"could not locate lon/lat/year columns in {src}: "
                f"got fields={reader.fieldnames!r}"
            )
        prop_cols: dict[str, str] = {}
        for canonical, candidates in LUCAS_ESDAC_COLUMNS.items():
            col = _first_present(reader.fieldnames, candidates)
            if col is not None:
                prop_cols[canonical] = col

        out_fields = ["point_id", "lon", "lat", "year"] + list(prop_cols.keys())
        with dst.open("w", newline="") as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=out_fields)
            writer.writeheader()
            for i, row in enumerate(reader):
                lon = _decimal_comma_float(row.get(lon_col))
                lat = _decimal_comma_float(row.get(lat_col))
                year = _decimal_comma_float(row.get(year_col))
                if lon is None or lat is None or year is None:
                    continue
                rec = {
                    "point_id": row.get("POINT_ID") or row.get("point_id") or f"row{i}",
                    "lon": f"{lon:.6f}",
                    "lat": f"{lat:.6f}",
                    "year": str(int(year)),
                }
                for canonical, src_col in prop_cols.items():
                    v = _decimal_comma_float(row.get(src_col))
                    rec[canonical] = "" if v is None else f"{v:g}"
                writer.writerow(rec)
    return dst


def _first_present(fields: list[str], candidates: Iterable[str]) -> str | None:
    fset = set(fields)
    for c in candidates:
        if c in fset:
            return c
    return None


def _decimal_comma_float(s: str | None) -> float | None:
    """Parse "1,23" as 1.23 (European decimal-comma convention) and "1.23" as 1.23."""
    if s is None or s == "":
        return None
    try:
        return float(s.replace(",", "."))
    except (ValueError, AttributeError):
        return None


__all__ = [
    "TileGrid",
    "tile_id",
    "Measurement",
    "GroundTruthDataset",
    "GroundTruthAdapter",
    "ISMNAdapter",
    "ISMNArchiveAdapter",
    "SoilGridsAdapter",
    "SoilGridsRESTAdapter",
    "LUCASAdapter",
    "GroundTruthSourceError",
    "RasterTrainingExamples",
    "assemble_raster_examples",
    "SOILGRIDS_LAYER_MAP",
    "SOILGRIDS_UNIT_DIVISORS",
    "LUCAS_ESDAC_COLUMNS",
    "normalize_lucas_esdac_csv",
    "grid_for_request",
]
