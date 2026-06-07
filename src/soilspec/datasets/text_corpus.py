"""Build a text-embedding corpus aligned to the training tile grid.

The joint trainer's text modality consumes `TextRecord`s keyed on
``(tile_id, time_bucket)``. This module produces those records from
real-world text sources tied to physical sample locations:

* ISMN station metadata — one descriptive snippet per soil-moisture
  station (network, station name, location, sensor depth range).
* LUCAS topsoil samples — one snippet per LUCAS point (point id,
  location, survey year, observed soil property values).

Each snippet is encoded once by a frozen external encoder
(``TextEncoder``), then broadcast across every relevant time bucket
inside the snippet's source AOI window so per-tile rasters at any
acquisition time can match.

The output is a ``.npz`` consumable by the ``--text-npz`` flag of
``python -m soilspec.train``.

Encoders are pluggable. The default :class:`HashTextEncoder` requires
nothing beyond numpy and is deterministic — useful for offline /
CI scaffolding. :class:`SentenceTransformerEncoder` wraps the
`sentence-transformers` package (lazy import) and is the real
production path.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import numpy as np

from ..groundtruth import TextRecord, TileGrid
from ..types import AOI, BoundingBox, TimeWindow


# ---------------------------------------------------------------------------
# Encoder protocol and built-in impls
# ---------------------------------------------------------------------------


class TextEncoder(Protocol):
    """Frozen, batch-friendly text encoder.

    Implementations should be deterministic for a given input — the
    pipeline persists the encoder *name* on the trained model so that
    inference-time text gets embedded by the same model used at train.
    """

    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class HashTextEncoder:
    """Deterministic numpy-only encoder, intended for tests + offline dev.

    Produces a fixed-dim float32 embedding by hashing the input string
    and seeding a numpy RNG — no actual semantic content, but stable.
    Useful when sentence-transformers isn't installed and you just want
    to validate the data-plumbing end-to-end.
    """

    dim: int = 64
    name: str = "hash64"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest()
            seed = int.from_bytes(digest, "little", signed=False) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.normal(size=self.dim).astype(np.float32)
            # L2-normalise so downstream LayerNorm has consistent scale.
            n = float(np.linalg.norm(v))
            out[i] = v / n if n > 0 else v
        return out


@dataclass
class SentenceTransformerEncoder:
    """Wrapper around `sentence-transformers`. Lazy-loads on first encode.

    ``model_name`` is anything the upstream library accepts
    (e.g. ``"all-MiniLM-L6-v2"``, ``"BAAI/bge-small-en-v1.5"``). The
    chosen name is persisted on the trained model so inference-time
    text gets the same embedding distribution.
    """

    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    _model: object = None
    _dim: int = 0

    @property
    def name(self) -> str:
        return f"sentence-transformers/{self.model_name}"

    @property
    def dim(self) -> int:
        if self._dim == 0:
            self._ensure_loaded()
        return self._dim

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "SentenceTransformerEncoder requires the `sentence-transformers` "
                "package. Install with: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(self.model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        emb = self._model.encode(  # type: ignore[union-attr]
            texts, batch_size=self.batch_size,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)


# ---------------------------------------------------------------------------
# Snippet extractors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Snippet:
    """One descriptive text tied to a fixed physical location.

    The corpus builder downstream broadcasts the same embedding across
    every time bucket inside the snippet's source AOI window — handy
    because the descriptive text is itself time-invariant.
    """

    doc_id: str
    lon: float
    lat: float
    text: str
    source: str  # "ismn" | "lucas"


def ismn_station_snippets(csv_path: str | Path) -> Iterator[Snippet]:
    """One snippet per (network, station) found in an ISMN CSV.

    Aggregates depth ranges across rows of the same station so the
    text mentions the actual sensor coverage. ``doc_id`` is
    ``"{network}/{station}"`` — stable across runs and unique to a
    physical sensor.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"ISMN CSV not found: {path}")
    by_station: dict[
        tuple[str, str], dict[str, object]
    ] = defaultdict(lambda: {
        "lon": math.nan, "lat": math.nan,
        "depth_min": math.inf, "depth_max": -math.inf, "n": 0,
    })
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                network = row["network"].strip()
                station = row["station"].strip()
                lon = float(row["lon"])
                lat = float(row["lat"])
            except (KeyError, ValueError):
                continue
            entry = by_station[(network, station)]
            entry["lon"] = lon
            entry["lat"] = lat
            entry["n"] = int(entry["n"]) + 1
            d_from = _safe_float(row.get("depth_from"))
            d_to = _safe_float(row.get("depth_to"))
            if d_from is not None:
                entry["depth_min"] = min(float(entry["depth_min"]), d_from)
            if d_to is not None:
                entry["depth_max"] = max(float(entry["depth_max"]), d_to)
    for (network, station), e in by_station.items():
        depth_text = _format_depth(
            float(e["depth_min"]), float(e["depth_max"])
        )
        text = (
            f"ISMN soil-moisture station {station} in network {network}, "
            f"located at latitude {float(e['lat']):.4f}, "
            f"longitude {float(e['lon']):.4f}. "
            f"{depth_text} "
            f"Volumetric water content sensor."
        )
        yield Snippet(
            doc_id=f"{network}/{station}",
            lon=float(e["lon"]), lat=float(e["lat"]),
            text=text, source="ismn",
        )


def lucas_point_snippets(csv_path: str | Path) -> Iterator[Snippet]:
    """One snippet per LUCAS topsoil sampling point.

    The text mentions the observed property values (SOC, N, P, K, pH,
    clay/sand %) so the encoder picks up agronomic context. ``doc_id``
    is the LUCAS point id.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"LUCAS CSV not found: {path}")
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                point_id = (row.get("point_id") or "").strip()
                lon = float(row["lon"])
                lat = float(row["lat"])
                year = int(float(row["year"])) if row.get("year") else None
            except (KeyError, ValueError):
                continue
            if not point_id:
                continue
            parts = [
                f"LUCAS topsoil sample {point_id}",
                f"at latitude {lat:.4f}, longitude {lon:.4f}",
            ]
            if year is not None:
                parts.append(f"surveyed in {year}")
            obs = []
            for canonical, label in (
                ("soc", "soil organic carbon"),
                ("nitrogen", "total nitrogen"),
                ("phosphorus", "extractable phosphorus"),
                ("potassium", "extractable potassium"),
                ("ph", "pH in CaCl2"),
                ("clay_pct", "clay percent"),
                ("sand_pct", "sand percent"),
            ):
                v = _safe_float(row.get(canonical))
                if v is not None:
                    obs.append(f"{label} {v:g}")
            if obs:
                parts.append("Measured: " + "; ".join(obs) + ".")
            text = ". ".join(parts) + "."
            yield Snippet(
                doc_id=f"lucas/{point_id}",
                lon=lon, lat=lat,
                text=text, source="lucas",
            )


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------


def build_text_corpus(
    snippets: Iterable[Snippet],
    aois: Iterable["AOIBinding"],
    encoder: TextEncoder,
    tile_size: int,
    target_shape: tuple[int, int],
    bucket_seconds: int = 86400,
    time_strategy: str = "window_start",
) -> tuple[list[TextRecord], str]:
    """Embed snippets and emit ``TextRecord``s for every aligned tile bucket.

    For each snippet:
      1. For every AOI binding whose bbox contains the snippet's lon/lat,
         compute the tile_id under that AOI's TileGrid.
      2. Emit text records at one or more time buckets covering the AOI's
         time window. ``time_strategy``:
           * ``"window_start"`` — single record at ``window.start`` bucket
             (cheapest; aligns only when rasters happen to land on the same
             bucket).
           * ``"daily"`` — one record per day inside the window
             (heaviest; maximizes alignment with arbitrary acquisition
             times).
      3. Each record carries the same embedding (the snippet is static),
         so we encode each unique snippet exactly once.

    Returns ``(records, encoder_name)``.
    """
    snippets = list(snippets)
    if not snippets:
        return [], encoder.name
    texts = [s.text for s in snippets]
    embeddings = encoder.encode(texts)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(snippets):
        raise ValueError(
            f"encoder returned wrong shape: {embeddings.shape}, "
            f"expected ({len(snippets)}, D)"
        )
    grids = [
        (binding, TileGrid.from_shape(
            binding.aoi.bbox, target_shape, tile_size,
        ))
        for binding in aois
    ]

    out: list[TextRecord] = []
    for i, snip in enumerate(snippets):
        emb = embeddings[i]
        for binding, grid in grids:
            if not binding.aoi.bbox.contains(snip.lon, snip.lat):
                continue
            tid = grid.locate(snip.lon, snip.lat)
            if tid is None:
                continue
            namespaced = f"{binding.aoi.aoi_id}/{tid}"
            for t in _bucket_times(
                binding.window, bucket_seconds, time_strategy,
            ):
                out.append(TextRecord(
                    tile_id=namespaced, time=t, embedding=emb,
                    doc_id=snip.doc_id, encoder=encoder.name,
                ))
    return out, encoder.name


@dataclass(frozen=True)
class AOIBinding:
    """An AOI + its time window — minimal subset of ``AOIWindow``.

    Keeping a thin local type avoids a hard dependency on
    ``initial_aois`` so callers can pass arbitrary AOI lists.
    """

    aoi: AOI
    window: TimeWindow


def _bucket_times(
    window: TimeWindow, bucket_seconds: int, strategy: str,
) -> list[int]:
    start = (int(window.start) // bucket_seconds) * bucket_seconds
    end = (int(window.end) // bucket_seconds) * bucket_seconds
    if strategy == "window_start":
        return [start]
    if strategy == "daily":
        if end < start:
            return [start]
        return list(range(start, end + 1, bucket_seconds))
    raise ValueError(
        f"unknown time_strategy {strategy!r}; "
        f"use 'window_start' or 'daily'"
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_text_corpus(
    records: list[TextRecord], encoder_name: str, out_path: str | Path,
) -> Path:
    """Write a ``.npz`` consumable by ``python -m soilspec.train --text-npz``.

    Arrays produced:
      * ``tile_ids`` — (N,) unicode
      * ``times`` — (N,) int64 epoch seconds (already bucketed)
      * ``embeddings`` — (N, D) float32
      * ``doc_ids`` — (N,) unicode
      * ``encoder`` — () unicode, the encoder identifier
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        np.savez(
            out,
            tile_ids=np.array([], dtype=np.str_),
            times=np.array([], dtype=np.int64),
            embeddings=np.zeros((0, 0), dtype=np.float32),
            doc_ids=np.array([], dtype=np.str_),
            encoder=np.array(encoder_name, dtype=np.str_),
        )
        return out
    tile_ids = np.array([r.tile_id for r in records], dtype=np.str_)
    times = np.array([int(r.time) for r in records], dtype=np.int64)
    doc_ids = np.array([r.doc_id for r in records], dtype=np.str_)
    embeddings = np.stack([
        np.asarray(r.embedding, dtype=np.float32) for r in records
    ], axis=0)
    np.savez(
        out,
        tile_ids=tile_ids, times=times,
        embeddings=embeddings, doc_ids=doc_ids,
        encoder=np.array(encoder_name, dtype=np.str_),
    )
    return out


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------


def encoder_from_spec(spec: str) -> TextEncoder:
    """Resolve a CLI/config string into a :class:`TextEncoder`.

    Accepted:
      * ``"hash"`` or ``"hash:<dim>"`` — :class:`HashTextEncoder`.
      * ``"st:<model>"`` — :class:`SentenceTransformerEncoder`,
        e.g. ``"st:all-MiniLM-L6-v2"``.
    """
    if spec == "hash":
        return HashTextEncoder()
    if spec.startswith("hash:"):
        return HashTextEncoder(dim=int(spec.split(":", 1)[1]))
    if spec.startswith("st:"):
        return SentenceTransformerEncoder(model_name=spec.split(":", 1)[1])
    raise ValueError(
        f"unknown encoder spec {spec!r}; use 'hash', 'hash:<dim>', or 'st:<model>'"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(s: object) -> float | None:
    if s is None or s == "":
        return None
    try:
        v = float(s)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _format_depth(d_min: float, d_max: float) -> str:
    if not math.isfinite(d_min) or not math.isfinite(d_max):
        return "Surface measurements."
    if d_max <= d_min:
        return f"Sensor at {d_min:g} cm depth."
    return f"Sensors spanning {d_min:g}-{d_max:g} cm depth."


__all__ = [
    "TextEncoder",
    "HashTextEncoder",
    "SentenceTransformerEncoder",
    "Snippet",
    "AOIBinding",
    "ismn_station_snippets",
    "lucas_point_snippets",
    "build_text_corpus",
    "save_text_corpus",
    "encoder_from_spec",
]
