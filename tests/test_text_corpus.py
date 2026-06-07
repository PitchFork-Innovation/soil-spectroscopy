"""Tests for the text-embedding corpus builder.

Covers: encoder determinism, snippet extraction, AOI alignment,
time-strategy broadcast, and round-trip through the .npz / TextRecord
loading path used by ``python -m soilspec.train --text-npz``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from soilspec.datasets.text_corpus import (
    AOIBinding, HashTextEncoder, build_text_corpus, encoder_from_spec,
    ismn_station_snippets, lucas_point_snippets, save_text_corpus,
)
from soilspec.groundtruth import TextRecord
from soilspec.types import AOI, BoundingBox, TimeWindow


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def test_hash_encoder_is_deterministic_and_unit_norm():
    enc = HashTextEncoder(dim=32)
    a = enc.encode(["hello world", "soil moisture station"])
    b = enc.encode(["hello world", "soil moisture station"])
    assert a.shape == (2, 32)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    # Different inputs → different embeddings.
    assert not np.allclose(a[0], a[1])
    # L2-normalised.
    for row in a:
        assert float(np.linalg.norm(row)) == pytest.approx(1.0, abs=1e-5)


def test_encoder_from_spec_hash_variants():
    assert encoder_from_spec("hash").dim == 64
    assert encoder_from_spec("hash:128").dim == 128
    with pytest.raises(ValueError):
        encoder_from_spec("unknown:foo")


# ---------------------------------------------------------------------------
# Snippet extraction from CSVs
# ---------------------------------------------------------------------------


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path


def test_ismn_station_snippets_dedup_per_station(tmp_path: Path):
    path = _write_csv(
        tmp_path / "ismn.csv",
        ["network", "station", "lon", "lat", "timestamp", "soil_moisture",
         "depth_from", "depth_to", "qc_flag"],
        [
            ["NET", "STA1", "5.0", "45.0", "1700000000", "0.21", "0", "5", "G"],
            ["NET", "STA1", "5.0", "45.0", "1700003600", "0.22", "0", "5", "G"],
            ["NET", "STA2", "5.5", "45.5", "1700000000", "0.30", "0", "5", "G"],
        ],
    )
    snippets = list(ismn_station_snippets(path))
    assert len(snippets) == 2
    by_doc = {s.doc_id: s for s in snippets}
    assert "NET/STA1" in by_doc
    assert "NET/STA2" in by_doc
    assert "depth" in by_doc["NET/STA1"].text.lower()
    assert by_doc["NET/STA1"].source == "ismn"


def test_lucas_point_snippets_include_observations(tmp_path: Path):
    path = _write_csv(
        tmp_path / "lucas.csv",
        ["point_id", "lon", "lat", "year", "soc", "nitrogen", "ph"],
        [
            ["42", "10.0", "50.0", "2018", "12.4", "1.1", "5.5"],
            ["43", "10.5", "50.5", "2018", "", "", "6.0"],
        ],
    )
    snippets = list(lucas_point_snippets(path))
    assert len(snippets) == 2
    by_doc = {s.doc_id: s for s in snippets}
    assert "soil organic carbon" in by_doc["lucas/42"].text
    assert "extractable phosphorus" not in by_doc["lucas/42"].text
    assert "pH in CaCl2" in by_doc["lucas/43"].text


# ---------------------------------------------------------------------------
# Corpus assembly + AOI alignment
# ---------------------------------------------------------------------------


def _binding(aoi_id: str, lon: float, lat: float, half_deg: float = 0.05):
    aoi = AOI(
        aoi_id=aoi_id,
        bbox=BoundingBox(
            min_lon=lon - half_deg, min_lat=lat - half_deg,
            max_lon=lon + half_deg, max_lat=lat + half_deg,
        ),
    )
    return AOIBinding(
        aoi=aoi,
        window=TimeWindow(start=1700000000, end=1700000000 + 86400 * 3),
    )


def test_build_corpus_aligns_snippets_only_to_containing_aoi(tmp_path: Path):
    """A snippet should land only on AOIs whose bbox contains its lon/lat."""
    path = _write_csv(
        tmp_path / "ismn.csv",
        ["network", "station", "lon", "lat", "timestamp", "soil_moisture",
         "depth_from", "depth_to", "qc_flag"],
        [["NET", "S1", "5.0", "45.0", "1700000000", "0.2", "0", "5", "G"]],
    )
    snippets = list(ismn_station_snippets(path))
    aois = [
        _binding("inside", 5.0, 45.0),
        _binding("outside", 20.0, 30.0),
    ]
    records, encoder = build_text_corpus(
        snippets=snippets, aois=aois,
        encoder=HashTextEncoder(dim=8),
        tile_size=32, target_shape=(32, 32),
        time_strategy="window_start",
    )
    assert encoder == "hash64"
    assert len(records) == 1
    assert records[0].tile_id.startswith("inside/")
    assert records[0].embedding.shape == (8,)
    assert records[0].doc_id == "NET/S1"


def test_time_strategy_daily_broadcasts(tmp_path: Path):
    path = _write_csv(
        tmp_path / "ismn.csv",
        ["network", "station", "lon", "lat", "timestamp", "soil_moisture",
         "depth_from", "depth_to", "qc_flag"],
        [["NET", "S1", "5.0", "45.0", "1700000000", "0.2", "0", "5", "G"]],
    )
    snippets = list(ismn_station_snippets(path))
    aois = [_binding("a", 5.0, 45.0)]
    rec_start, _ = build_text_corpus(
        snippets=snippets, aois=aois, encoder=HashTextEncoder(dim=8),
        tile_size=32, target_shape=(32, 32),
        time_strategy="window_start",
    )
    rec_daily, _ = build_text_corpus(
        snippets=snippets, aois=aois, encoder=HashTextEncoder(dim=8),
        tile_size=32, target_shape=(32, 32),
        time_strategy="daily",
    )
    assert len(rec_start) == 1
    # 3-day window → 4 daily buckets (start through end inclusive).
    assert len(rec_daily) == 4
    # All records share embedding (the snippet is static).
    for r in rec_daily[1:]:
        np.testing.assert_array_equal(r.embedding, rec_daily[0].embedding)


# ---------------------------------------------------------------------------
# .npz round-trip
# ---------------------------------------------------------------------------


def test_save_and_reload_via_train_loader(tmp_path: Path):
    """Records → .npz → trainer's loader returns equivalent TextRecords."""
    import argparse
    from soilspec.train import _load_text_records

    records = [
        TextRecord(
            tile_id="aoi1/r000c000", time=1700000000,
            embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            doc_id="NET/S1", encoder="hash64",
        ),
        TextRecord(
            tile_id="aoi2/r000c000", time=1700086400,
            embedding=np.array([0.4, 0.5, 0.6], dtype=np.float32),
            doc_id="lucas/42", encoder="hash64",
        ),
    ]
    out = save_text_corpus(records, "hash64", tmp_path / "corpus.npz")
    assert out.exists()

    args = argparse.Namespace(text_npz=out, text_encoder=None)
    loaded, encoder = _load_text_records(args)
    assert encoder == "hash64"
    assert len(loaded) == 2
    assert loaded[0].tile_id == "aoi1/r000c000"
    assert loaded[0].doc_id == "NET/S1"
    np.testing.assert_allclose(loaded[1].embedding, [0.4, 0.5, 0.6])


def test_save_empty_corpus(tmp_path: Path):
    """Empty corpus produces a valid .npz that loads to an empty list."""
    import argparse
    from soilspec.train import _load_text_records

    out = save_text_corpus([], "hash64", tmp_path / "empty.npz")
    args = argparse.Namespace(text_npz=out, text_encoder=None)
    loaded, encoder = _load_text_records(args)
    assert encoder == "hash64"
    assert loaded == []
