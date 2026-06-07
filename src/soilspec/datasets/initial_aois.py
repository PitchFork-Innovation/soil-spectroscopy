"""Registry of AOIs + time windows for the first joint training cut.

Each entry is an :class:`AOIWindow` carrying a 0.05° × 0.05° box (~5 km
at mid-latitudes) centred on a *real* GT measurement location pulled
from the converted ISMN / LUCAS CSVs. Coordinates are no longer
hand-picked — they live in ``data/registry/initial_aois.json``, which
is produced by :mod:`soilspec.datasets.regenerate_aois`.

This file used to carry tuples like ``("ismn_uscrn_co_boulder", "USCRN",
40.04, -105.54)`` but those guessed coordinates frequently missed the
actual station by tens of km, causing the build driver to silently drop
the AOI for "no GT samples in window". Loading from CSV-derived JSON
guarantees every AOI's bbox contains at least one labelled measurement.

If the JSON is absent, importing this module raises a clear error
telling the caller to run the regenerator. The windows themselves are
still hard-coded here since they're a design choice, not a data-driven
one.

Re-generate after any CSV change::

    python -m soilspec.datasets.regenerate_aois \\
        --ismn-csv data/groundtruth/ismn_2023.csv \\
        --lucas-csv data/groundtruth/lucas_2018_topsoil.csv \\
        --output data/registry/initial_aois.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..types import AOI, BoundingBox, TimeWindow


# 0.05° box ≈ 5 km at mid-latitudes; half-width = 0.025°.
_HALF_DEG = 0.025

# Where the regenerator writes its output. Resolved relative to repo root
# (computed from this file's path), so importing the module works whether
# the caller runs from the repo root or elsewhere.
_REGISTRY_JSON = (
    Path(__file__).resolve().parents[3]
    / "data" / "registry" / "initial_aois.json"
)


def _aoi(aoi_id: str, lat: float, lon: float) -> AOI:
    return AOI(
        aoi_id=aoi_id,
        bbox=BoundingBox(
            min_lon=lon - _HALF_DEG,
            min_lat=lat - _HALF_DEG,
            max_lon=lon + _HALF_DEG,
            max_lat=lat + _HALF_DEG,
        ),
    )


def _epoch(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


# Northern-hemisphere ISMN: Apr-Oct 2023 covers the full growing season +
# shoulder months. The wider window: (a) catches stations whose 2023
# uptime starts mid-summer (e.g. SCAN FortReno#1 begins reporting July 7),
# and (b) gives ~30+ S1 passes / ~40+ S2 passes per AOI, so each AOI
# contributes many daily-paired training rows instead of just 1-2.
ISMN_NH_WINDOW = TimeWindow(start=_epoch(2023, 4, 1), end=_epoch(2023, 10, 31))

# Southern-hemisphere ISMN window (kept for forward compatibility — no
# currently-contributing SH networks landed in our CSV).
ISMN_SH_WINDOW = TimeWindow(start=_epoch(2023, 10, 1), end=_epoch(2023, 11, 30))

# LUCAS 2018 fieldwork ran Apr-Oct 2018. We start the window at Jan 1
# because :func:`groundtruth._year_to_epoch` maps every LUCAS sample's
# ``year=2018`` to Jan 1 2018 UTC and the adapter then runs
# ``window.contains(t)``. Without Jan 1 inside the window every LUCAS row
# is silently dropped. The end stays at Sep 30 to cap S1/S2 fetch volume
# at the cloud-friendlier months. Combined with the per-source 1-year GT
# bucket in build.py, the Jan-1-stamped GT pairs with every 2018-dated
# S1+S2 acquisition in the AOI.
LUCAS_WINDOW = TimeWindow(start=_epoch(2018, 1, 1), end=_epoch(2018, 9, 30))


@dataclass(frozen=True)
class AOIWindow:
    """One AOI + its time window + the GT source it should be joined to."""

    aoi: AOI
    window: TimeWindow
    label_source: str    # "ismn" | "lucas"
    network: str         # e.g. "USCRN", "LUCAS-2018"


class RegistryNotGeneratedError(RuntimeError):
    """Raised when initial_aois.json is missing."""


def _load_registry() -> tuple[list[AOIWindow], list[AOIWindow]]:
    """Read the JSON registry and build ISMN/LUCAS AOIWindow lists.

    Returns ``(ismn_aois, lucas_aois)``.
    """
    if not _REGISTRY_JSON.exists():
        raise RegistryNotGeneratedError(
            f"AOI registry not found at {_REGISTRY_JSON}. Run:\n"
            f"  python -m soilspec.datasets.regenerate_aois "
            f"--ismn-csv data/groundtruth/ismn_2023.csv "
            f"--lucas-csv data/groundtruth/lucas_2018_topsoil.csv"
        )
    with _REGISTRY_JSON.open("r") as fh:
        payload = json.load(fh)

    ismn: list[AOIWindow] = []
    for entry in payload.get("ismn", []):
        hemisphere = entry.get("hemisphere", "NH")
        window = ISMN_SH_WINDOW if hemisphere == "SH" else ISMN_NH_WINDOW
        ismn.append(AOIWindow(
            aoi=_aoi(entry["aoi_id"], float(entry["lat"]), float(entry["lon"])),
            window=window,
            label_source="ismn",
            network=entry["network"],
        ))

    lucas: list[AOIWindow] = []
    for entry in payload.get("lucas", []):
        lucas.append(AOIWindow(
            aoi=_aoi(entry["aoi_id"], float(entry["lat"]), float(entry["lon"])),
            window=LUCAS_WINDOW,
            label_source="lucas",
            network=entry["network"],
        ))
    return ismn, lucas


# Lazy load: the AOI tuples are populated on first access via module
# __getattr__. This keeps importing the module cheap and — more
# importantly — keeps it possible to import the package even before the
# JSON registry has been generated (so ``regenerate_aois`` can run).
_LAZY_NAMES: frozenset[str] = frozenset({
    "ISMN_AOIS", "LUCAS_AOIS", "INITIAL_AOIS",
})
_CACHE: dict[str, tuple[AOIWindow, ...]] = {}


def __getattr__(name: str):
    if name not in _LAZY_NAMES:
        raise AttributeError(name)
    if not _CACHE:
        ismn, lucas = _load_registry()
        _CACHE["ISMN_AOIS"] = tuple(ismn)
        _CACHE["LUCAS_AOIS"] = tuple(lucas)
        _CACHE["INITIAL_AOIS"] = _CACHE["ISMN_AOIS"] + _CACHE["LUCAS_AOIS"]
    return _CACHE[name]


__all__ = [
    "AOIWindow",
    "INITIAL_AOIS",
    "ISMN_AOIS",
    "LUCAS_AOIS",
    "ISMN_NH_WINDOW",
    "ISMN_SH_WINDOW",
    "LUCAS_WINDOW",
    "RegistryNotGeneratedError",
]
