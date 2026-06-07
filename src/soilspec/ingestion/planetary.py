"""Microsoft Planetary Computer STAC adapters for Sentinel-1 and Sentinel-2.

Real source adapters that issue STAC queries against
``https://planetarycomputer.microsoft.com/api/stac/v1`` and read windowed
COG slices for the requested AOI / time window. Output payload shape
matches the synthetic adapters so downstream preprocessing is unaffected:

- Sentinel-2: ``payload = {"reflectance": (6, H, W) float32 in [0, 1],
  "qa": (H, W) uint8 cloud mask}`` with bands ``B02, B03, B04, B08, B11, B12``.
- Sentinel-1: ``payload = (2, H, W) float32`` with bands ``VV, VH``. Values
  are ``sigma0`` in dB when reading the RTC collection, raw DN otherwise.

Optional dependency: install with ``pip install soilspec[planetary]``. The
``planetary_computer`` library auto-signs asset URLs with a short-lived SAS
token; both ``sentinel-2-l2a`` and ``sentinel-1-rtc`` on PC are
anonymous-readable today, so no subscription key is required for normal
use. (``sentinel-1-grd`` exists too but is stored in raw radar geometry
without a projected CRS, which prevents bbox-windowed reads — RTC is the
only useful S1 collection for this pipeline.)

CRS caveat: the read is windowed but not reprojected — pixels stay in the
source COG's native CRS (typically a UTM zone). Pipeline encoders are
shape-agnostic so this is fine for AOIs within a single zone; AOIs that
straddle multiple zones should be tiled before fetching.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from datetime import datetime, timezone
from typing import Callable, Iterable, TypeVar

import numpy as np

from ..types import AOI, AssetMetadata, SENTINEL1, SENTINEL2, TimeWindow
from .adapters import RawAsset, UnreachableSourceError


log = logging.getLogger(__name__)


PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B08", "B11", "B12")
S2_QA_ASSET = "SCL"  # 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=thin cirrus
S2_CLOUD_CLASSES = (3, 8, 9, 10)
S2_REFLECTANCE_SCALE = 1.0 / 10000.0


# Retry knobs. PC blob storage routinely returns 503 under load; STAC search
# occasionally 502s. Four attempts with 1.5s base + exponential backoff caps
# total wait at ~22s per operation, which is well below typical timeouts but
# enough to ride out the multi-second outages we observed in the first run.
_MAX_ATTEMPTS = 4
_BASE_DELAY_S = 1.5
_RETRIABLE_HTTP_CODES = ("408", "429", "500", "502", "503", "504")
_RETRIABLE_KEYWORDS = (
    "timeout", "timed out", "connection reset", "connection aborted",
    "temporary failure", "remote end closed", "ssl", "read error",
    "broken pipe", "incomplete read", "gateway", "max retries",
)

_T = TypeVar("_T")


def _is_retriable(exc: BaseException) -> bool:
    """Heuristic for distinguishing transient cloud failures from real ones.

    Anything matching an HTTP 5xx / 429 status code in its message, or any
    classic connection-level error keyword, is treated as retriable. Auth
    failures (401/403) and missing assets (404) are explicitly NOT retried.
    """
    msg = str(exc).lower()
    if any(f" {code}" in msg or f"({code}" in msg or f"[{code}" in msg
           for code in _RETRIABLE_HTTP_CODES):
        return True
    # rasterio bubbles up CURL errors like "CURL error: ..." for transport
    # failures — treat all CURL errors as retriable.
    if "curl error" in msg or "curl_easy_perform" in msg:
        return True
    return any(k in msg for k in _RETRIABLE_KEYWORDS)


def _retry(fn: Callable[[], _T], *, what: str,
           attempts: int = _MAX_ATTEMPTS) -> _T:
    """Run ``fn`` with exponential backoff + jitter on transient failures."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_attempt = i == attempts - 1
            if last_attempt or not _is_retriable(e):
                raise
            delay = _BASE_DELAY_S * (2 ** i) + random.random() * 0.5
            log.warning(
                "%s: transient failure (attempt %d/%d): %s; retrying in %.1fs",
                what, i + 1, attempts, e, delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def _stable_id(*parts) -> str:
    return hashlib.sha256("/".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _epoch(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _open_pc_client():
    """Lazy import + STAC client opener; wraps deps in UnreachableSourceError."""
    try:
        from pystac_client import Client
        import planetary_computer
    except ImportError as e:
        raise UnreachableSourceError(
            "Planetary Computer adapters require the [planetary] extra: "
            "pip install soilspec[planetary]"
        ) from e
    try:
        return _retry(
            lambda: Client.open(
                PC_STAC_URL, modifier=planetary_computer.sign_inplace,
            ),
            what="pc_client.open",
        )
    except Exception as e:
        raise UnreachableSourceError(f"failed to reach {PC_STAC_URL}: {e}") from e


def _read_windowed(href: str, bbox, target_shape: tuple[int, int]) -> np.ndarray:
    """Read a windowed slice of a remote COG, resampled to ``target_shape``.

    Wrapped in :func:`_retry` because PC blob storage occasionally returns
    503s under load — a single transient read failure should not abort the
    whole AOI's worth of acquisitions.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    def _do_read() -> np.ndarray:
        with rasterio.open(href) as ds:
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", ds.crs,
                bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat,
                densify_pts=21,
            )
            window = from_bounds(left, bottom, right, top, ds.transform)
            return ds.read(
                1, window=window, out_shape=target_shape,
                resampling=Resampling.bilinear, boundless=True, fill_value=0,
            ).astype(np.float32)

    return _retry(_do_read, what=f"cog_read({href.rsplit('/', 1)[-1][:32]})")


def _datetime_range(window: TimeWindow) -> str:
    start = datetime.fromtimestamp(window.start, tz=timezone.utc).isoformat()
    end = datetime.fromtimestamp(window.end, tz=timezone.utc).isoformat()
    return f"{start}/{end}"


# ---------------------------------------------------------------------------
# Sentinel-2 L2A
# ---------------------------------------------------------------------------


class PlanetaryComputerSentinel2Adapter:
    """Sentinel-2 L2A surface reflectance from Microsoft Planetary Computer."""

    provider = "planetary-s2"
    modality = SENTINEL2
    BANDS: tuple[str, ...] = S2_BANDS + ("QA",)

    def __init__(
        self,
        tile_size: int = 32,
        max_cloud_cover: float = 30.0,
        collection: str = "sentinel-2-l2a",
    ) -> None:
        self.tile_size = int(tile_size)
        self.max_cloud_cover = float(max_cloud_cover)
        self.collection = collection

    def fetch(self, aoi: AOI, window: TimeWindow) -> Iterable[RawAsset]:
        client = _open_pc_client()
        try:
            def _search() -> list:
                search = client.search(
                    collections=[self.collection],
                    bbox=[aoi.bbox.min_lon, aoi.bbox.min_lat,
                          aoi.bbox.max_lon, aoi.bbox.max_lat],
                    datetime=_datetime_range(window),
                    query={"eo:cloud_cover": {"lt": self.max_cloud_cover}},
                )
                return list(search.items())
            items = _retry(_search, what=f"s2_stac_search({aoi.aoi_id})")
        except UnreachableSourceError:
            raise
        except Exception as e:
            raise UnreachableSourceError(f"S2 STAC query failed: {e}") from e

        for item in items:
            t = _epoch(item.datetime)
            target = (self.tile_size, self.tile_size)
            # Per-item: a failed read after retries means *this* COG is broken
            # (or PC storage is down for this specific blob). Skip just this
            # item — the AOI's other 30+ acquisitions are still usable.
            try:
                bands = np.stack(
                    [
                        _read_windowed(item.assets[b].href, aoi.bbox, target)
                        * S2_REFLECTANCE_SCALE
                        for b in S2_BANDS
                    ],
                    axis=0,
                ).astype(np.float32)
                scl = _read_windowed(item.assets[S2_QA_ASSET].href, aoi.bbox, target)
                qa = np.isin(scl.astype(np.int16), S2_CLOUD_CLASSES).astype(np.uint8)
            except Exception as e:
                log.warning(
                    "S2 read failed for %s after retries; skipping item: %s",
                    item.id, e,
                )
                continue

            obs_id = _stable_id(aoi.aoi_id, t, self.provider)
            payload = {"reflectance": bands, "qa": qa}
            meta = AssetMetadata(
                observation_id=obs_id,
                request_id=_stable_id("req", aoi.aoi_id, window.start, window.end),
                provider=self.provider,
                modality=self.modality,
                timestamp=t,
                bbox=aoi.bbox,
                bands=self.BANDS,
                missing_entries=(),
                extra={
                    "stac_id": item.id,
                    "cloud_cover": float(item.properties.get("eo:cloud_cover", 0.0)),
                    "reflectance_geometry": "BOA",
                },
            )
            yield RawAsset(obs_id, self.provider, self.modality, payload, meta)


# ---------------------------------------------------------------------------
# Sentinel-1
# ---------------------------------------------------------------------------


class PlanetaryComputerSentinel1Adapter:
    """Sentinel-1 SAR backscatter from Microsoft Planetary Computer.

    Defaults to ``sentinel-1-rtc`` (terrain-corrected sigma0 in dB).
    ``sentinel-1-grd`` is also accepted but stored in raw radar geometry
    without a projected CRS, so bbox-windowed reads will fail — only use it
    if you intend to handle the raw-geometry case yourself.
    """

    provider = "planetary-s1"
    modality = SENTINEL1

    def __init__(
        self,
        tile_size: int = 32,
        collection: str = "sentinel-1-rtc",
    ) -> None:
        self.tile_size = int(tile_size)
        self.collection = collection

    def fetch(self, aoi: AOI, window: TimeWindow) -> Iterable[RawAsset]:
        client = _open_pc_client()
        try:
            def _search() -> list:
                search = client.search(
                    collections=[self.collection],
                    bbox=[aoi.bbox.min_lon, aoi.bbox.min_lat,
                          aoi.bbox.max_lon, aoi.bbox.max_lat],
                    datetime=_datetime_range(window),
                )
                return list(search.items())
            items = _retry(_search, what=f"s1_stac_search({aoi.aoi_id})")
        except UnreachableSourceError:
            raise
        except Exception as e:
            raise UnreachableSourceError(f"S1 STAC query failed: {e}") from e

        is_rtc = "rtc" in self.collection
        for item in items:
            t = _epoch(item.datetime)
            target = (self.tile_size, self.tile_size)
            # Per-item: skip on read failure rather than aborting the AOI.
            try:
                vv = _read_windowed(item.assets["vv"].href, aoi.bbox, target)
                vh = _read_windowed(item.assets["vh"].href, aoi.bbox, target)
            except Exception as e:
                log.warning(
                    "S1 read failed for %s after retries; skipping item: %s",
                    item.id, e,
                )
                continue

            if is_rtc:
                # RTC values are linear sigma0 — convert to dB to match the
                # synthetic adapter's value range. Clip avoids log(0).
                vv = 10.0 * np.log10(np.clip(vv, 1e-6, None))
                vh = 10.0 * np.log10(np.clip(vh, 1e-6, None))

            payload = np.stack([vv, vh], axis=0).astype(np.float32)
            obs_id = _stable_id(aoi.aoi_id, t, self.provider)
            meta = AssetMetadata(
                observation_id=obs_id,
                request_id=_stable_id("req", aoi.aoi_id, window.start, window.end),
                provider=self.provider,
                modality=self.modality,
                timestamp=t,
                bbox=aoi.bbox,
                bands=("VV", "VH"),
                missing_entries=(),
                extra={
                    "stac_id": item.id,
                    "collection": self.collection,
                    "calibration": "sigma0_db" if is_rtc else "raw_dn",
                },
            )
            yield RawAsset(obs_id, self.provider, self.modality, payload, meta)


__all__ = [
    "PlanetaryComputerSentinel1Adapter",
    "PlanetaryComputerSentinel2Adapter",
    "PC_STAC_URL",
]
