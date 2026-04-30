"""Common dataclasses shared across pipeline stages.

These types are part of the public contract: stage outputs are typed by these
schemas. Internal implementations (encoder backend, fusion strategy, ensemble
member) may change freely, but the schemas here are stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Geographic primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned geographic bounding box in EPSG:4326 (lon, lat)."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def __post_init__(self) -> None:
        if self.min_lon >= self.max_lon or self.min_lat >= self.max_lat:
            raise ValueError(f"degenerate bbox: {self}")

    def contains(self, lon: float, lat: float) -> bool:
        return self.min_lon <= lon <= self.max_lon and self.min_lat <= lat <= self.max_lat


@dataclass(frozen=True)
class AOI:
    """Area of interest. `aoi_id` is stable; `bbox` is the analyzed extent."""

    aoi_id: str
    bbox: BoundingBox


@dataclass(frozen=True)
class TimeWindow:
    """Inclusive [start, end] integer-epoch-second window."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("TimeWindow.start must be <= end")

    def contains(self, t: int) -> bool:
        return self.start <= t <= self.end


# ---------------------------------------------------------------------------
# Modality identifiers
# ---------------------------------------------------------------------------


SENTINEL1 = "s1"
SENTINEL2 = "s2"
VECTOR = "vector"  # topographic / land-cover / soil-grid layers
INSITU = "insitu"  # optional ground truth
HYPERSPECTRAL = "hyperspectral"  # listed in PRD; not in MVP

ALL_MODALITIES: tuple[str, ...] = (SENTINEL1, SENTINEL2, VECTOR, INSITU, HYPERSPECTRAL)


# ---------------------------------------------------------------------------
# Asset / metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssetMetadata:
    """Normalized metadata extracted from a raw observation.

    The MCP adapter and metadata parser produce records of this shape from
    arbitrary provider-specific inputs.
    """

    observation_id: str
    request_id: str
    provider: str
    modality: str
    timestamp: int
    bbox: BoundingBox
    bands: tuple[str, ...] = ()
    missing_entries: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawObservationHandle:
    """Stable pointer into the raw store + the parsed metadata sidecar."""

    observation_id: str
    provider: str
    modality: str
    storage_key: str
    metadata: AssetMetadata


# ---------------------------------------------------------------------------
# Preprocessed records and embeddings
# ---------------------------------------------------------------------------


@dataclass
class PreprocessedRecord:
    """Co-aligned multimodal record after the preprocessing pathways converge."""

    aoi_id: str
    tile_id: str
    time: int
    crs: str
    bounds: BoundingBox
    spatial: dict[str, np.ndarray] = field(default_factory=dict)
    """modality (s1/s2/...) -> raster array (H, W) or (B, H, W)."""
    vector: dict[str, np.ndarray] = field(default_factory=dict)
    """attribute name -> 1-D feature vector aligned to this tile."""
    pathway_descriptor: tuple[str, ...] = ()
    """ordered names of the preprocessing steps that ran (for audit/confidence)."""

    @property
    def key(self) -> tuple[str, int]:
        return (self.tile_id, self.time)


@dataclass(frozen=True)
class SpectralEmbedding:
    tile_id: str
    time: int
    vector: np.ndarray  # shape (latent_dim,)
    backend: str
    valid_bands: int


@dataclass(frozen=True)
class SpatialEmbedding:
    tile_id: str
    time: int
    vector: np.ndarray  # shape (latent_dim,)
    backend: str
    patch_size: int


@dataclass(frozen=True)
class FusedRepresentation:
    tile_id: str
    time: int
    vector: np.ndarray  # shape (fused_dim,)
    channels: dict[str, slice]
    """capability channel name -> slice into `vector`."""
    strategy: str
    degraded: bool
    missing_modalities: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Inference outputs
# ---------------------------------------------------------------------------


SOIL_PROPERTY_NAMES: tuple[str, ...] = (
    "smi",  # Soil Moisture Index
    "infiltration_potential",
    "erosion_susceptibility",
)

# Directly measurable soil properties — populated from real ground-truth
# sources (ISMN, LUCAS, SMAP, SoilGrids). The functional properties above are
# *derived* from these by the capability layer, so they live in a separate
# vocabulary: training labels are measured, downstream consumers are functional.
MEASURED_PROPERTY_NAMES: tuple[str, ...] = (
    "soil_moisture",   # volumetric water content, m3/m3 (ISMN, SMAP)
    "soc",             # soil organic carbon, g/kg (LUCAS, SoilGrids, WORLDSOILS)
    "nitrogen",        # total N, g/kg (LUCAS)
    "phosphorus",      # extractable P, mg/kg (LUCAS)
    "potassium",       # extractable K, mg/kg (LUCAS)
    "ph",              # pH in CaCl2 (LUCAS)
    "clay_pct",        # %, 0-100 (SoilGrids, LUCAS)
    "sand_pct",        # %, 0-100 (SoilGrids, LUCAS)
    "bulk_density",    # g/cm3 (SoilGrids)
)


@dataclass(frozen=True)
class SoilFunctionalProperties:
    """Per-cell soil functional properties with calibrated uncertainty."""

    tile_id: str
    time: int
    properties: dict[str, float]
    uncertainty: dict[str, float]
    member_outputs: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruthSample:
    """An aggregated ground-truth observation tied to a single tile + time bucket.

    Multiple raw measurements (e.g. several ISMN sensor stations falling in
    the same tile within a time bucket) are aggregated into one sample with
    per-property mean and uncertainty (std-error of the mean). `n_observations`
    records how many raw measurements went in, so downstream code can weight
    samples or filter sparse tiles.
    """

    tile_id: str
    time: int
    properties: dict[str, float]
    uncertainty: dict[str, float]
    n_observations: int
    source: str  # provider id: "ismn", "lucas", "smap", "soilgrids", ...


@dataclass(frozen=True)
class TemporalFeatureSet:
    """Higher-order temporal descriptors extracted from a time series."""

    tile_id: str
    trend: dict[str, float]
    rate_of_change: dict[str, float]
    persistence: dict[str, float]
    volatility: dict[str, float]
    recovery: dict[str, float]
    baseline_deviation: dict[str, float]
    n_samples: int


@dataclass(frozen=True)
class TemporalSignals:
    """Output of the expert temporal analysis stage."""

    tile_id: str
    trend_label: str  # e.g., "improving", "stable", "stressed", "degrading"
    anomaly_score: float
    behavior_class: str
    expert_outputs: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Recommendations and capability classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecommendationLayers:
    aoi_id: str
    priority_zones: dict[str, str]  # tile_id -> priority class
    risk_areas: dict[str, str]       # tile_id -> risk label
    management_actions: dict[str, list[str]]


CAPABILITY_CLASSES: tuple[str, ...] = (
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII",
)


@dataclass(frozen=True)
class CharacteristicScores:
    tile_id: str
    scores: dict[str, float]


@dataclass(frozen=True)
class CapabilityClassification:
    tile_id: str
    capability_class: str
    score: float
    explanation: dict[str, Any]


# ---------------------------------------------------------------------------
# Confidence and provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceMetadata:
    temporal_consistency: float
    data_completeness: float
    model_agreement: float
    degradation_flag: bool
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AnnotatedOutput:
    """Wrapper attaching confidence metadata to any pipeline output."""

    output: Any
    confidence: ConfidenceMetadata


# ---------------------------------------------------------------------------
# Map publishing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MapHandle:
    aoi_id: str
    output_type: str
    generation_time: int
    storage_key: str


@dataclass(frozen=True)
class AnalysisRequest:
    aoi: AOI
    time_window: TimeWindow
    modalities: tuple[str, ...] = (SENTINEL1, SENTINEL2, VECTOR)
    insitu_overrides: tuple[Any, ...] = ()
    output_selection: tuple[str, ...] = ("capability", "recommendation", "properties")


@dataclass(frozen=True)
class JobHandle:
    job_id: str
    request: AnalysisRequest


@dataclass(frozen=True)
class RunResult:
    job_id: str
    aoi_id: str
    map_handles: tuple[MapHandle, ...]
    generated_at: int


# Convenience: a flat enumeration of preprocessing step names so descriptors
# are checked against a known vocabulary.
SPATIAL_STEPS: tuple[str, ...] = (
    "cloud_shadow_mask",
    "radar_calibration",
    "resolution_harmonization",
    "tile_extraction",
)
VECTOR_STEPS: tuple[str, ...] = (
    "imputation",
    "normalization",
    "attribute_filter",
    "geospatial_alignment",
)


def assert_finite(arr: np.ndarray, name: str = "array") -> None:
    """Helper used widely to enforce the `no NaNs propagate` invariant."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"non-finite values in {name}")


def stable_iter(items: Iterable[Any]) -> Sequence[Any]:
    """Materialize an iterable into a deterministic, ordered sequence."""
    return tuple(items)
