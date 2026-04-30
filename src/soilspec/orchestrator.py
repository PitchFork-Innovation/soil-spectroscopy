"""Pipeline orchestrator + scheduled update loop.

Drives end-to-end runs as a stage DAG:
  ingest -> preprocess -> encode -> fuse -> append-temporal -> sufficiency-check
         -> infer -> analyze -> recommend -> classify -> publish

Stage outputs are keyed by storage-tier handles so each stage is independently
retryable and idempotent. The scheduled loop fires `tick()` on a configurable
cadence; insufficient temporal data simply waits for the next interval.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .capability import CapabilityScoringEngine, MeasuredToFunctional, RulesEngine
from .confidence import ConfidenceModule
from .encoders import (
    SpatialEncoderRegistry, SpectralEncoderRegistry,
)
from .encoders.spatial import SpatialEncoder
from .encoders.spectral import SpectralEncoder
from .fusion import FusionConfig, FusionEngine
from .ingestion import (
    AdapterRegistry, Ingestion, MetadataParser, MCPAdapter,
)
from .ingestion.adapters import SourceAdapter
from .inference import EnsembleInferenceEngine, InferenceConfig
from .preprocessing import Preprocessor
from .preprocessing.pipeline import PreprocessConfig
from .publisher import MapPublisher
from .recommendation import RecommendationLogicEngine
from .storage import StorageTierManager
from .storage.tiers import StorageTier
from .temporal import (
    SufficiencyCriteria, TemporalAnalysisModule, TemporalDataset,
    TemporalFeatureExtractor,
)
from .types import (
    SENTINEL1, SENTINEL2, VECTOR, AnalysisRequest, AOI, BoundingBox, JobHandle,
    PreprocessedRecord, RunResult, SoilFunctionalProperties, TimeWindow,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    spectral_backend: str = "1d_cnn"
    spatial_backend: str = "cnn"
    spectral_latent_dim: int = 32
    spatial_latent_dim: int = 32
    fusion: FusionConfig = field(default_factory=FusionConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    sufficiency: SufficiencyCriteria = field(default_factory=lambda: SufficiencyCriteria(min_samples=2))
    seed: int = 0
    # Optional: load a trained pipeline from StorageTier.MODEL on construction
    # and use it instead of the random-init EnsembleInferenceEngine. None
    # preserves today's behavior so existing tests keep passing.
    model_key: tuple[str, str] | None = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    def __init__(
        self,
        storage: StorageTierManager | None = None,
        adapters: dict[str, SourceAdapter] | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.storage = storage or StorageTierManager()
        if adapters is None:
            adapters = {
                m: AdapterRegistry.create(m)
                for m in (SENTINEL1, SENTINEL2, VECTOR)
            }
        self.adapters = adapters
        self.parser = MetadataParser()
        self.mcp = MCPAdapter(adapters=adapters, metadata_parser=self.parser)
        self.ingestion = Ingestion(self.storage, adapters, self.parser)
        self.preprocessor = Preprocessor(self.config.preprocess)
        self.spectral_encoder: SpectralEncoder = SpectralEncoderRegistry.create(
            self.config.spectral_backend,
            latent_dim=self.config.spectral_latent_dim,
            seed=self.config.seed,
        )
        self.spatial_encoder: SpatialEncoder = SpatialEncoderRegistry.create(
            self.config.spatial_backend,
            latent_dim=self.config.spatial_latent_dim,
            patch_size=8, stride=4, context_patches=0, seed=self.config.seed,
        )
        self.fusion = FusionEngine(self.config.fusion)
        self.temporal = TemporalDataset(self.storage)
        self.feature_extractor = TemporalFeatureExtractor(min_samples=self.config.sufficiency.min_samples)
        self.analysis = TemporalAnalysisModule()
        self.inferer = EnsembleInferenceEngine(
            fused_dim=self.config.fusion.output_dim,
            channels={k: v for k, v in _channel_slices(self.config.fusion).items()},
            config=self.config.inference,
        )
        self.recommender = RecommendationLogicEngine(strategy="rules")
        self.scoring = CapabilityScoringEngine()
        self.rules = RulesEngine()
        self.publisher = MapPublisher(storage=self.storage)
        self.confidence = ConfidenceModule()

        # Optional trained pipeline. When loaded, it replaces the random-init
        # EnsembleInferenceEngine call in `_publish_for`; functional
        # properties are derived from measured ones via MeasuredToFunctional.
        self.trained_pipeline = None
        self._measured_to_functional = MeasuredToFunctional()
        if self.config.model_key is not None:
            from .training import load_pipeline  # local: avoid import cycle in linters
            family, version = self.config.model_key
            self.trained_pipeline = load_pipeline(self.storage, family, version)

    # -------------------------- single-shot run ---------------------------

    def run_request(self, request: AnalysisRequest) -> RunResult:
        # 1) ingest
        handles = self.ingestion.fetch(request.aoi, request.time_window, request.modalities)
        # 2) preprocess
        records = self.preprocessor.preprocess(handles, self.storage)
        # 3) encode + 4) fuse + 5) append temporal
        for record in records:
            fused = self._encode_and_fuse(record)
            self.temporal.append(record.tile_id, record.time, fused.vector)
        # 6) sufficiency-check + 7-10) downstream stages
        return self._publish_for(request, generation_time=int(_time.time()))

    # ---------------------------- scheduled loop --------------------------

    def tick(self, request: AnalysisRequest, now: int | None = None) -> RunResult | None:
        """One scheduled iteration. Returns None if temporal data is insufficient."""
        now = int(now if now is not None else _time.time())
        handles = self.ingestion.fetch(request.aoi, request.time_window, request.modalities)
        records = self.preprocessor.preprocess(handles, self.storage)
        for record in records:
            fused = self._encode_and_fuse(record)
            self.temporal.append(record.tile_id, record.time, fused.vector)
        # gate
        cells_with_enough = [c for c in self.temporal.cells() if self.temporal.sufficient(c, self.config.sufficiency)]
        if not cells_with_enough:
            return None
        return self._publish_for(request, generation_time=now)

    # ------------------------- internal helpers ---------------------------

    def _encode_and_fuse(self, record: PreprocessedRecord):
        spectral_emb = None
        spatial_emb = None
        if SENTINEL2 in record.spatial:
            spectral_emb = self.spectral_encoder.encode(
                record.tile_id, record.time, record.spatial[SENTINEL2]
            )
        if SENTINEL1 in record.spatial:
            sar = record.spatial[SENTINEL1]
            spatial_emb = self.spatial_encoder.encode(record.tile_id, record.time, sar)
        elif SENTINEL2 in record.spatial:
            # fallback: derive a spatial embedding from S2 too if S1 absent
            spatial_emb = self.spatial_encoder.encode(record.tile_id, record.time, record.spatial[SENTINEL2])
        return self.fusion.fuse(spectral_emb, spatial_emb)

    def _publish_for(self, request: AnalysisRequest, generation_time: int) -> RunResult:
        capability_outputs = {}
        confidence_outputs = {}
        properties_by_tile = {}
        signals_by_tile = {}
        for cell_id in self.temporal.cells():
            series = self.temporal.series(cell_id)
            if not self.config.sufficiency.evaluate(series):
                continue
            features = self.feature_extractor.extract(series)
            # current "fused" representation = last vector of the series
            from .types import FusedRepresentation  # local to avoid cycles in linters
            fused = FusedRepresentation(
                tile_id=cell_id, time=series.times[-1], vector=series.vectors[-1],
                channels=_channel_slices(self.config.fusion), strategy=self.config.fusion.strategy,
                degraded=False,
            )
            properties = self._infer_properties(request.aoi.aoi_id, cell_id, series.times[-1], fused)
            signals = self.analysis.analyze(features, properties)
            scores = self.scoring.score(features, properties, signals)
            classification = self.rules.classify(scores)
            capability_outputs[cell_id] = classification
            properties_by_tile[cell_id] = properties
            signals_by_tile[cell_id] = signals
            annotated = self.confidence.compute(
                properties=properties, fused=fused, record=None, features=features,
                provenance={"rules_version": self.rules.version, "fusion_strategy": self.config.fusion.strategy},
            )
            confidence_outputs[cell_id] = annotated
        recommendations = self.recommender.recommend(request.aoi, signals_by_tile, properties_by_tile)
        cap, rec, conf = self.publisher.publish(
            request.aoi, generation_time, capability_outputs, recommendations, confidence_outputs,
        )
        return RunResult(
            job_id=f"job-{request.aoi.aoi_id}-{generation_time}",
            aoi_id=request.aoi.aoi_id,
            map_handles=(cap, rec, conf),
            generated_at=generation_time,
        )


    def _infer_properties(
        self, aoi_id: str, cell_id: str, time: int, fused,
    ) -> SoilFunctionalProperties:
        """Predict per-tile soil functional properties.

        If a trained pipeline is loaded, run it on the latest preprocessed
        rasters for this tile and derive functional properties from the
        measured-property predictions. Otherwise fall back to the existing
        random-weight EnsembleInferenceEngine for backward compatibility.
        """
        if self.trained_pipeline is None:
            return self.inferer.infer(fused)

        from .storage.tiers import preprocessed_key

        tp = self.trained_pipeline
        # Pull the rasters for this (tile, time). If either is missing we
        # cannot run the trained pipeline — fall back.
        try:
            s1 = self.storage.get(
                StorageTier.PREPROCESSED,
                preprocessed_key(aoi_id, cell_id, time, SENTINEL1),
            )
        except KeyError:
            return self.inferer.infer(fused)
        try:
            s2 = self.storage.get(
                StorageTier.PREPROCESSED,
                preprocessed_key(aoi_id, cell_id, time, SENTINEL2),
            )
        except KeyError:
            return self.inferer.infer(fused)
        # Optional vector covariates for this tile.
        try:
            vector_attrs = self.storage.get(
                StorageTier.PREPROCESSED,
                preprocessed_key(aoi_id, cell_id, time, "vector"),
            )
        except KeyError:
            vector_attrs = {}
        if tp.vector_attr_names:
            vec_features = np.array([
                float(np.nanmean(vector_attrs[k]))
                if k in vector_attrs and np.size(vector_attrs[k])
                else 0.0
                for k in tp.vector_attr_names
            ], dtype=np.float32)
        else:
            vec_features = np.zeros(0, dtype=np.float32)
        measured = tp.predict(s1, s2, vec_features)
        return self._measured_to_functional.derive(
            cell_id, time, measured,
            member_outputs={"trained_pipeline": measured},
        )


def _channel_slices(cfg: FusionConfig) -> dict[str, slice]:
    out: dict[str, slice] = {}
    cur = 0
    for name, dim in cfg.channels:
        out[name] = slice(cur, cur + dim)
        cur += dim
    return out


# ---------------------------------------------------------------------------
# Scheduled trigger
# ---------------------------------------------------------------------------


@dataclass
class ScheduledTrigger:
    """Fake-clock-friendly scheduler."""

    interval_seconds: int
    clock: Callable[[], int] = _time.time  # type: ignore[assignment]
    last_fired: int | None = None

    def due(self, now: int | None = None) -> bool:
        now = int(now if now is not None else self.clock())
        if self.last_fired is None:
            return True
        return now - self.last_fired >= self.interval_seconds

    def fire(self, now: int | None = None) -> None:
        self.last_fired = int(now if now is not None else self.clock())
