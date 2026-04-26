"""Map publisher.

Renders capability classifications, recommendation layers, and confidence
layers as cacheable map tiles, and writes them to the cached map repository
keyed by `(aoi_id, output_type, generation_time)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .storage import StorageTier, StorageTierManager, map_key
from .types import (
    AOI, CapabilityClassification, ConfidenceMetadata, MapHandle,
    RecommendationLayers,
)


CAPABILITY_OUTPUT = "capability"
RECOMMENDATION_OUTPUT = "recommendation"
CONFIDENCE_OUTPUT = "confidence"


@dataclass
class MapPublisher:
    storage: StorageTierManager

    def publish(
        self,
        aoi: AOI,
        generation_time: int,
        capability: Mapping[str, CapabilityClassification],
        recommendations: RecommendationLayers,
        confidence: Mapping[str, ConfidenceMetadata],
    ) -> tuple[MapHandle, MapHandle, MapHandle]:
        cap_handle = self._put(aoi, CAPABILITY_OUTPUT, generation_time, _capability_payload(capability))
        rec_handle = self._put(aoi, RECOMMENDATION_OUTPUT, generation_time, _recommendation_payload(recommendations))
        conf_handle = self._put(aoi, CONFIDENCE_OUTPUT, generation_time, _confidence_payload(confidence))
        return cap_handle, rec_handle, conf_handle

    def _put(self, aoi: AOI, output_type: str, generation_time: int, payload: dict) -> MapHandle:
        key = map_key(aoi.aoi_id, output_type, generation_time)
        self.storage.put(StorageTier.MAP, key, payload)
        return MapHandle(
            aoi_id=aoi.aoi_id,
            output_type=output_type,
            generation_time=generation_time,
            storage_key=key,
        )


def _capability_payload(items: Mapping[str, CapabilityClassification]) -> dict:
    return {
        "type": CAPABILITY_OUTPUT,
        "tiles": {
            tid: {
                "capability_class": c.capability_class,
                "score": c.score,
                "explanation": dict(c.explanation),
            }
            for tid, c in items.items()
        },
    }


def _recommendation_payload(rec: RecommendationLayers) -> dict:
    return {
        "type": RECOMMENDATION_OUTPUT,
        "aoi_id": rec.aoi_id,
        "priority_zones": dict(rec.priority_zones),
        "risk_areas": dict(rec.risk_areas),
        "management_actions": {k: list(v) for k, v in rec.management_actions.items()},
    }


def _confidence_payload(items: Mapping[str, ConfidenceMetadata]) -> dict:
    return {
        "type": CONFIDENCE_OUTPUT,
        "tiles": {
            tid: {
                "temporal_consistency": c.temporal_consistency,
                "data_completeness": c.data_completeness,
                "model_agreement": c.model_agreement,
                "degradation_flag": c.degradation_flag,
                "provenance": dict(c.provenance),
            }
            for tid, c in items.items()
        },
    }
