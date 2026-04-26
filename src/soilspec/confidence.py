"""Confidence and explanation module.

Computes per-output metadata: temporal consistency, data completeness, model
agreement (across ensemble members), and degradation flags from missing
modalities. Attached to every property estimate, recommendation, and
capability class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .types import (
    AnnotatedOutput, ConfidenceMetadata, FusedRepresentation,
    PreprocessedRecord, SoilFunctionalProperties, TemporalFeatureSet,
)


@dataclass
class ConfidenceModule:
    """Pure functions; no state."""

    def annotate(
        self,
        output: object,
        *,
        properties: SoilFunctionalProperties | None = None,
        fused: FusedRepresentation | None = None,
        record: PreprocessedRecord | None = None,
        features: TemporalFeatureSet | None = None,
        provenance: Mapping[str, str] | None = None,
    ) -> AnnotatedOutput:
        confidence = self.compute(
            properties=properties, fused=fused, record=record, features=features, provenance=provenance,
        )
        return AnnotatedOutput(output=output, confidence=confidence)

    def compute(
        self,
        *,
        properties: SoilFunctionalProperties | None,
        fused: FusedRepresentation | None,
        record: PreprocessedRecord | None,
        features: TemporalFeatureSet | None,
        provenance: Mapping[str, str] | None,
    ) -> ConfidenceMetadata:
        # temporal consistency: 1 - mean volatility, clamped to [0,1]
        if features is not None and features.volatility:
            mean_vol = float(np.mean(list(features.volatility.values())))
            temporal_consistency = float(max(0.0, min(1.0, 1.0 - mean_vol)))
        else:
            temporal_consistency = 0.5
        # data completeness: how many declared steps actually ran
        if record is not None:
            ran = len(record.pathway_descriptor)
            expected_max = 8  # spatial(4) + vector(4) per PRD
            data_completeness = float(min(1.0, ran / expected_max))
        else:
            data_completeness = 1.0
        # model agreement: 1 - mean ensemble spread (uncertainty)
        if properties is not None and properties.uncertainty:
            spread = float(np.mean(list(properties.uncertainty.values())))
            model_agreement = float(max(0.0, min(1.0, 1.0 - spread)))
        else:
            model_agreement = 0.5
        degraded = bool(fused.degraded) if fused is not None else False
        return ConfidenceMetadata(
            temporal_consistency=temporal_consistency,
            data_completeness=data_completeness,
            model_agreement=model_agreement,
            degradation_flag=degraded,
            provenance=dict(provenance or {}),
        )
