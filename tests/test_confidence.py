from soilspec.confidence import ConfidenceModule
from soilspec.types import (
    FusedRepresentation, PreprocessedRecord, SoilFunctionalProperties,
    TemporalFeatureSet, BoundingBox,
)
import numpy as np


def _properties(spread=0.05):
    return SoilFunctionalProperties(
        tile_id="t", time=0,
        properties={"smi": 0.5, "infiltration_potential": 0.5, "erosion_susceptibility": 0.5},
        uncertainty={"smi": spread, "infiltration_potential": spread, "erosion_susceptibility": spread},
    )


def _features(vol=0.1):
    return TemporalFeatureSet(
        tile_id="t",
        trend={"a": 0.1}, rate_of_change={"a": 0.0}, persistence={"a": 0.5},
        volatility={"a": vol}, recovery={"a": 0.5}, baseline_deviation={"a": 0.0},
        n_samples=5,
    )


def _record(steps=("cloud_shadow_mask",)):
    return PreprocessedRecord(
        aoi_id="aoi1", tile_id="t", time=0, crs="EPSG:4326",
        bounds=BoundingBox(0, 0, 1, 1),
        spatial={}, vector={}, pathway_descriptor=tuple(steps),
    )


def _fused(degraded=False):
    return FusedRepresentation(
        tile_id="t", time=0, vector=np.zeros(48, dtype=np.float32),
        channels={}, strategy="concat", degraded=degraded,
    )


def test_compute_metadata_shape():
    cm = ConfidenceModule().compute(
        properties=_properties(), fused=_fused(), record=_record(),
        features=_features(), provenance={"k": "v"},
    )
    assert 0.0 <= cm.temporal_consistency <= 1.0
    assert 0.0 <= cm.data_completeness <= 1.0
    assert 0.0 <= cm.model_agreement <= 1.0
    assert cm.degradation_flag is False
    assert cm.provenance == {"k": "v"}


def test_higher_volatility_lowers_consistency():
    a = ConfidenceModule().compute(properties=None, fused=None, record=None, features=_features(0.0), provenance=None)
    b = ConfidenceModule().compute(properties=None, fused=None, record=None, features=_features(0.5), provenance=None)
    assert a.temporal_consistency >= b.temporal_consistency


def test_degraded_flag_propagates_from_fusion():
    out = ConfidenceModule().compute(properties=None, fused=_fused(degraded=True), record=None, features=None, provenance=None)
    assert out.degradation_flag is True


def test_data_completeness_grows_with_more_steps():
    a = ConfidenceModule().compute(properties=None, fused=None, record=_record(("cloud_shadow_mask",)), features=None, provenance=None)
    b = ConfidenceModule().compute(
        properties=None, fused=None, features=None, provenance=None,
        record=_record(("cloud_shadow_mask", "radar_calibration", "resolution_harmonization", "tile_extraction")),
    )
    assert b.data_completeness > a.data_completeness


def test_annotate_wraps_output():
    out = ConfidenceModule().annotate("payload", properties=_properties(), fused=_fused(), record=_record(), features=_features())
    assert out.output == "payload"
    assert out.confidence is not None
