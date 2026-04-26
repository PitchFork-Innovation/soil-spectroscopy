from soilspec.publisher import (
    CAPABILITY_OUTPUT, CONFIDENCE_OUTPUT, MapPublisher, RECOMMENDATION_OUTPUT,
)
from soilspec.storage import StorageTier, StorageTierManager
from soilspec.types import (
    AOI, BoundingBox, CapabilityClassification, ConfidenceMetadata,
    RecommendationLayers,
)


def test_publish_writes_three_layers():
    storage = StorageTierManager()
    aoi = AOI(aoi_id="aoi1", bbox=BoundingBox(0, 0, 1, 1))
    cap = {"t1": CapabilityClassification(tile_id="t1", capability_class="III", score=0.6, explanation={"k": 1})}
    rec = RecommendationLayers(
        aoi_id="aoi1", priority_zones={"t1": "monitor"}, risk_areas={}, management_actions={"t1": []},
    )
    conf = {"t1": ConfidenceMetadata(0.8, 0.9, 0.7, False, {"src": "x"})}
    handles = MapPublisher(storage=storage).publish(aoi, generation_time=1000, capability=cap,
                                                     recommendations=rec, confidence=conf)
    types = sorted(h.output_type for h in handles)
    assert types == sorted([CAPABILITY_OUTPUT, RECOMMENDATION_OUTPUT, CONFIDENCE_OUTPUT])
    for h in handles:
        assert storage.exists(StorageTier.MAP, h.storage_key)


def test_capability_payload_round_trip():
    storage = StorageTierManager()
    aoi = AOI(aoi_id="aoi1", bbox=BoundingBox(0, 0, 1, 1))
    cap = {"t1": CapabilityClassification(tile_id="t1", capability_class="II", score=0.8, explanation={"x": 1})}
    rec = RecommendationLayers(aoi_id="aoi1", priority_zones={}, risk_areas={}, management_actions={})
    conf = {"t1": ConfidenceMetadata(0.5, 0.5, 0.5, False)}
    handles = MapPublisher(storage=storage).publish(aoi, 1, cap, rec, conf)
    cap_handle = next(h for h in handles if h.output_type == CAPABILITY_OUTPUT)
    payload = storage.get(StorageTier.MAP, cap_handle.storage_key)
    assert payload["type"] == CAPABILITY_OUTPUT
    assert payload["tiles"]["t1"]["capability_class"] == "II"
