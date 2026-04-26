# soil-spectroscopy

Reference implementation of the multimodal remote sensing pipeline described
in PRD #1: ingest Sentinel-1/Sentinel-2/vector observations for an AOI, derive
soil functional properties (SMI, infiltration potential, erosion
susceptibility), and publish land capability classifications + management
recommendations as map layers — without in-situ sampling.

## Layout

```
src/soilspec/
  types.py                # public dataclasses (AOI, embeddings, properties, ...)
  registry.py             # name->factory registry for swappable backends
  storage/                # six-tier storage abstraction (in-memory backend)
  ingestion/              # source adapters, MCP adapter, metadata parser
  preprocessing/          # spatial & vector pathways + co-alignment
  encoders/               # spectral (1d_cnn/transformer/autoencoder/statistical)
                          # spatial (cnn/vit/autoencoder)
  fusion.py               # concat/attention/gating/deep + capability channels
  temporal/               # time-ordered dataset, feature extractor, expert ensemble
  inference.py            # lifting layer + ensemble (classic ML / DL / interp)
  recommendation.py       # rules / learned / hybrid recommenders
  capability.py           # scoring engine + rules engine (I..VIII)
  publisher.py            # cached map repository writes
  confidence.py           # per-output annotations
  orchestrator.py         # stage DAG + scheduled trigger
```

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

End-to-end run from Python:

```python
from soilspec.orchestrator import PipelineOrchestrator
from soilspec.types import AnalysisRequest, AOI, BoundingBox, TimeWindow

o = PipelineOrchestrator()
res = o.run_request(AnalysisRequest(
    aoi=AOI(aoi_id="aoi1", bbox=BoundingBox(0, 0, 1, 1)),
    time_window=TimeWindow(start=0, end=30 * 86400),
))
for h in res.map_handles:
    print(h.output_type, h.storage_key)
```

## Tests

`pytest` runs unit + integration + determinism + golden tests (~125 tests).
Use `pytest -m golden --update-goldens` to refresh golden fixtures.

The pipeline is deterministic across processes: backends derive seeds via
SHA-256 of their configuration rather than Python's randomized `hash()`.
