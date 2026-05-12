# PRD: Multimodal Remote Sensing for Deriving Soil Functional Properties and Land Capability Classifications

> Source: U.S. Provisional Utility Patent Application by Jerry Wang and Patrick Feng — *Multimodal Remote Sensing for Deriving Soil Functional Properties and Land Capability Classifications*. The patent (FIG. 1A–5C and paragraphs [0001]–[0084]) is the canonical reference for system architecture; this PRD operationalizes it for greenfield implementation.

## Problem Statement

End users responsible for land-related decisions, including agricultural planners, land managers, water resource planners, infrastructure engineers, environmental researchers, sustainability teams, and government policy stakeholders, need timely, spatially comprehensive, and operationally meaningful information about how soil behaves across the regions they steward. Today, the foundation of that information is in-situ field sampling: technicians physically traverse a site to measure functional soil characteristics such as moisture and salinity. While accurate at the specific points where samples are taken, this practice is spatially constricted, exhausting, and physically difficult to scale across larger or more heterogeneous regions. The result is that decision-makers either accept narrow, sparse, point-based observations as a proxy for entire watersheds and project corridors, or they delay decisions while waiting for additional fieldwork that is slow and costly to commission.

Remote sensing has been pursued as a way to extend evaluation across wider scopes, but the existing single-modality approaches each carry well-known weaknesses that surface in operational use. Optical multispectral imagery (e.g., Sentinel-2-style acquisitions) infers soil conditions only indirectly, through vegetation and surface appearance, and is degraded by cloud cover, atmospheric fluctuations, and illumination changes. Radar sensing offers all-weather material sensitivity but is noisy in isolation and difficult to interpret over time-series plots without careful calibration. Hyperspectral imagery offers richer spectral resolution for specific soil conditions but is similarly vulnerable in isolation. When any of these modalities is applied independently, confounding variables, including vegetation cover, atmospheric noise, illumination changes, and temporal variability, contaminate the signal users actually care about, leaving outputs that are not reliable enough to drive grading, irrigation, or infrastructure decisions.

Existing in silico soil mapping pipelines compound the problem rather than resolve it. Many rely on equation-based models or restrict themselves to a single modality, which limits robustness and generalizability across the heterogeneous geographies a single planner or agency typically covers. Just as importantly, these pipelines tend to predict isolated soil parameters, individual chemical or physical variables, rather than the high-level functional characteristics that modern day applications such as land grading, water conservation, and infrastructure suitability actually require. End users are then left to translate raw parameter outputs into the irrigation suitability, infrastructure suitability, or erosion risk assessments they need, often without the domain pipelines or temporal context to do so consistently.

More recent data-driven and deep learning approaches have improved per-pixel accuracy, but they typically depend on extensive in-situ ground truth sampling for training. This dependency reintroduces the original scaling and cost problem, ties model performance to the geographies where labeled samples happen to exist, and limits generalizability and applicability to downstream decision making tasks in regions where such ground truth is unavailable. Across the user base, the gap is the same: there is no scalable, region-agnostic way to convert heterogeneous satellite observations into temporally consistent, decision-ready land capability outputs without continuous field campaigns.

## Solution

The invention is a computer-implemented platform that ingests multimodal geospatial data for any user-specified area of interest and returns decision-ready land capability classifications and management recommendations as map layers, with no in-situ sampling required. Inputs include Sentinel-1 radar imagery, Sentinel-2 optical multispectral imagery, hyperspectral data, topographic vector characteristics, land-cover masks, and soil-mapping derived environmental indices. A scheduled trigger activates the system at consistent intervals to retrieve the latest satellite observations, so the analysis stays current as conditions on the ground evolve. From the user's perspective, a single soil analysis request for a location is sufficient to drive the entire pipeline.

Once data is ingested through an MCP adapter and metadata parser, the system harmonizes it through a chain of preprocessing modules: data imputation for non-uniform sampling, resolution harmonization across satellite streams, cloud and shadow masking, noise reduction, and radar backscatter calibration, with imagery and vector data routed through dedicated pathways before being aligned to a common spatial grid and temporal index. The cleaned, aligned data flows through parallel feature extraction pathways: a spectral encoding module (built from one-dimensional CNNs, transformer-based encoders, or autoencoders) that produces latent spectral embeddings, and a spatial encoding module (built from CNNs or vision transformers) that produces latent spatial embeddings capturing terrain structure, spatial context, and vegetative cover. A multimodal fusion operator, using concatenation, attention-based weighting, gating, or deep learning modules, combines the spectral and spatial embeddings into a unified representation organized into capability channels indexed by location and time.

The fusion engine accumulates these unified representations across observation intervals to construct a temporally-resolved soil dataset. From this dataset, an inference component, implemented as an ensemble of classic machine learning, deep learning, and mathematical interpolation models combined through a fusion meta-model, derives concrete soil functional properties such as the Soil Moisture Index (SMI), surface-layer infiltration potential, and erosion susceptibility. A temporal feature extraction module then derives behavior-over-time features, including trends, rates of change, persistence, volatility, recovery following environmental events, and deviations from historical baselines, which are passed to an expert-ensemble temporal analysis stage covering trend detection, anomaly identification, and behavior classification.

The temporal analysis outputs feed a recommendation logic engine that combines rule-based and learned decision processes to produce land capability classifications and ecohydrological or land management recommendation outputs, including irrigation guidance layers, infrastructure suitability indicators, priority zones, and risk areas, along with confidence indicators reflecting temporal consistency, data completeness, and model agreement. Results are published as map layer visualizations and refreshed continuously as new satellite observations arrive. Because the platform learns from self-supervised representations of multimodal satellite data rather than from labeled field samples, it is region-agnostic, scalable across all environmental and operational contexts, and adaptable to varying data availability, giving end users consistent, data-driven guidance grounded in holistic analysis of underlying soil behavior rather than isolated point measurements.

## User Stories

### Agricultural Planner / Farm Operations Manager

1. As an agricultural planner, I want a way to submit a polygon defining my farm's boundaries as a soil analysis request, so that I receive functional soil property estimates for only the land I manage.
2. As an agricultural planner, I want a Soil Moisture Index (SMI) layer derived from fused Sentinel-1 radar and Sentinel-2 optical observations, so that I can plan planting dates without waiting for in-situ probe data.
3. As an agricultural planner, I want infiltration potential estimates for each field tile, so that I can route irrigation to plots most likely to absorb applied water efficiently.
4. As an agricultural planner, I want erosion susceptibility maps refreshed each time new Sentinel passes are ingested, so that I can intervene on at-risk plots before a storm event.
5. As an agricultural planner, I want temporal trend, volatility, and recovery features summarizing each field over the growing season, so that I can rank fields by stability and prioritize remediation.
6. As an agricultural planner, I want continuously updated land capability classifications produced by the scheduled update loop, so that my crop rotation plan reflects current soil behavior rather than last year's snapshot.
7. As an agricultural planner, I want confidence indicators attached to every soil functional property estimate, so that I know when a low-data tile warrants manual scouting before acting.

### Land Manager / Rancher

8. As a rancher, I want ecohydrological recommendation layers showing priority grazing zones derived from temporal soil behavior, so that I can rotate herds to areas with healthier recovery patterns.
9. As a land manager, I want anomaly identification outputs from the expert temporal models, so that I can be alerted to unexpected degradation events on my parcels.
10. As a land manager, I want behavior classification labels (improving, stable, stressed, degrading) for each spatial tile, so that I can communicate land health to absentee owners with simple categories.
11. As a land manager, I want deviations-from-historical-baseline features surfaced in the UI, so that I can distinguish seasonal variability from genuine long-term decline.
12. As a land manager, I want the system to run without requiring me to collect in-situ samples, so that I can assess remote or inaccessible parcels at scale.

### Water Resource Planner / Irrigation Engineer

13. As a water resource planner, I want irrigation guidance layers generated by the recommendation logic engine, so that I can allocate water across a watershed based on inferred soil water-holding behavior.
14. As an irrigation engineer, I want infiltration potential outputs at sub-field tile resolution, so that I can size and place drip versus pivot systems appropriately.
15. As a water resource planner, I want persistence features quantifying how long soil moisture lingers after precipitation, so that I can model groundwater recharge contributions across my district.
16. As a water resource planner, I want trend detection outputs from the expert ensemble, so that I can identify aquifer-recharge zones whose surface soils are drying out year over year.
17. As an irrigation engineer, I want recovery-behavior features following dry-down events, so that I can calibrate irrigation scheduling models against observed soil response.

### Infrastructure / Civil Engineer

18. As a civil engineer, I want infrastructure suitability indicators for candidate development sites, so that I can pre-screen parcels before commissioning geotechnical surveys.
19. As a civil engineer, I want erosion susceptibility classifications along proposed road and pipeline corridors, so that I can route around high-risk soils.
20. As a civil engineer, I want capability classes derived from temporal trends rather than single observations, so that I am not misled by an anomalously dry or wet day of imagery.
21. As an infrastructure planner, I want explanation metadata accompanying each capability class, so that I can defend siting decisions to regulators with traceable model reasoning.

### Environmental Researcher / Soil Scientist

22. As a soil scientist, I want access to the temporally resolved dataset of fused spatial-spectral feature vectors, so that I can study soil evolution at locations without instrumented field sites.
23. As an environmental researcher, I want the latent spectral embeddings produced by the spectral encoder, so that I can perform downstream clustering of soil signatures across regions.
24. As a soil scientist, I want the system to ingest open-source soil grids alongside Sentinel imagery, so that my analyses are grounded in established pedological priors.
25. As an environmental researcher, I want the option to inject precise in-situ measurements as additional validation data, so that I can reinforce inference accuracy in my study area.
26. As a soil scientist, I want the system's spectral pipeline to impute missing bands when band-quality filtering rejects them, so that partially obscured passes remain usable for my analysis.
27. As an environmental researcher, I want context patches incorporated into the spatial encoder's window, so that edge tiles in my study extent are not analyzed in isolation.

### Government Policy / Land-Use Stakeholder

28. As a land-use policy analyst, I want categorical land capability classes aggregated to administrative boundaries, so that I can publish region-level reports without exposing parcel-level data.
29. As a government stakeholder, I want continuously re-classified outputs as new Sentinel observations arrive, so that policy dashboards always reflect the latest soil status.
30. As a policy analyst, I want capability scoring engine outputs alongside rule-engine aggregations, so that I can audit how raw scores translate into final classifications.
31. As a government planner, I want the system to operate without in-situ sampling demands, so that I can extend land-use programs into regions where field campaigns are infeasible.

### Sustainability / ESG Analyst

32. As an ESG analyst, I want temporal degradation and improvement trends per asset, so that I can quantify land-stewardship impact across a portfolio of properties.
33. As a sustainability analyst, I want recommendation outputs identifying risk areas and priority zones, so that I can direct restoration spending where it has the highest leverage.
34. As an ESG analyst, I want confidence indicators reflecting model agreement and data completeness, so that I can flag low-evidence claims in disclosure reports.
35. As a sustainability analyst, I want historical baselines to be tracked in the temporal dataset, so that I can compute multi-year change metrics defensibly.

### ML Engineer / Data Scientist

36. As an ML engineer, I want the spectral encoder to be swappable between 1D CNNs, transformer encoders, autoencoders, and statistical extractors, so that I can experiment with the best architecture per dataset.
37. As an ML engineer, I want the spatial encoder to support vision transformers, CNNs, or autoencoders interchangeably, so that I can match model complexity to compute budget.
38. As a data scientist, I want the multimodal fusion operator to be configurable across concatenation, attention-weighting, gating, and deep learning fusion, so that I can ablate fusion strategies on the same upstream features.
39. As an ML engineer, I want the inference engine implemented as an ensemble of classic ML, deep learning, and mathematical interpolation models combined by a fusion meta-model, so that I can rely on per-component weaknesses being offset.
40. As a data scientist, I want the back-propagation feedback loop in the spectral and spatial autoencoders to be exposed for fine-tuning, so that I can adapt embeddings to new biomes.
41. As an ML engineer, I want the lifting layer projecting fused embeddings into higher-dimensional space to be tunable, so that I can balance richness against overfitting.
42. As a data scientist, I want the expert temporal models (trend detection, anomaly identification, behavior classification) to be independently trainable, so that I can update one expert without retraining the others.
43. As an ML engineer, I want versioned models stored in the model store, so that I can roll back fusion configurations or learned rule engines safely.
44. As a data scientist, I want temporal feature extraction (trends, rates of change, persistence, volatility, recovery, baseline deviations) to be standardized, so that downstream experts and rule engines consume a stable feature contract.
45. As an ML engineer, I want cross-modal influence between the spectral and spatial pathways to be optionally enabled, so that I can test joint vs. independent encoding regimes.

### DevOps / Platform Engineer

46. As a DevOps engineer, I want the scheduled trigger to fire at consistent intervals to retrieve the latest Sentinel observations, so that the temporal dataset stays current without manual intervention.
47. As a platform engineer, I want a decision module that defers analysis when temporal samples are insufficient, so that compute is not wasted on under-conditioned tiles.
48. As a DevOps engineer, I want distinct storage tiers for raw data, preprocessed tiles, embeddings, the temporal dataset, models, and the cached map repository, so that I can independently scale and back up each layer.
49. As a platform engineer, I want components to run flexibly across distributed infrastructure, cloud virtual environments, or local devices, so that I can deploy the system at the edge or in regional data centers.
50. As a DevOps engineer, I want the unaltered raw data store retained for fallback retrieval and documentation, so that I can reproduce any historical inference end-to-end.
51. As a platform engineer, I want the MCP adapter to expose a tool schema bridging ingestion to preprocessing, so that I can plug in new data sources without rewriting downstream filters.
52. As a DevOps engineer, I want metadata parsers extracting timestamps, request IDs, GPS, and missing-entry flags up front, so that bad records fail fast before consuming GPU time.

### API Consumer / Third-Party Integrator

53. As an API consumer, I want to POST a soil analysis request specifying a geographic area and receive an asynchronous job ID, so that I can integrate the pipeline into my own application.
54. As a third-party integrator, I want to subscribe to map-layer updates whenever the cached map repository is refreshed, so that my downstream UI reflects new classifications immediately.
55. As an API consumer, I want to retrieve fused capability channel vectors for a tile and timestamp, so that I can build my own analytics on top of the learned feature dimensions.
56. As an integrator, I want stable request IDs propagated through ingestion, preprocessing, embedding, and inference, so that I can trace an analysis through every stage.
57. As an API consumer, I want to query the temporal dataset by spatial location and time range, so that I can pull historical sequences for time-series modeling on my side.
58. As a third-party integrator, I want to override individual preprocessing modules (cloud masking, radar calibration, resolution harmonization, imputation), so that I can substitute organization-specific implementations.

### End User via the Map UI

59. As a map UI user, I want to draw or upload a region of interest and trigger a soil analysis request, so that I can explore any geography without leaving the browser.
60. As a map UI user, I want to toggle between SMI, infiltration potential, and erosion susceptibility layers, so that I can compare functional properties side by side.
61. As a map UI user, I want a time slider over the temporally resolved dataset, so that I can scrub through historical soil behavior for the same parcel.
62. As a map UI user, I want final land capability class visualizations rendered directly in the interface, so that I can interpret results without post-processing.
63. As a map UI user, I want optional attention layers rendered on top of capability classes, so that I can see which spatial regions most influenced a classification.
64. As a map UI user, I want recommendation outputs (priority zones, risk areas, management actions) displayed as distinct layers, so that I can act on guidance rather than only on raw scores.
65. As a map UI user, I want confidence and explanation metadata surfaced when I click a tile, so that I understand the certainty behind any recommendation before sharing it.
66. As a map UI user, I want the visualization to refresh automatically once the scheduled update loop produces new outputs, so that I never see stale soil classifications.

### Cross-Cutting Pipeline Capabilities

67. As a system operator, I want spatial samples routed through cloud and shadow masking, radar calibration, resolution harmonization, and tile extraction with spatial reprojection, so that imagery from heterogeneous sensors is harmonized before encoding.
68. As a system operator, I want vector samples routed through missing value imputation, numeric variable normalization, attribute filtering, and geospatial alignment, so that ground-truth and soil-grid data align with imagery tiles.
69. As a system operator, I want both pathways converged through a final geospatial alignment step, so that downstream encoders see one consistent spatial-temporal index.
70. As a system operator, I want the fusion engine to accumulate fused vectors over all observation intervals into a temporally resolved dataset, so that single-pass artifacts cannot dominate downstream inference.
71. As a system operator, I want the rules engine and self-supervised decision model to operate jointly over the temporal dataset, so that interpretable recommendations are grounded in both learned and rule-based logic.
72. As a system operator, I want the capability scoring engine to extract characteristic scores that the rules engine then composes with weights into an aggregate classification, so that final land capability classes are auditable end to end.
73. As a system operator, I want the pipeline to gracefully omit or parallelize preprocessing modules depending on available data sources, so that partial-data regions still receive a best-effort classification.

## Implementation Decisions

### Guiding Principles

- Favor deep modules: each major component exposes a small, stable interface and hides substantial internal complexity (encoders, fusion, ensembles, scoring, storage).
- All modules are testable in isolation with synthetic fixtures (no live satellite calls in tests).
- Configuration over code: encoders, fusion strategies, and inference ensembles are pluggable via a registry pattern so new variants ship without touching orchestration.
- Deployment-agnostic: every module accepts an execution context (local, cloud, distributed) and never assumes a specific runtime.

### Major Modules

Each module below is intended to be deep: a narrow logical interface above; substantial internal machinery below.

#### 1. Ingestion Module
- Responsibility: Acquire raw multimodal observations (Sentinel-1 SAR, Sentinel-2 multispectral, vector environmental data, optional in-situ labs, optional open soil grids) for a requested AOI and time window. Persist into the raw store with provenance.
- Interface: `fetch(aoi, time_window, modalities) -> RawObservationHandle[]`. Idempotent; handles are stable IDs into the raw store.
- Consumes: AOI geometry, time window, source configuration.
- Produces: Raw observation records (binary + metadata sidecar) in the raw store.
- Internals: Sentinel adapters, in-situ adapters, soil-grid adapters, retry/backoff, source-availability registry.

#### 2. MCP Adapter & Metadata Parser
- Responsibility: Normalize provider-specific metadata (reflectance geometry, timestamps, GPS, request IDs, sensor calibration, missing-entry markers) into a single internal metadata schema. The MCP adapter exposes a tool schema so other components fetch, list, or describe observations through one contract regardless of provider.
- Interface: `parse(raw_handle) -> NormalizedMetadata`; `list/describe/fetch` tools over MCP.
- Consumes: Raw observation handles.
- Produces: Normalized metadata records keyed by `(observation_id)`.
- Why deep: hides the long tail of per-provider metadata quirks behind one schema.

#### 3. Storage Tier Module
- Responsibility: Single abstraction over six tiers. Callers never see filesystem, object store, or DB specifics.
- Interface: `get/put/exists/list` parameterized by tier and key. Tier-specific key schemas:
  - **Raw store**: `{provider}/{observation_id}` — unaltered bytes + sidecar; serves as fallback for re-derivation.
  - **Preprocessed store**: `{aoi_id}/{tile_id}/{time}/{modality}` — harmonized tiles/tensors.
  - **Embedding store**: `{tile_id}/{time}/{modality_or_fused}` — spectral, spatial, and fused embeddings.
  - **Temporal dataset store**: `{spatial_cell_id}` -> time-ordered vector sequence.
  - **Model store**: `{model_family}/{version}` — encoder weights, fusion configs, rules engine snapshots, ensemble members.
  - **Cached map repository**: `{aoi_id}/{output_type}/{generation_time}` — published map tiles + metadata.
- Why deep: callers ask for "the embedding for tile T at time t" and never care where it lives.

#### 4. Preprocessing Module
- Responsibility: Route each ingested record down the correct pathway and emit co-aligned multimodal records.
- Interface: `preprocess(raw_handles) -> PreprocessedRecord[]` keyed by `(tile_id, time)`.
- Pathways:
  - **Spatial pathway** (in order): cloud & shadow masking → radar calibration (SAR only) → resolution harmonization → tile extraction & spatial reprojection.
  - **Vector pathway** (in order): missing-value imputation → numeric normalization → attribute filtering → geospatial alignment to common grid.
  - **Co-alignment stage**: both pathways are joined to a shared `(tile_id, time)` index before emission.
- Internals: each step is a swappable processor with a uniform `apply(tile) -> tile` contract; ordering and inclusion are config-driven so a step can be omitted when its modality is missing.

#### 5. Spectral Encoder Module (deep)
- Responsibility: Turn a per-pixel/per-tile bandwise spectral vector into a latent representation.
- Interface: `encode(spectral_input) -> SpectralEmbedding`.
- Consumes: Preprocessed bandwise vectors with band-quality metadata.
- Produces: Fixed-dimension spectral embeddings keyed by `(tile_id, time)`.
- Pluggable backends: 1D CNN, transformer encoder, autoencoder, and statistical feature extractor — selected via registry.
- Internal pipeline: band-quality filter → missing-band imputation → encoder forward pass → (training only) reconstruction head + back-propagation feedback loop.
- Why deep: callers never know which backend is active; the interface and embedding shape are stable.

#### 6. Spatial Encoder Module (deep)
- Responsibility: Turn a preprocessed raster tile into a latent spatial embedding capturing terrain, context, and vegetation structure.
- Interface: `encode(raster_tile) -> SpatialEmbedding`.
- Pluggable backends: CNN, Vision Transformer, autoencoder.
- Internal pipeline: sliding-window patch selection → context-patch incorporation for adjacency → encoder forward pass → (training only) reconstruction + back-prop loop.
- Why deep: window size, context strategy, and backend choice are encapsulated.

#### 7. Multimodal Fusion Engine (deep)
- Responsibility: Combine spectral and spatial embeddings into a unified representation organized into capability channels.
- Interface: `fuse(spectral_emb, spatial_emb, missing_mask) -> FusedRepresentation` keyed by `(tile_id, time)`.
- Pluggable strategies: concatenation, attention-based weighting, gating, deep-learning fusion module — selected via configuration.
- Capability channels: the fused vector is partitioned into named channel groups (moisture-relevant, infiltration-relevant, erosion-relevant, etc.) so downstream inference can attend selectively.
- Missing-modality fallback: if a modality is absent, fusion runs in degraded mode (e.g., pass-through of the available modality with a learned imputation prior) and stamps the output with a degradation flag.

#### 8. Temporal Dataset Module
- Responsibility: Maintain a time-ordered store of fused representations and inferred properties keyed by spatial location.
- Interface: `append(spatial_cell_id, time, vector)`, `series(spatial_cell_id, window) -> TimeSeries`, `sufficient(spatial_cell_id, criteria) -> bool`.
- Internals: vector store with time index; sufficiency predicate (count, recency, gap distribution) used by the scheduled loop.

#### 9. Temporal Feature Extractor (deep)
- Responsibility: Derive higher-order temporal descriptors from a time series of soil-related vectors.
- Interface: `extract(series) -> TemporalFeatureSet`.
- Output features: trends, rates of change, persistence, volatility, recovery behavior after environmental events, deviations from historical baselines.
- Why deep: hides choice of baseline definition, event detection, and statistical estimators.

#### 10. Ensemble Inference Engine (deep)
- Responsibility: Map fused multimodal embeddings to soil functional property estimates (SMI, infiltration potential, erosion susceptibility, and other configured channels).
- Interface: `infer(fused_repr) -> SoilFunctionalProperties`.
- Internal architecture:
  - Lifting layer projects fused embeddings into a higher-dimensional space.
  - Member models run in parallel: classic ML, deep learning, mathematical interpolation.
  - Fusion meta-model combines member outputs into a unified estimate with calibrated uncertainty.
- Why deep: members and the meta-model are swappable; callers see only properties + confidence.

#### 11. Temporal Analysis Module (Expert Ensemble)
- Responsibility: Analyze temporal soil behavior to identify stress, degradation, improvement, or anomalous behavior.
- Interface: `analyze(temporal_features, current_state) -> TemporalSignals`.
- Internal architecture: a soil-evolution inference model produces preliminary relationships; specialized expert sub-models (trend detection, anomaly identification, behavior classification) refine the analysis; outputs are merged for downstream consumption.

#### 12. Recommendation Logic Engine
- Responsibility: Translate temporal signals plus inferred properties into actionable recommendations and ecohydrological/management layers.
- Interface: `recommend(temporal_signals, properties, aoi) -> RecommendationLayers`.
- Internals: rule-based, learned, and hybrid decision processes selectable per deployment; outputs include priority zones, risk areas, and management actions.

#### 13. Capability Scoring & Rules Engine (deep)
- Responsibility: Convert temporal decision features into per-cell characteristic scores and aggregate them into ordinal/categorical land capability classes.
- Interface: `score(temporal_decision_features) -> CharacteristicScores`; `classify(scores) -> CapabilityClass`.
- Internals: ML scoring head (per characteristic), rules engine that composes scores with configurable weights, class boundary table.
- Why deep: callers receive a class label and an explanation; the internal scoring/aggregation strategy is hidden.

#### 14. Map Publisher
- Responsibility: Render capability classifications, recommendation layers, and confidence layers as cacheable map tiles and write them to the cached map repository.
- Interface: `publish(aoi, generation_time, layers) -> MapHandle`.

#### 15. Confidence & Explanation Module
- Responsibility: Compute per-output metadata: temporal consistency, data completeness, model agreement (across ensemble members), and degradation flags from missing modalities.
- Interface: `annotate(output, evidence) -> AnnotatedOutput`. Attached to every property estimate, recommendation, and capability class.

#### 16. Pipeline Orchestrator
- Responsibility: Drive end-to-end runs and the scheduled-update loop. Responsible only for orchestration; never owns model logic.
- Interface: `run_request(request) -> RunResult`; `tick()` for scheduled execution.
- Internals: DAG of stages (ingest → preprocess → encode → fuse → append-temporal → sufficiency-check → infer → analyze → recommend → classify → publish), checkpointing per stage, retries with backoff, idempotent restarts from any stage using storage-tier handles.

### Architectural Decisions

- **Pipeline orchestration**: a stage DAG with explicit inputs/outputs by storage-tier key. Every stage is independently retryable and resumable; no stage holds state outside the storage tier.
- **Scheduled update loop**: a periodic trigger fetches the latest Sentinel observations, runs preprocessing and embedding, appends to the temporal dataset, then performs a sufficiency check before invoking inference, temporal analysis, and capability scoring. Insufficient data simply waits for the next interval.
- **Continuous outputs**: outputs are regenerated on each tick that produces new sufficient data; downstream consumers always read the latest published map handle.
- **Fault tolerance**: stage-level checkpointing to the appropriate storage tier; failed stages re-run from the last successful tier without re-fetching upstream data; ingestion errors fall back to the raw store for replay.
- **Missing-modality fallback**: every stage advertises which modalities it requires vs. tolerates. Preprocessing skips inapplicable steps; encoders run only on available modalities; fusion enters a degraded mode and stamps a degradation flag; confidence module reflects the degradation.
- **Deployment flexibility**: each module accepts an execution context (local process, cloud function, distributed worker). The orchestrator schedules stages onto any backend; storage tiers abstract over local filesystem vs. object storage vs. geospatial DB.
- **Configuration registry**: encoder backends, fusion strategies, ensemble members, and rules engines are registered by name and selected per deployment, enabling A/B comparison without code changes.

### Data Ingestion Decisions

- **Sources**: Sentinel-1 (SAR) and Sentinel-2 (multispectral) as primary remote sensing inputs; vector environmental data (topology, land-cover masks, soil-mapping indices); optional in-situ lab measurements and open soil grids for training/validation reinforcement.
- **MCP adapter**: exposes a uniform tool schema (`list`, `describe`, `fetch`) so the rest of the system pulls observations through one contract. New providers are added by registering a new adapter; no caller changes.
- **Metadata parser**: extracts timestamps, request IDs, GPS bounds, reflectance geometry, and missing-entry markers into the normalized metadata schema. Runs in parallel with the MCP adapter so ingestion is not blocked by metadata processing.
- **Raw store fallback**: every ingested observation lands in the raw store unmodified before any transformation. Any downstream stage can re-derive its inputs from the raw store, supporting replay, schema migrations, and audit.

### Storage Tiering

| Tier | Format | Key Strategy | Purpose |
|---|---|---|---|
| Raw store | Provider-native bytes + sidecar JSON | `{provider}/{observation_id}` | Unaltered fallback, audit, replay |
| Preprocessed store | Harmonized tiles/tensors | `{aoi_id}/{tile_id}/{time}/{modality}` | Aligned, calibrated inputs to encoders |
| Embedding store | Fixed-dim vectors | `{tile_id}/{time}/{modality_or_fused}` | Spectral, spatial, and fused embeddings |
| Temporal dataset store | Time-ordered vector sequences | `{spatial_cell_id}` -> ordered (time, vector) | Sequential analysis input |
| Model store | Serialized weights + configs | `{model_family}/{version}` | Encoders, fusion configs, ensemble members, rules engines |
| Cached map repository | Map tiles + metadata | `{aoi_id}/{output_type}/{generation_time}` | Published outputs for serving |

All keys include version stamps so re-runs with new model versions do not overwrite prior results.

### Preprocessing Decisions

- **Pathway routing**: an early classifier directs each record to the spatial or vector pathway based on metadata.
- **Spatial pathway ordering** (strict): cloud and shadow masking → radar calibration → resolution harmonization → tile extraction & spatial reprojection. Cloud masking precedes calibration so noisy pixels do not bias normalization. Resolution harmonization precedes tiling so tiles are uniform across modalities.
- **Vector pathway ordering**: missing-value imputation → numeric normalization → attribute filtering → geospatial alignment to the common grid.
- **Co-alignment**: both pathways converge on a join keyed by `(tile_id, time)` to produce a single preprocessed multimodal record.
- **Step toggling**: any step can be omitted when its modality is absent; the pathway descriptor records which steps actually ran for downstream confidence accounting.

### Feature Extraction Decisions

- **Parallel encoders**: spectral and spatial encoders run independently on the same `(tile_id, time)` records. Cross-modal influence is allowed via an optional cross-encoder hook but is off by default.
- **Spectral encoder** options (registry): 1D CNN, transformer encoder, autoencoder, statistical features. Includes a band-quality filter; if too few valid bands, missing-band imputation is invoked before encoding. Training uses a back-propagation feedback loop with reconstruction.
- **Spatial encoder** options (registry): CNN, ViT, autoencoder. Uses sliding-window patch extraction with context patches incorporated to preserve adjacency. Trained via the same back-prop reconstruction loop.
- **Embedding contract**: encoders emit fixed-dimension vectors. Backend changes do not change the interface; only the model store key.

### Fusion Decisions

- **Strategies (configurable)**: concatenation, attention-based weighting, gating, deep-learning fusion module. Selected by configuration per deployment.
- **Capability channels**: the fused output is partitioned into named channel groups so downstream inference for SMI, infiltration potential, and erosion susceptibility attends to relevant subspaces.
- **Output**: a single fused representation indexed by `(tile_id, time)`, written to the embedding store under a `fused` modality key.
- **Degraded fusion**: if only one modality is present, fusion runs in single-modality pass-through mode and emits a degradation flag.

### Temporal Dataset Decisions

- **Structure**: time-ordered vector store keyed by spatial cell. Each cell stores a sequence of fused representations and, after inference, the corresponding soil functional property vectors.
- **Sufficiency check**: predicate parameterized by minimum sample count, minimum coverage window, and maximum gap; consulted by the scheduled loop before running inference and analysis.
- **Temporal feature extraction**: trends, rates of change, persistence, volatility, recovery behavior following environmental events, deviations from historical baselines. Baselines are stored per cell and updated on a slower cadence than observations.

### Inference Decisions

- **Ensemble**: classic ML model + deep learning model + mathematical interpolation model run in parallel; outputs are combined by a fusion meta-model into a single property vector with calibrated uncertainty.
- **Lifting layer**: projects fused embeddings into higher-dim space before ensemble members consume them, giving each member capacity to learn richer relationships.
- **Outputs**: SMI, infiltration potential, erosion susceptibility, and any additional configured properties — each with confidence and model-agreement metadata.
- **Validation reinforcement**: when in-situ measurements are supplied, they are routed to a training/testing benchmark path that updates ensemble members; this path is fully optional and the system runs without it.

### Temporal Analysis Decisions

- **Expert ensemble**: a soil-evolution inference model produces preliminary cross-time relationships; outputs feed expert sub-models for trend detection, anomaly identification, and behavior classification. Each expert is independently versioned in the model store.
- **Recommendation logic engine**: consumes expert outputs plus current properties; applies rule-based, learned, or hybrid decision processes to emit recommendation layers (priority zones, risk areas, management actions) and to feed the capability scoring engine.

### Capability Scoring & Classification Decisions

- **Scoring**: capability scoring engine produces one or more characteristic scores from temporal decision features via an ML head.
- **Rules engine**: aggregates characteristic scores with recomposed weights into an ordinal/categorical land capability class. Weight tables and class boundaries are versioned in the model store.
- **Aggregation**: per-cell classifications accumulate across the AOI to form a final land capability map. Optional attention overlays can be rendered on top of the base map for downstream geospatial analysis.
- **Publishing**: capability map, recommendation layers, and confidence layer are written to the cached map repository keyed by `(aoi_id, output_type, generation_time)`.

### Scheduled Update Loop Decisions

- **Trigger**: configurable interval per AOI.
- **Steps per tick**: retrieve latest Sentinel-1/2 observations → preprocess → encode → fuse → append to temporal dataset → check sufficiency.
- **Branching on sufficiency**: if insufficient, wait for the next interval; if sufficient, run temporal analysis, recommendation logic, and capability scoring, then publish updated maps.
- **Continuous refinement**: outputs are republished on every tick that yields new sufficient data; older outputs remain available via their generation-time keys for diffing and audit.

### Confidence & Explanation Metadata

Every output (property estimate, recommendation, capability class) carries:
- **Temporal consistency**: stability of the estimate across recent observations.
- **Data completeness**: which modalities and preprocessing steps were available; degradation flags propagated from fusion.
- **Model agreement**: spread across ensemble members and across temporal experts.
- **Provenance pointers**: handles into the raw, preprocessed, embedding, and model stores used to produce the output.

### API Contract

- **Analysis request**
  - Inputs: AOI geometry, time window, optional in-situ measurements, optional modality preferences, optional output selection (capability map / recommendations / properties).
  - Behavior: synchronous request returns a job handle; the scheduled loop owns execution; the handle resolves to map handles once published.
- **Analysis response (per output type)**
  - Capability map layer (ordinal/categorical classes per cell).
  - Recommendation layers (priority zones, risk areas, management actions).
  - Confidence layer (temporal consistency, data completeness, model agreement) co-registered to the capability map.
  - Provenance metadata: model versions, preprocessing pathway descriptors, source observation IDs.
- **Stability**: the request/response contract is independent of internal model choices; encoder, fusion, and ensemble swaps do not alter the API.

## Testing Decisions

### What makes a good test in this codebase

Tests in this codebase pin down the **external behavior and contract** of each deep module — what inputs are accepted, what outputs are produced, and what invariants hold across calls — rather than asserting on internal implementation details (private helpers, intermediate tensor shapes that aren't part of the public API, the exact sequence of internal calls, or the choice of underlying library). The patent describes alternative implementations for nearly every component (e.g., the spectral encoder may be a 1D CNN, transformer, autoencoder, or statistical extractor; fusion may be concat, attention, gating, or classic ML). Tests must therefore lock down the *contract* — input schema, output schema, determinism guarantees, error behavior on missing data — so that the implementation underneath can be swapped without rewriting tests.

Concretely, a good test:
- Treats the module under test as a black box; mocks only its declared collaborators.
- Asserts on the shape, dtype, value range, and semantic invariants of outputs — not on weights, intermediate activations, or call order of internal helpers.
- Uses small synthetic fixtures (a few bands, a few tiles, a handful of timestamps) so failures are easy to diagnose.
- Fails for one well-defined reason, and the failure message identifies which contract was broken.

There is **no in-tree prior art** at this time — the repo is greenfield. Conventions below borrow from comparable open-source projects (see "Prior art" below).

### Unit tests (deep modules tested in isolation)

Each module below gets its own unit-test module. Each line states the contract the tests must verify; implementation may change freely below the contract.

**Ingestion layer**

- **Data ingestion / source adapters**: given a request for a region + time window, returns a normalized iterable of raw assets (Sentinel-1, Sentinel-2, SoilGrids, in-situ lab data) with stable IDs; honors caching and retries; raises typed errors for unreachable sources.
- **MCP adapter**: given a tool-schema-conformant request, routes structured multi-modal inputs through the configured filter chain; the adapter's tool-schema is stable and validated against a checked-in JSON schema.
- **Metadata parser**: given a raw asset, extracts a complete `AssetMetadata` record (timestamps, request IDs, GPS bounds, sensor geometry, band list, missing-entry flags); known-bad inputs surface as typed validation errors, never silent defaults.

**Preprocessing pipeline — spatial pathway**

- **Cloud masking**: given a Sentinel-2 tile + QA band, returns a boolean mask matching tile shape; pixels flagged as cloud/shadow are masked; idempotent under repeat application.
- **Radar calibration**: given Sentinel-1 backscatter, returns calibrated sigma-nought in dB on a documented scale; output range is bounded; calibration is deterministic.
- **Resolution harmonization**: given inputs at heterogeneous GSDs, returns rasters on a common grid with declared CRS, transform, and pixel size; nodata is preserved; no silent reprojection drift across repeated calls.
- **Tile extraction & spatial reprojection**: given a large scene + AOI, returns tiles of the configured size, each carrying a precise geo-transform, with the AOI fully covered and tile bounds tiling without overlap or gap.

**Preprocessing pipeline — vector pathway**

- **Imputation**: given vector records with NaNs, returns records with no NaNs in declared-required columns; imputation strategy is configurable; original non-null values are not mutated.
- **Normalization**: given numeric features, returns features with documented mean/variance (or min/max) per column; fit-then-transform on identical data is deterministic.
- **Attribute filtering**: given a record set + an allow/deny schema, returns only retained attributes; contract is schema-driven, not column-name string-matching inside tests.
- **Geospatial alignment**: given vector records + raster tiles, returns records reprojected/joined to the raster CRS and clipped to AOI.

**Co-alignment**

- **Co-alignment step**: given a spatial output and a vector output, asserts both share the same `(tile_id, time_index, CRS, bounds)` key; mismatched pairs raise an explicit `MisalignedSampleError`.

**Encoders**

- **Spectral encoder**: outputs a fixed-size latent vector (`latent_dim` configurable, default verified); deterministic given a fixed seed and inputs; handles missing bands by invoking the band-quality filter / imputation path and never propagating NaNs.
- **Spatial encoder**: consumes preprocessed raster tiles via a sliding window; correctly handles configured `patch_size`, `stride`, and `context_patches`; outputs latent vectors of the documented `latent_dim`; padding behavior at tile edges is well-defined and documented.

**Fusion**

- **Multimodal fusion engine**: supports each strategy independently — `concat`, `attention`, `gating` — with strategy-specific tests verifying output shape, that each strategy is deterministic given a seed, and that switching strategies does not change the public input/output contract.
- **Capability channel construction**: produces the configured number of capability channels (e.g., SMI, infiltration potential, erosion susceptibility) keyed by `(tile_id, time_index)`; channel ordering is stable and documented.

**Temporal stack**

- **Temporal dataset store**: supports `insert`, `query_by_tile_and_time`, and `query_range`; ordering invariant — query results are sorted by time ascending; idempotent insert (same key + same payload is a no-op); insert with conflicting payload raises.
- **Temporal feature extractor**: computes each documented feature family — trends, rates of change, persistence, volatility, recovery (post-event), baseline deviation — each producing a vector of fixed shape; behavior on short series (below minimum sufficiency) is explicit (returns `None` or raises `InsufficientHistoryError`).

**Inference**

- **Ensemble inference engine**: produces a soil functional property vector of declared shape; verifies each ensemble member independently — classic ML, deep learning, mathematical interpolation, fusion meta-model — and verifies the meta-model output is bounded by member outputs in a documented way.
- **Expert ensemble**: individually exercises trend detection, anomaly identification, and behavior classification experts; each expert's output schema is fixed; downstream combiner is tested with stub experts.

**Recommendation & capability**

- **Recommendation logic engine**: given soil functional properties + temporal features, returns recommendations of the documented schema (priority zones, risk areas, management actions); pure-function for a fixed rule set + inputs.
- **Capability scoring engine**: given temporal decision features, returns characteristic scores in `[0, 1]` per dimension; deterministic for a fixed model.
- **Rules engine**: given characteristic scores + weights, returns a single ordinal land capability class; rules are loaded from a versioned config; given the same config + inputs, the output is identical.

**Orchestration**

- **Scheduled update loop**: on each tick, retrieves new observations, runs preprocessing → embedding → temporal-store insert; the **sufficiency gate** correctly defers analysis until the temporal dataset contains at least the configured minimum samples; if not sufficient, the loop reschedules without invoking the inference stack.

**Storage**

- **Storage tier abstraction**: each tier — raw, preprocessed, embedding, temporal, model, map cache — implements a common `get/put/list/exists` interface; tests verify the contract once on an in-memory backend and parametrize across configured backends; key schemas (e.g., embeddings keyed by `(tile_id, time)`) are validated.

### Integration tests

- **End-to-end pipeline on a small fixture region**: a few synthetic Sentinel-1 + Sentinel-2 tiles over a small AOI, with a 3–5 timestamp window, exercises ingestion → preprocessing → encoding → fusion → temporal store → inference → capability classification, asserting the final map has the expected shape, valid class labels, and tile coverage matches the AOI.
- **Modality-missing behavior**: identical pipeline run with **Sentinel-1 only** and **Sentinel-2 only** — the system must produce a valid (possibly lower-confidence) output rather than crash, exercising the patent's claim that the system is "adaptable to varying data availability."
- **Scheduled update loop wiring**: drive the scheduler with a fake clock, feed a stream of observations, verify the temporal store grows monotonically, the sufficiency gate fires at the configured threshold, and downstream recommendations are produced exactly once per qualifying tick.

### Determinism and reproducibility tests

- **Same seed + same inputs ⇒ same outputs**: applied to spectral encoder, spatial encoder, fusion engine (each strategy), ensemble inference, capability scoring. Each test runs the module twice with identical seed and input, and asserts byte-equality (or per-element float equality within a documented tolerance) on outputs.
- **Cross-process determinism**: same seed across processes / fresh interpreters yields the same outputs (catches hidden global state).

### Golden-file tests

- **Capability classifications on a fixed fixture**: a small region + short time window is checked into the test suite. The expected land capability raster is stored as a checked-in golden file. A regression in any layer of the inference stack (encoder, fusion, ensemble, scoring, rules) surfaces as a mismatch against this golden output. Golden updates are gated by an explicit `--update-goldens` flag and reviewed by a human in PR.
- **Recommendation outputs**: golden JSON of priority-zone / risk-area outputs over the same fixture region.

### Performance and scale benchmarks (non-blocking)

Run separately from the unit suite (e.g., `pytest -m bench` or a nightly job). Not gates on PRs.

- **Tile throughput**: tiles preprocessed per second.
- **Embedding throughput**: spatial and spectral embeddings produced per second.
- **Full-pipeline runtime**: end-to-end seconds over a fixed-size benchmark region (e.g., 100 km² × 12 timestamps).
- Results are logged to MLflow (or equivalent) for trend monitoring; thresholds are advisory until the system stabilizes.

### Test data strategy

- **Unit tests** use small **synthetic GeoTIFFs** (a few bands, 64×64 or 128×128 pixels, configurable nodata, configurable cloud masks) and **synthetic vector data** (small GeoJSON / Parquet records with known coordinates and attributes), generated by fixture factories. No external network access; test data is constructed in `conftest.py` or loaded from the in-repo `tests/fixtures/` directory.
- **Integration tests** use **one small real-world fixture**: a few real Sentinel-1 + Sentinel-2 tiles over a publicly known AOI for a short time window, checked in (or fetched once and cached) so CI is reproducible offline.
- **No ground-truth dependence**: per the patent's claim that the system "is free from in situ demands," tests must not require in-situ soil measurements as inputs *or* as labels for assertions. Where validation is needed, tests assert on **structural invariants** (shape, range, schema, monotonicity, determinism) and on **golden outputs from the system itself**, not on agreement with field samples. In-situ data, when present in the patent's alternative embodiments, is treated by tests as an optional auxiliary input, not a required ground-truth signal.

### Prior art

The repo is greenfield, so there is **no in-tree prior art**. Conventions are drawn from well-known patterns in comparable open-source projects:

- **rasterio** — synthetic in-memory `MemoryFile` GeoTIFFs as fixtures, parametrization over CRS / dtype / nodata, and explicit testing of windowed reads and reprojection invariants. Used as the model for the spatial preprocessing tests.
- **xarray** — fixture builders that construct small `DataArray` / `Dataset` objects with known coords, and assertion helpers (`assert_equal`, `assert_allclose`) that compare on the public data model rather than on internal storage. Used as the model for tensor-shaped outputs from encoders and the temporal store.
- **scikit-learn** — the `check_estimator` convention: every estimator-shaped module (encoders, fusion engine, scoring engine) is verified against a common contract (`fit` is idempotent, `transform` is deterministic given a seeded `random_state`, `get_params` / `set_params` round-trips, fitted-vs-unfitted state is explicit). Adopted for our encoder / fusion / scoring modules.
- **PyTorch** — unit tests that compare tensors with `torch.testing.assert_close` and explicitly seed `torch.manual_seed` per test; `pytest` markers separate CPU / GPU / slow tests. Adopted for encoder and fusion-engine determinism tests.
- **Kedro / Dagster** — pipeline / DAG tests that exercise nodes in isolation against a fake catalog, then a small end-to-end run on a tiny dataset, with the orchestration layer (scheduler, sufficiency gate) tested via fake clocks. Adopted for the scheduled update loop and end-to-end integration tests.
- **MLflow** — experiment-tracking tests that use a temp-dir tracking URI per test and assert on logged-artifact contracts rather than the storage backend. Adopted for our model store and benchmark logging tests.

These patterns are referenced as conventions, not dependencies — we may adopt different libraries, but the test *style* (small synthetic fixtures, contract-level assertions, seeded determinism, golden files for end-to-end behavior) is what we inherit.

## Out of Scope

- **Hardware / sensor design.** The system consumes existing satellite feeds (Sentinel-1, Sentinel-2, and analogous sources per FIG. 1B). Designing imagers, radars, or hyperspectral payloads is not relevant to the patent claim, which presumes upstream data already exists.
- **Manufacturing or deployment of in-situ measurement devices.** The patent's core differentiator is that the system operates "free from in situ demands" (paragraph [0007]). Building probes, soil sensors, or field-sampling hardware contradicts the design invariant.
- **Real-time / sub-hourly inference.** Inference cadence is bounded by satellite revisit intervals and the scheduled trigger described in FIG. 2C. Sub-hourly latency is not achievable from the upstream feeds and is not relevant to the claim.
- **Map UI/UX implementation details beyond layer publishability.** The PRD defines what map layers (capability classes, recommendation overlays, confidence layers) must be emitted from the cached map repository (66). Pixel-level UI design, theming, and frontend interaction polish are deferred to a downstream design milestone.
- **Authentication, billing, multi-tenancy, and user-account features.** These are platform concerns covered by existing libraries / SaaS frameworks and are orthogonal to the inventive subject matter of the patent.
- **Specific commercial integrations (Esri, Google Earth Engine, etc.).** Beyond providing a clean API contract over the outputs (38), specific connector implementations are deferred — they are integration work, not core system work, and are best built once the API contract is stable.
- **Hyperspectral data ingestion at MVP.** The patent lists hyperspectral as one supported modality (paragraph [0005]), but the MVP scopes ingestion to Sentinel-1 + Sentinel-2 + topographic vector data to keep the preprocessing surface tractable. Hyperspectral is a future extension that the architecture must accommodate but does not need to deliver at MVP.
- **Ground-truth-based supervised training pipelines.** The system is explicitly self-supervised and no-in-situ (paragraphs [0005], [0007], [0067]). Optional supervised fine-tuning with in-situ validation data is mentioned in alternative embodiments ([0067]) and is deferred to a future milestone.
- **Carbon / GHG inventory features.** These are not claimed in the patent. They may be supported in a future milestone if they fall directly out of derived soil functional properties (e.g., SMI, infiltration potential, erosion susceptibility), but are not in scope for MVP.
- **Mobile / offline-first clients.** The system is a backend-heavy geospatial pipeline producing map layers; mobile clients are not relevant to the patent claim and are deferred.
- **Localization / internationalization beyond English at MVP.** Not relevant to the inventive subject matter; deferred to a future product-polish milestone.

## Further Notes

- **Source of truth.** The patent claim language (single claim, page 18) and FIG. 1A through FIG. 5C are the canonical reference for system architecture. When ambiguity arises during implementation, defer to the figure-numbered subsystem boundaries and the corresponding paragraphs in the Detailed Description ([0027]–[0083]) rather than reinterpreting from scratch.
- **No-in-situ-dependence is a design invariant.** The patent repeatedly emphasizes that the system operates without reliance on extensive in-situ sampling (paragraphs [0004], [0007], Abstract). Implementers must preserve this property: no code path on the inference side may make in-situ data a hard prerequisite. In-situ data may only enter as optional validation/fine-tuning per [0067].
- **Storage tiering maps to a layered repository pattern.** FIG. 1C defines six distinct stores: raw (54), preprocessed (56), embedding (60), temporal dataset (62), model (64), and cached map (66). Implementers should expose each as a separate repository interface and not collapse them (e.g., do not write embeddings into the preprocessed store, do not write map outputs into the temporal store). This separation supports fallback retrieval, reproducibility, and independent versioning.
- **Fusion strategies must be configurable.** The patent's "in some embodiments" language for the fusion engine ([0034], [0064]) explicitly calls out concatenation, attention-based weighting, gating mechanisms, and deep learning modules. The fusion operator (30 / 402) should be implemented behind a strategy interface so the active fusion method is selectable per deployment / experiment, not hard-coded.
- **Ensemble inference is part of the claim.** FIG. 4B and paragraph [0070] describe an ensemble of a classic ML model (414), a deep learning model (416), a mathematical interpolation model (418), and a fusion meta-model (420). Implementers must not collapse this to a single model — the ensemble structure is load-bearing for robustness, generalization, and uncertainty tolerance, and is part of what is claimed.
- **The temporal dataset is the load-bearing artifact.** The temporal soil evolution dataset (34 / 408 / 500) is the hub through which most downstream value flows: capability classifications, recommendations, anomaly detection, trend inference, and the scheduled update loop all read from it. Treat it as a first-class, well-versioned, queryable artifact — not as an internal cache. Schema, indexing, and update semantics deserve careful design.
- **Open questions to resolve before / during implementation.**
  - Target CRS and tile size for the preprocessed store (e.g., EPSG:4326 vs. a projected CRS; 256x256 vs. 512x512 tiles).
  - Default temporal cadence for the scheduled trigger (FIG. 2C, item 240) — daily, 5-day Sentinel-2 revisit, or bespoke.
  - Model framework choice (PyTorch vs. JAX) for the spectral/spatial encoders and ensemble.
  - Cloud target (AWS vs. GCP) — affects object-storage choice for the six stores in FIG. 1C and the choice of compute for tile-parallel inference.
  - Distribution model: SaaS API vs. self-hosted deployable. Affects auth, multi-tenancy, and billing scope (which are themselves out of scope at MVP, but the choice constrains future work).
  - Definition of "sufficient temporal samples" for the decision module (250) in FIG. 2C — needs an explicit threshold.
- **Suggested phased rollout.**
  - **Phase 1:** Ingestion + preprocessing + storage tiers (raw, preprocessed) + minimal map output (cached map repository) for a single small AOI. End-to-end thin slice through the FIG. 1B / FIG. 2B pipelines.
  - **Phase 2:** Spectral and spatial encoders (FIG. 3A–3C) + fusion operator (FIG. 4A) + temporal dataset construction. Embedding store and temporal dataset store come online.
  - **Phase 3:** Ensemble inference (FIG. 4B) + functional property estimation (SMI, infiltration potential, erosion susceptibility per [0064]).
  - **Phase 4:** Expert ensemble temporal analysis (FIG. 5A / 5B) + recommendation logic engine + capability classification (FIG. 5C).
  - **Phase 5:** Scheduled update loop (FIG. 2C) for continuous refresh, plus confidence/explanation metadata per [0077].
