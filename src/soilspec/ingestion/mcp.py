"""MCP adapter.

Exposes a stable tool schema (`list`, `describe`, `fetch`) bridging arbitrary
upstream providers to the rest of the pipeline. New providers register an
adapter; no caller changes are required.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..types import AOI, BoundingBox, TimeWindow
from .adapters import AdapterRegistry, RawAsset, SourceAdapter, UnreachableSourceError
from .metadata import MetadataParser


class MCPRequestError(ValueError):
    """Raised when an MCP tool call has a malformed request payload."""


# Stable tool schema. Tests validate adapter input against this.
MCP_TOOL_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft-07/schema#",
    "title": "soilspec.mcp.tools",
    "type": "object",
    "properties": {
        "tools": {
            "type": "object",
            "properties": {
                "list": {
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "object",
                            "required": ["aoi_id", "bbox", "time_window", "modality"],
                            "properties": {
                                "aoi_id": {"type": "string"},
                                "bbox": {"type": "array", "minItems": 4, "maxItems": 4},
                                "time_window": {
                                    "type": "object",
                                    "required": ["start", "end"],
                                    "properties": {
                                        "start": {"type": "integer"},
                                        "end": {"type": "integer"},
                                    },
                                },
                                "modality": {"type": "string"},
                            },
                        },
                        "response": {"type": "array"},
                    },
                },
                "describe": {
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "object",
                            "required": ["observation_id"],
                            "properties": {"observation_id": {"type": "string"}},
                        },
                        "response": {"type": "object"},
                    },
                },
                "fetch": {
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "object",
                            "required": ["observation_id"],
                            "properties": {"observation_id": {"type": "string"}},
                        },
                        "response": {"type": "object"},
                    },
                },
            },
            "required": ["list", "describe", "fetch"],
        }
    },
    "required": ["tools"],
}


class MCPAdapter:
    """One contract over many providers.

    Internally backed by the `AdapterRegistry`. The tool surface (`list`,
    `describe`, `fetch`) intentionally matches the JSON schema in
    `MCP_TOOL_SCHEMA`.
    """

    def __init__(
        self,
        adapters: dict[str, SourceAdapter] | None = None,
        metadata_parser: MetadataParser | None = None,
    ) -> None:
        self._adapters = adapters or {}
        self._parser = metadata_parser or MetadataParser()
        self._cache: dict[str, RawAsset] = {}

    @classmethod
    def from_registry(cls, modalities: tuple[str, ...]) -> "MCPAdapter":
        adapters = {m: AdapterRegistry.create(m) for m in modalities}
        return cls(adapters=adapters)

    def list(self, request: Mapping[str, Any]) -> list[dict[str, Any]]:
        self._require(request, ("aoi_id", "bbox", "time_window", "modality"))
        try:
            aoi = AOI(aoi_id=request["aoi_id"], bbox=BoundingBox(*request["bbox"]))
            window = TimeWindow(
                start=int(request["time_window"]["start"]),
                end=int(request["time_window"]["end"]),
            )
        except (TypeError, ValueError) as e:
            raise MCPRequestError(str(e)) from None
        modality = request["modality"]
        adapter = self._require_adapter(modality)
        out: list[dict[str, Any]] = []
        for asset in adapter.fetch(aoi, window):
            self._cache[asset.observation_id] = asset
            out.append(self._summarize(asset))
        return out

    def describe(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._require(request, ("observation_id",))
        asset = self._lookup(request["observation_id"])
        meta = self._parser.parse(asset)
        return {
            "observation_id": meta.observation_id,
            "request_id": meta.request_id,
            "provider": meta.provider,
            "modality": meta.modality,
            "timestamp": meta.timestamp,
            "bbox": [meta.bbox.min_lon, meta.bbox.min_lat, meta.bbox.max_lon, meta.bbox.max_lat],
            "bands": list(meta.bands),
            "missing_entries": list(meta.missing_entries),
            "extra": dict(meta.extra),
        }

    def fetch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._require(request, ("observation_id",))
        asset = self._lookup(request["observation_id"])
        return {
            "observation_id": asset.observation_id,
            "provider": asset.provider,
            "modality": asset.modality,
            "payload": asset.payload,
            "metadata": asset.metadata,
        }

    @staticmethod
    def _require(req: Mapping[str, Any], keys: tuple[str, ...]) -> None:
        for k in keys:
            if k not in req:
                raise MCPRequestError(f"missing field: {k}")

    def _require_adapter(self, modality: str) -> SourceAdapter:
        if modality not in self._adapters:
            raise UnreachableSourceError(f"no adapter registered for modality: {modality}")
        return self._adapters[modality]

    def _lookup(self, obs_id: str) -> RawAsset:
        if obs_id not in self._cache:
            raise MCPRequestError(f"observation_id not in cache: {obs_id}")
        return self._cache[obs_id]

    @staticmethod
    def _summarize(asset: RawAsset) -> dict[str, Any]:
        m = asset.metadata
        return {
            "observation_id": m.observation_id,
            "provider": m.provider,
            "modality": m.modality,
            "timestamp": m.timestamp,
        }
