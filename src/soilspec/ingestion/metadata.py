"""Metadata parser.

Normalizes provider-specific sidecar metadata into the canonical
`AssetMetadata` schema. Known-bad inputs raise typed errors, never silently
default. Runs in parallel with the MCP adapter so ingestion is not blocked
by metadata validation.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..types import AssetMetadata, BoundingBox
from .adapters import RawAsset


class MetadataValidationError(ValueError):
    """Raised when raw metadata cannot be normalized into AssetMetadata."""


_REQUIRED_FIELDS = ("observation_id", "request_id", "provider", "modality", "timestamp", "bbox")


class MetadataParser:
    """Parses raw asset metadata into a normalized record."""

    def parse(self, raw_asset: RawAsset) -> AssetMetadata:
        """Validate and (re)materialize the metadata record on the asset."""
        meta = raw_asset.metadata
        if not isinstance(meta, AssetMetadata):
            raise MetadataValidationError(
                f"raw_asset.metadata must be AssetMetadata, got {type(meta).__name__}"
            )
        for field in _REQUIRED_FIELDS:
            if getattr(meta, field, None) in (None, ""):
                raise MetadataValidationError(f"missing required metadata field: {field}")
        if not isinstance(meta.bbox, BoundingBox):
            raise MetadataValidationError("metadata.bbox must be a BoundingBox")
        if meta.timestamp < 0:
            raise MetadataValidationError("metadata.timestamp must be non-negative")
        return meta

    def parse_dict(self, payload: Mapping[str, Any]) -> AssetMetadata:
        """Parse from a raw dict (provider-shaped). Used by MCPAdapter.describe."""
        try:
            bbox = payload["bbox"]
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                bbox = BoundingBox(*bbox)
            elif not isinstance(bbox, BoundingBox):
                raise MetadataValidationError("bbox must be (min_lon,min_lat,max_lon,max_lat)")
            return AssetMetadata(
                observation_id=str(payload["observation_id"]),
                request_id=str(payload["request_id"]),
                provider=str(payload["provider"]),
                modality=str(payload["modality"]),
                timestamp=int(payload["timestamp"]),
                bbox=bbox,
                bands=tuple(payload.get("bands", ())),
                missing_entries=tuple(payload.get("missing_entries", ())),
                extra=dict(payload.get("extra", {})),
            )
        except KeyError as e:
            raise MetadataValidationError(f"missing required field: {e.args[0]}") from None
        except (TypeError, ValueError) as e:
            raise MetadataValidationError(str(e)) from None
