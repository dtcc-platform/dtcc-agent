"""GeoJSON loading, querying, and property summarization.

Stores GeoJSON FeatureCollections in the object store for chaining
with other tools (inspect, delete, query, summarize).
"""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Any

import numpy as np


# -- Load -------------------------------------------------------------------

def load_geojson(file_path: str) -> dict[str, Any]:
    """Load a GeoJSON FeatureCollection from disk.

    Returns a dict with summary metadata and, for small datasets
    (<=500 features), the full features array.
    """
    path = Path(file_path)
    if not path.is_file():
        return {"error": f"File not found: {file_path}"}
    if not path.suffix.lower() == ".geojson" and not path.suffix.lower() == ".json":
        return {"error": f"Unsupported file type: {path.suffix}"}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return {"error": f"Failed to read GeoJSON: {exc}"}

    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        return {"error": "File is not a valid GeoJSON FeatureCollection"}

    features = data["features"]
    geom_types: dict[str, int] = {}
    for feat in features:
        gt = feat.get("geometry", {}).get("type", "unknown")
        geom_types[gt] = geom_types.get(gt, 0) + 1

    # Property schema from first feature
    props_schema: dict[str, str] = {}
    if features:
        sample_props = features[0].get("properties") or {}
        for k, v in sample_props.items():
            props_schema[k] = type(v).__name__

    # Bounds
    bounds = _compute_bounds(features)

    # Layer type breakdown (if layer_type property exists)
    layer_types: dict[str, int] = {}
    for feat in features:
        lt = (feat.get("properties") or {}).get("layer_type")
        if lt:
            layer_types[str(lt)] = layer_types.get(str(lt), 0) + 1

    summary: dict[str, Any] = {
        "file": str(path.name),
        "feature_count": len(features),
        "geometry_types": geom_types,
        "property_schema": props_schema,
        "bounds": bounds,
    }
    if layer_types:
        summary["layer_types"] = layer_types

    # Include full features for small datasets
    if len(features) <= 500:
        summary["features"] = features

    return {"summary": summary, "geojson": data}


# -- Query ------------------------------------------------------------------

_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


def query_geojson(
    geojson: dict,
    property_name: str,
    op: str,
    value: Any,
) -> dict[str, Any]:
    """Filter features by a property value.

    Parameters
    ----------
    geojson : dict
        A GeoJSON FeatureCollection (from the object store).
    property_name : str
        The property name to filter on.
    op : str
        Comparison operator: ==, !=, >, <, >=, <=, contains.
    value : Any
        The value to compare against.

    Returns a dict with filtered GeoJSON and match count.
    """
    features = geojson.get("features", [])

    if op == "contains":
        matched = [
            f for f in features
            if str(value).lower() in str((f.get("properties") or {}).get(property_name, "")).lower()
        ]
    elif op in _OPS:
        cmp = _OPS[op]
        matched = []
        for f in features:
            prop_val = (f.get("properties") or {}).get(property_name)
            if prop_val is None:
                continue
            try:
                if cmp(prop_val, value):
                    matched.append(f)
            except TypeError:
                # Type mismatch (e.g., comparing str to int) -- try numeric coercion
                try:
                    if cmp(float(prop_val), float(value)):
                        matched.append(f)
                except (ValueError, TypeError):
                    continue
    else:
        return {"error": f"Unknown operator '{op}'. Supported: ==, !=, >, <, >=, <=, contains"}

    filtered = {"type": "FeatureCollection", "features": matched}

    result: dict[str, Any] = {
        "match_count": len(matched),
        "total_count": len(features),
        "query": {"property": property_name, "operator": op, "value": value},
    }
    if len(matched) <= 500:
        result["features"] = matched
    return {"result": result, "geojson": filtered}


# -- Summarize --------------------------------------------------------------

def summarize_geojson_property(
    geojson: dict,
    property_name: str,
) -> dict[str, Any]:
    """Compute statistics for a single property across all features.

    For numeric properties: min, max, mean, std, median.
    For string/categorical: unique value counts.
    """
    features = geojson.get("features", [])
    values = [
        (f.get("properties") or {}).get(property_name)
        for f in features
        if (f.get("properties") or {}).get(property_name) is not None
    ]

    if not values:
        return {
            "property": property_name,
            "error": f"No features have property '{property_name}'",
        }

    # Try numeric
    numeric: list[float] = []
    for v in values:
        try:
            numeric.append(float(v))
        except (ValueError, TypeError):
            break
    else:
        # All values are numeric
        arr = np.array(numeric)
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return {"property": property_name, "error": "All numeric values are NaN/inf"}
        return {
            "property": property_name,
            "type": "numeric",
            "count": len(valid),
            "min": round(float(np.min(valid)), 4),
            "max": round(float(np.max(valid)), 4),
            "mean": round(float(np.mean(valid)), 4),
            "std": round(float(np.std(valid)), 4),
            "median": round(float(np.median(valid)), 4),
        }

    # Categorical
    counts: dict[str, int] = {}
    for v in values:
        key = str(v)
        counts[key] = counts.get(key, 0) + 1

    # Sort by count descending
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    return {
        "property": property_name,
        "type": "categorical",
        "count": len(values),
        "unique_values": len(sorted_counts),
        "value_counts": sorted_counts,
    }


# -- Helpers ----------------------------------------------------------------

def _compute_bounds(features: list[dict]) -> dict[str, float] | None:
    """Compute bounding box from all feature geometries."""
    min_lng = float("inf")
    min_lat = float("inf")
    max_lng = float("-inf")
    max_lat = float("-inf")

    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        for lng, lat in _extract_coords(geom):
            if lng < min_lng:
                min_lng = lng
            if lat < min_lat:
                min_lat = lat
            if lng > max_lng:
                max_lng = lng
            if lat > max_lat:
                max_lat = lat

    if min_lng == float("inf"):
        return None
    return {
        "west": round(min_lng, 6),
        "south": round(min_lat, 6),
        "east": round(max_lng, 6),
        "north": round(max_lat, 6),
    }


def _extract_coords(geometry: dict) -> list[tuple[float, float]]:
    """Recursively extract (lng, lat) pairs from a GeoJSON geometry."""
    gtype = geometry.get("type", "")
    coords = geometry.get("coordinates", [])

    if gtype == "Point":
        return [(coords[0], coords[1])]
    if gtype in ("MultiPoint", "LineString"):
        return [(c[0], c[1]) for c in coords]
    if gtype in ("MultiLineString", "Polygon"):
        return [(c[0], c[1]) for ring in coords for c in ring]
    if gtype == "MultiPolygon":
        return [(c[0], c[1]) for poly in coords for ring in poly for c in ring]
    if gtype == "GeometryCollection":
        result = []
        for g in geometry.get("geometries", []):
            result.extend(_extract_coords(g))
        return result
    return []
