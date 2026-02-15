"""Type-aware JSON serializers for dtcc-core objects.

Converts dtcc-core objects into LLM-friendly JSON summaries.
Never dumps raw arrays — only statistics and metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import summarize_field


def _bounds_dict(obj: Any) -> dict | None:
    """Extract bounds as a simple dict, if available."""
    bounds = getattr(obj, "bounds", None)
    if bounds is None:
        return None
    try:
        return {
            "xmin": round(float(bounds.xmin), 2),
            "ymin": round(float(bounds.ymin), 2),
            "xmax": round(float(bounds.xmax), 2),
            "ymax": round(float(bounds.ymax), 2),
            "zmin": round(float(bounds.zmin), 2),
            "zmax": round(float(bounds.zmax), 2),
        }
    except Exception:
        return None


def _classification_stats(classification: np.ndarray) -> dict:
    """Summarize point cloud classification codes."""
    if len(classification) == 0:
        return {}
    unique, counts = np.unique(classification, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


# ── Per-type serializers ──────────────────────────────────────────────

def _serialize_pointcloud(obj: Any) -> dict:
    return {
        "type": "PointCloud",
        "num_points": len(obj.points) if hasattr(obj, "points") else 0,
        "bounds": _bounds_dict(obj),
        "classification_counts": _classification_stats(
            getattr(obj, "classification", np.empty(0))
        ),
        "z_stats": summarize_field(
            obj.points[:, 2], "elevation"
        ) if hasattr(obj, "points") and len(obj.points) > 0 else None,
    }


def _serialize_mesh(obj: Any) -> dict:
    num_vertices = getattr(obj, "num_vertices", 0)
    num_faces = getattr(obj, "num_faces", 0)
    markers = getattr(obj, "markers", np.empty(0))
    marker_info = {}
    if isinstance(markers, np.ndarray) and len(markers) > 0:
        unique = np.unique(markers)
        marker_info = {"unique_markers": [int(m) for m in unique[:20]]}
    return {
        "type": "Mesh",
        "num_vertices": num_vertices,
        "num_faces": num_faces,
        "bounds": _bounds_dict(obj),
        "markers": marker_info,
    }


def _serialize_volume_mesh(obj: Any) -> dict:
    return {
        "type": "VolumeMesh",
        "num_vertices": getattr(obj, "num_vertices", 0),
        "num_cells": getattr(obj, "num_cells", 0),
        "bounds": _bounds_dict(obj),
    }


def _serialize_raster(obj: Any) -> dict:
    data = getattr(obj, "data", None)
    shape = data.shape if data is not None else ()
    cell_size = getattr(obj, "cell_size", None)
    result = {
        "type": "Raster",
        "shape": list(shape),
        "bounds": _bounds_dict(obj),
    }
    if cell_size is not None:
        try:
            result["cell_size"] = [round(abs(float(c)), 4) for c in cell_size]
        except Exception:
            pass
    if data is not None and data.size > 0:
        result["value_stats"] = summarize_field(data.ravel(), "value")
    return result


def _serialize_city(obj: Any) -> dict:
    buildings = getattr(obj, "buildings", [])
    has_terrain = obj.has_terrain() if hasattr(obj, "has_terrain") else False
    heights = [b.height for b in buildings if getattr(b, "height", None) and b.height > 0]
    height_stats = summarize_field(np.array(heights), "building_height") if heights else None
    return {
        "type": "City",
        "num_buildings": len(buildings),
        "has_terrain": has_terrain,
        "num_trees": len(getattr(obj, "trees", [])),
        "bounds": _bounds_dict(obj),
        "height_stats": height_stats,
    }


def _serialize_building_list(obj: list) -> dict:
    heights = [b.height for b in obj if getattr(b, "height", None) and b.height > 0]
    lods = set()
    for b in obj[:20]:
        for attr in ("lod0", "lod1", "lod2", "lod3"):
            if getattr(b, attr, None) is not None:
                lods.add(attr.upper())
    return {
        "type": "list[Building]",
        "count": len(obj),
        "height_stats": summarize_field(np.array(heights), "height") if heights else None,
        "available_lods": sorted(lods),
    }


def _serialize_tree_list(obj: list) -> dict:
    heights = [t.height for t in obj if getattr(t, "height", 0) > 0]
    radii = [t.crown_radius for t in obj if getattr(t, "crown_radius", 0) > 0]
    return {
        "type": "list[Tree]",
        "count": len(obj),
        "height_stats": summarize_field(np.array(heights), "height") if heights else None,
        "crown_radius_stats": summarize_field(np.array(radii), "crown_radius") if radii else None,
    }


def _serialize_terrain(obj: Any) -> dict:
    has_mesh = getattr(obj, "mesh", None) is not None
    has_raster = getattr(obj, "raster", None) is not None
    result: dict[str, Any] = {
        "type": "Terrain",
        "has_mesh": has_mesh,
        "has_raster": has_raster,
        "bounds": _bounds_dict(obj),
    }
    raster = getattr(obj, "raster", None)
    if raster is not None:
        data = getattr(raster, "data", None)
        if data is not None and data.size > 0:
            result["elevation_stats"] = summarize_field(data.ravel(), "elevation")
    return result


def _serialize_road_network(obj: Any) -> dict:
    edges = getattr(obj, "edges", [])
    return {
        "type": "RoadNetwork",
        "num_edges": len(edges) if hasattr(edges, "__len__") else 0,
        "bounds": _bounds_dict(obj),
    }


def _serialize_dolfinx_function(obj: Any) -> dict:
    arr = obj.x.array
    return {
        "type": "dolfinx.Function",
        "num_dofs": len(arr),
        "value_stats": summarize_field(arr, "value"),
    }


# ── Dispatch table ────────────────────────────────────────────────────

def _get_type_name(obj: Any) -> str:
    """Return the qualified type name for dispatch."""
    return type(obj).__qualname__


# Lazy-built dispatch to avoid import-time dependency on dtcc_core
_DISPATCH: dict[str, Any] | None = None


def _build_dispatch() -> dict[str, Any]:
    """Build type-name → serializer mapping."""
    table: dict[str, Any] = {}
    try:
        from dtcc_core.model.geometry.pointcloud import PointCloud
        table[PointCloud] = _serialize_pointcloud
    except ImportError:
        pass
    try:
        from dtcc_core.model.geometry.mesh import Mesh, VolumeMesh
        table[Mesh] = _serialize_mesh
        table[VolumeMesh] = _serialize_volume_mesh
    except ImportError:
        pass
    try:
        from dtcc_core.model.values.raster import Raster
        table[Raster] = _serialize_raster
    except ImportError:
        pass
    try:
        from dtcc_core.model.object.city import City
        table[City] = _serialize_city
    except ImportError:
        pass
    try:
        from dtcc_core.model.object.terrain import Terrain
        table[Terrain] = _serialize_terrain
    except ImportError:
        pass
    try:
        from dtcc_core.model.object.road_network import RoadNetwork
        table[RoadNetwork] = _serialize_road_network
    except ImportError:
        pass
    return table


def _get_dispatch() -> dict:
    global _DISPATCH
    if _DISPATCH is None:
        _DISPATCH = _build_dispatch()
    return _DISPATCH


def serialize(obj: Any) -> dict[str, Any]:
    """Convert a dtcc-core object into an LLM-friendly JSON summary.

    Parameters
    ----------
    obj : Any
        A dtcc-core object, list, tuple, dict, or primitive.

    Returns
    -------
    dict
        JSON-serializable summary (never raw arrays).
    """
    # dolfinx.Function
    if hasattr(obj, "x") and hasattr(getattr(obj, "x", None), "array"):
        return _serialize_dolfinx_function(obj)

    # Exact type match in dispatch table
    dispatch = _get_dispatch()
    obj_type = type(obj)
    if obj_type in dispatch:
        return dispatch[obj_type](obj)

    # Lists — check if homogeneous Building or Tree list
    if isinstance(obj, list) and len(obj) > 0:
        first_type = type(obj[0]).__name__
        if first_type == "Building":
            return _serialize_building_list(obj)
        if first_type == "Tree":
            return _serialize_tree_list(obj)
        # Generic list: serialize each element
        return {
            "type": f"list[{first_type}]",
            "count": len(obj),
            "elements": [serialize(item) for item in obj[:10]],
        }

    # Tuples — serialize each element with index
    if isinstance(obj, tuple):
        return {
            "type": "tuple",
            "count": len(obj),
            "elements": [serialize(item) for item in obj],
        }

    # Numpy arrays — summarize
    if isinstance(obj, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "stats": summarize_field(obj.ravel(), "value") if obj.size > 0 else None,
        }

    # Primitives and dicts — pass through
    if isinstance(obj, (int, float, str, bool, type(None))):
        return {"type": type(obj).__name__, "value": obj}

    if isinstance(obj, dict):
        return {"type": "dict", "keys": list(obj.keys())[:20], "num_keys": len(obj)}

    # Fallback
    return {"type": type(obj).__name__, "repr": str(obj)[:200]}
