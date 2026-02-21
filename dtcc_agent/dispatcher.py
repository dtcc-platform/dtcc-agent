"""Dispatcher: resolves parameters, calls functions, stores results.

Handles the plumbing between the MCP tool layer and the function registry:
  1. Looks up the operation in the registry
  2. Resolves object-ref parameters from the ObjectStore
  3. Converts bounds lists → Bounds objects, strings → enums
  4. Calls the function
  5. Stores the result in the ObjectStore
  6. Returns a serialized summary
"""

from __future__ import annotations

import inspect
import logging
from copy import deepcopy
from typing import Any

from .object_store import ObjectStore
from .registry import get_operation, OperationInfo
from .serializers import serialize
from .disk_cache import (
    DiskCache, CACHE_ALLOWLIST,
    content_fingerprint, canonical_params_hash,
)
from .crop import crop_to_bounds

logger = logging.getLogger(__name__)


def _resolve_bounds(value: Any) -> Any:
    """Convert a [minx, miny, maxx, maxy] list to a dtcc Bounds object."""
    if isinstance(value, (list, tuple)) and len(value) in (4, 6):
        try:
            from dtcc_core.model.geometry.bounds import Bounds
            if len(value) == 4:
                return Bounds(xmin=value[0], ymin=value[1],
                            xmax=value[2], ymax=value[3])
            else:
                return Bounds(xmin=value[0], ymin=value[1], zmin=value[2],
                            xmax=value[3], ymax=value[4], zmax=value[5])
        except ImportError:
            pass
    return value


def _resolve_enum(value: Any, type_hint: str) -> Any:
    """Try to convert a string value to a known enum type."""
    if not isinstance(value, str):
        return value
    # GeometryType enum
    if "GeometryType" in type_hint:
        try:
            from dtcc_core.model.object.object import GeometryType
            return GeometryType[value.upper()]
        except (ImportError, KeyError):
            pass
    return value


def _should_copy(obj: Any) -> bool:
    """Check if an object should be deep-copied before passing to a function."""
    type_name = type(obj).__name__
    return type_name in ("PointCloud", "Mesh", "VolumeMesh", "Raster",
                         "City", "Terrain", "Surface", "MultiSurface")


def run_operation(
    name: str,
    params: dict[str, Any] | None,
    store: ObjectStore,
    label: str = "",
    cache: DiskCache | None = None,
) -> dict[str, Any]:
    """Execute a registered operation and store the result.

    Parameters
    ----------
    name : str
        Fully qualified operation name (e.g. "builder.build_terrain_raster").
    params : dict
        Parameters to pass to the function. Object-ref params should be
        string IDs that will be resolved from the store.
    store : ObjectStore
        The object store for resolving inputs and storing outputs.
    label : str
        Optional human-readable label for the stored result.
    cache : DiskCache | None
        Optional persistent disk cache. When provided, the dispatcher
        checks the cache before running the operation and populates it
        after a successful run.

    Returns
    -------
    dict with keys: operation, result_id (or result_ids for tuples),
    summary, and label.
    """
    params = params or {}
    try:
        op = get_operation(name)
    except KeyError as exc:
        return {"error": str(exc)}

    # --- Disk cache check ---
    if cache and name in CACHE_ALLOWLIST:
        hit = _check_cache(name, op.category, params, store, cache)
        if hit is not None:
            hit["cache_hit"] = True
            return hit

    # Datasets are special: called as ds(bounds=..., **params)
    if op.category == "datasets":
        result = _run_dataset(op, params, store, label)
    else:
        result = _run_function(op, params, store, label)

    # --- Store in disk cache on success ---
    if cache and name in CACHE_ALLOWLIST and "error" not in result:
        _populate_cache(name, op.category, params, result, store, cache)

    return result


def _run_function(
    op: OperationInfo,
    params: dict[str, Any],
    store: ObjectStore,
    label: str,
) -> dict[str, Any]:
    """Execute a regular function (builder, io, reproject, etc.)."""
    func = op._callable
    resolved = {}
    missing = []

    for pinfo in op.params:
        pname = pinfo.name
        if pname not in params:
            if pinfo.default is not inspect.Parameter.empty:
                continue  # will use the function's default
            missing.append(pname)
            continue

        value = params[pname]

        # 1. Object-ref resolution
        if pinfo.is_object_param and isinstance(value, str):
            try:
                obj = store.get(value)
                if _should_copy(obj):
                    obj = deepcopy(obj)
                value = obj
            except KeyError:
                pass  # not an object ID, pass through as-is

        # 2. Bounds resolution
        if "Bounds" in pinfo.type_hint or pname == "bounds":
            value = _resolve_bounds(value)

        # 3. Enum resolution
        if isinstance(value, str):
            value = _resolve_enum(value, pinfo.type_hint)

        resolved[pname] = value

    if missing:
        return {"error": f"Operation '{op.name}' missing required parameters: {', '.join(missing)}"}

    # Call the function
    try:
        result = func(**resolved)
    except Exception as exc:
        return {"error": f"Operation '{op.name}' failed: {exc}"}

    return _store_and_summarize(result, op.name, store, label)


def _run_dataset(
    op: OperationInfo,
    params: dict[str, Any],
    store: ObjectStore,
    label: str,
) -> dict[str, Any]:
    """Execute a dataset operation."""
    ds = op._callable

    # Ensure bounds is present and is a list
    kwargs = dict(params)
    bounds = kwargs.get("bounds")
    if bounds is not None:
        # Datasets expect bounds as a list, not a Bounds object
        if hasattr(bounds, "tuple"):
            kwargs["bounds"] = list(bounds.tuple)
        elif not isinstance(bounds, (list, tuple)):
            kwargs["bounds"] = list(bounds)

    try:
        result = ds(**kwargs)
    except Exception as exc:
        return {"error": f"Dataset '{op.name}' failed: {exc}"}

    return _store_and_summarize(result, op.name, store, label)


def _store_and_summarize(
    result: Any,
    op_name: str,
    store: ObjectStore,
    label: str,
) -> dict[str, Any]:
    """Store a result and return a serialized summary."""
    # Handle tuples: store each element separately
    if isinstance(result, tuple):
        ids = []
        summaries = []
        for i, item in enumerate(result):
            item_label = f"{label}_part{i}" if label else f"part{i}"
            obj_id = store.store(item, source_op=op_name, label=item_label)
            ids.append(obj_id)
            summaries.append(serialize(item))
        return {
            "operation": op_name,
            "result_ids": ids,
            "label": label,
            "summary": summaries,
        }

    # Handle lists of dtcc objects
    if isinstance(result, list) and len(result) > 0:
        first_type = type(result[0]).__name__
        if first_type in ("Building", "Tree", "Surface"):
            obj_id = store.store(result, source_op=op_name, label=label)
            return {
                "operation": op_name,
                "result_id": obj_id,
                "label": label,
                "summary": serialize(result),
            }

    # Primitives / dicts — don't store, just return
    if isinstance(result, (int, float, str, bool, type(None), dict)):
        return {
            "operation": op_name,
            "label": label,
            "summary": serialize(result),
        }

    # Single object — store it
    obj_id = store.store(result, source_op=op_name, label=label)
    summary = serialize(result)

    return {
        "operation": op_name,
        "result_id": obj_id,
        "label": label,
        "summary": summary,
    }


# -- Disk cache helpers ------------------------------------------------------

def _check_cache(
    name: str,
    category: str,
    params: dict[str, Any],
    store: ObjectStore,
    cache: DiskCache,
) -> dict[str, Any] | None:
    """Check disk cache. Returns dispatcher-format response or None."""
    try:
        if category == "datasets":
            return _check_cache_dataset(name, params, store, cache)
        else:
            return _check_cache_builder(name, params, store, cache)
    except Exception:
        logger.warning("Disk cache lookup failed for %s", name, exc_info=True)
        return None


def _check_cache_dataset(
    name: str,
    params: dict[str, Any],
    store: ObjectStore,
    cache: DiskCache,
) -> dict[str, Any] | None:
    """Check disk cache for a dataset operation."""
    bounds = params.get("bounds")
    if not bounds:
        return None
    source = params.get("source", "LM")
    non_bounds = {k: v for k, v in params.items() if k != "bounds"}
    ph = canonical_params_hash(name, non_bounds)
    result = cache.dataset_lookup(name, source, ph, bounds)
    if result is None:
        return None
    cache_id, cached_bounds = result
    obj = cache.load(cache_id)
    # Crop if cached bounds are larger than requested
    if cached_bounds != list(bounds):
        obj = crop_to_bounds(obj, list(bounds))
    obj_id = store.store(obj, source_op=name, label="(cached)")
    return {
        "operation": name,
        "result_id": obj_id,
        "label": "(cached)",
        "summary": serialize(obj),
    }


def _check_cache_builder(
    name: str,
    params: dict[str, Any],
    store: ObjectStore,
    cache: DiskCache,
) -> dict[str, Any] | None:
    """Check disk cache for a builder operation."""
    fingerprints = _compute_fingerprints(params, store)
    ph = canonical_params_hash(name, params, fingerprints)
    cache_id = cache.builder_lookup(name, ph)
    if cache_id is None:
        return None
    obj = cache.load(cache_id)
    obj_id = store.store(obj, source_op=name, label="(cached)")
    return {
        "operation": name,
        "result_id": obj_id,
        "label": "(cached)",
        "summary": serialize(obj),
    }


def _compute_fingerprints(
    params: dict[str, Any],
    store: ObjectStore,
) -> dict[str, str]:
    """Compute content fingerprints for object-ref params."""
    fingerprints = {}
    for key, value in params.items():
        if isinstance(value, str) and value in store:
            meta = None
            for entry in store.list(limit=500):
                if entry["id"] == value:
                    meta = entry
                    break
            if meta:
                fingerprints[key] = content_fingerprint(meta)
    return fingerprints


def _populate_cache(
    name: str,
    category: str,
    params: dict[str, Any],
    result: dict[str, Any],
    store: ObjectStore,
    cache: DiskCache,
) -> None:
    """Store a successful operation result in the disk cache."""
    result_id = result.get("result_id")
    if not result_id:
        return  # tuple/primitive results — skip for v1
    try:
        obj = store.get(result_id)
    except KeyError:
        return

    if category == "datasets":
        bounds = params.get("bounds")
        source = params.get("source", "LM")
        non_bounds = {k: v for k, v in params.items() if k != "bounds"}
        ph = canonical_params_hash(name, non_bounds)
        bounds_list = list(bounds) if bounds else None
    else:
        fingerprints = _compute_fingerprints(params, store)
        ph = canonical_params_hash(name, params, fingerprints)
        bounds_list = None
        source = None

    try:
        cache.store(
            obj=obj,
            operation=name,
            category=category,
            params_hash=ph,
            bounds=bounds_list,
            source=source,
            object_type=type(obj).__name__,
        )
    except Exception:
        logger.warning("Failed to store %s result in disk cache", name,
                        exc_info=True)
