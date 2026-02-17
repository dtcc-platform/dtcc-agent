"""MCP server for dtcc-agent.

Exposes urban digital twin simulations as MCP tools. All computation
happens in-process — dtcc_core and dtcc_sim are called directly,
no HTTP backend needed.

Run with: python -m dtcc_agent
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP

from .geocode import geocode as _geocode
from .analysis import summarize_field, compare_fields
from .object_store import ObjectStore
from .serializers import serialize

mcp = FastMCP("dtcc-agent")

# In-memory store for simulation results so the agent can refer back
# to previous runs (e.g. for comparison). Keyed by run_id.
_results: dict[str, dict[str, Any]] = {}

# Shared object store for dtcc-core objects (PointCloud, Mesh, Raster, etc.)
_object_store = ObjectStore()


# -- Helpers -----------------------------------------------------------------

def _fmt(data: Any) -> str:
    """Pretty-print a dict/list as JSON for the LLM."""
    if isinstance(data, str):
        return data
    return json.dumps(data, indent=2, default=str)


def _store_result(name: str, bounds: list[float], parameters: dict, result: Any) -> str:
    """Store a simulation result and return a run_id."""
    run_id = str(uuid.uuid4())[:8]
    _results[run_id] = {
        "simulation": name,
        "bounds": bounds,
        "parameters": parameters,
        "result": result,
        "timestamp": time.time(),
    }
    # Also store in the object store so simulation results can feed pipelines
    _object_store.store(result, source_op=f"simulation.{name}", label=run_id)
    return run_id


# -- Geocoding ---------------------------------------------------------------

@mcp.tool()
def geocode(place_name: str, radius: float = 250.0) -> str:
    """Convert a place name to an EPSG:3006 bounding box.

    Use this to translate human-readable locations like "Lindholmen"
    or "Haga, Gothenburg" into the coordinate bounds required by
    simulations and building queries.

    Args:
        place_name: Free-text place name (e.g. "Lindholmen",
            "Chalmers", "Haga, Gothenburg").
        radius: Half-width in meters for the bounding box when
            the geocoder returns a point. Default 250m.

    Returns a JSON object with bounds [minx, miny, maxx, maxy] in
    EPSG:3006, center coordinates, and source attribution.
    """
    try:
        result = _geocode(place_name, radius=radius)
    except (RuntimeError, ValueError) as exc:
        return _fmt({"error": str(exc)})
    return _fmt(result)


# -- Urban context -----------------------------------------------------------

@mcp.tool()
def get_buildings(
    bounds: list[float],
    source: str = "LM",
    max_buildings: int = 100,
) -> str:
    """Inspect buildings within a bounding box.

    Returns building count, per-building details (height, footprint
    area), and aggregate height statistics. Use this to understand
    the urban context before running simulations.

    Args:
        bounds: Bounding box [minx, miny, maxx, maxy] in EPSG:3006.
            Use geocode() to get bounds from a place name.
        source: Data source — "LM" (Lantmäteriet, Sweden) or "OSM"
            (OpenStreetMap, global).
        max_buildings: Maximum number of per-building details to
            return (default 100). Set lower for large areas.

    Returns a JSON object with building list and height statistics.
    """
    from .runner import get_buildings as _get_buildings

    try:
        result = _get_buildings(
            bounds=bounds,
            source=source,
            max_buildings=max_buildings,
        )
    except Exception as exc:
        return _fmt({"error": f"Failed to fetch buildings: {exc}"})
    return _fmt(result)


# -- Simulation discovery ----------------------------------------------------

@mcp.tool()
def list_simulations() -> str:
    """List all available simulation types.

    Returns the name and description of each registered simulation.
    Use get_simulation_schema() to see the parameters for a specific
    simulation before running it.
    """
    from .runner import list_simulations as _list

    return _fmt(_list())


@mcp.tool()
def get_simulation_schema(simulation_name: str) -> str:
    """Get the parameter schema for a simulation.

    Returns a JSON Schema describing all accepted parameters including
    their types, defaults, and descriptions. Use this to understand
    what parameters to pass to run_simulation().

    Args:
        simulation_name: Name of the simulation (e.g.
            "urban_heat_simulation", "air_quality_field").
    """
    from .runner import get_schema

    try:
        schema = get_schema(simulation_name)
    except KeyError:
        return _fmt({"error": f"Unknown simulation: {simulation_name}"})
    return _fmt(schema)


# -- Run simulation ----------------------------------------------------------

@mcp.tool()
def run_simulation(
    simulation_name: str,
    bounds: list[float],
    parameters: dict[str, Any] | None = None,
    label: str | None = None,
) -> str:
    """Run a simulation and return summary statistics of the result.

    This calls the simulation directly in-process. For urban_heat_simulation,
    expect ~1–5 minutes depending on domain size and mesh resolution.

    The result is stored in memory with a run_id so you can reference it
    later in compare_scenarios().

    Args:
        simulation_name: Which simulation to run (e.g. "urban_heat_simulation").
        bounds: Bounding box [minx, miny, maxx, maxy] in EPSG:3006.
        parameters: Simulation-specific parameters (see get_simulation_schema()).
            Omit to use defaults.
        label: Optional human-readable label for this run (e.g. "baseline",
            "heatwave"). Used in comparison output.

    Returns a JSON object with run_id, simulation metadata, and summary
    statistics (min, max, mean, std, median, percentiles).
    """
    from .runner import run as _run

    params = parameters or {}

    try:
        result = _run(simulation_name, bounds, params)
    except Exception as exc:
        return _fmt({"error": f"Simulation failed: {exc}"})

    # Extract values — dolfinx Function has .x.array
    if hasattr(result, "x") and hasattr(result.x, "array"):
        values = result.x.array
        field_name = _infer_field_name(simulation_name)
    else:
        return _fmt({
            "error": (
                f"Unexpected result type: {type(result).__name__}. "
                "Expected dolfinx Function with .x.array attribute."
            )
        })

    run_id = _store_result(simulation_name, bounds, params, result)
    summary = summarize_field(values, field_name)

    return _fmt({
        "run_id": run_id,
        "label": label or run_id,
        "simulation": simulation_name,
        "bounds": bounds,
        "parameters_used": params,
        "summary": summary,
    })


@mcp.tool()
def compare_scenarios(
    simulation_name: str,
    bounds: list[float],
    scenario_a_parameters: dict[str, Any],
    scenario_b_parameters: dict[str, Any],
    label_a: str = "scenario_a",
    label_b: str = "scenario_b",
) -> str:
    """Run two simulation scenarios on the same area and compare results.

    Runs both simulations sequentially on the same bounding box (and
    therefore the same mesh), then computes per-DOF differences and
    summary statistics for each scenario and the difference.

    Typical use: compare baseline vs heatwave, or current vs mitigated.

    Args:
        simulation_name: Which simulation to run for both scenarios.
        bounds: Bounding box [minx, miny, maxx, maxy] in EPSG:3006.
        scenario_a_parameters: Parameters for the first scenario.
        scenario_b_parameters: Parameters for the second scenario.
        label_a: Human-readable label for scenario A (e.g. "baseline").
        label_b: Human-readable label for scenario B (e.g. "heatwave").

    Returns a JSON object with summary stats for each scenario and
    the difference (B minus A).
    """
    from .runner import run as _run

    # Run scenario A
    try:
        result_a = _run(simulation_name, bounds, scenario_a_parameters)
    except Exception as exc:
        return _fmt({"error": f"Scenario A ({label_a}) failed: {exc}"})

    # Run scenario B
    try:
        result_b = _run(simulation_name, bounds, scenario_b_parameters)
    except Exception as exc:
        return _fmt({"error": f"Scenario B ({label_b}) failed: {exc}"})

    # Extract values
    if not (hasattr(result_a, "x") and hasattr(result_b, "x")):
        return _fmt({
            "error": "Unexpected result type — expected dolfinx Function."
        })

    values_a = result_a.x.array
    values_b = result_b.x.array
    field_name = _infer_field_name(simulation_name)

    # Store both
    run_id_a = _store_result(simulation_name, bounds, scenario_a_parameters, result_a)
    run_id_b = _store_result(simulation_name, bounds, scenario_b_parameters, result_b)

    comparison = compare_fields(
        values_a, values_b,
        label_a=label_a,
        label_b=label_b,
        field_name=field_name,
    )

    return _fmt({
        "simulation": simulation_name,
        "bounds": bounds,
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "parameters_a": scenario_a_parameters,
        "parameters_b": scenario_b_parameters,
        **comparison,
    })


# -- Utilities ---------------------------------------------------------------

@mcp.tool()
def list_past_runs(limit: int = 10) -> str:
    """List recent simulation runs stored in memory.

    Args:
        limit: Maximum number of runs to return (most recent first).

    Returns a JSON array of past runs with run_id, simulation name,
    bounds, parameters, and summary stats.
    """
    runs = sorted(_results.items(), key=lambda kv: kv[1]["timestamp"], reverse=True)
    output = []
    for run_id, info in runs[:limit]:
        entry = {
            "run_id": run_id,
            "simulation": info["simulation"],
            "bounds": info["bounds"],
            "parameters": info["parameters"],
        }
        # Add summary if we can extract values
        result = info["result"]
        if hasattr(result, "x") and hasattr(result.x, "array"):
            field_name = _infer_field_name(info["simulation"])
            entry["summary"] = summarize_field(result.x.array, field_name)
        output.append(entry)
    return _fmt(output)


@mcp.tool()
def get_run_summary(run_id: str) -> str:
    """Get summary statistics for a previous simulation run.

    Args:
        run_id: The run_id returned by run_simulation() or compare_scenarios().

    Returns the full summary statistics for that run.
    """
    if run_id not in _results:
        return _fmt({"error": f"Run {run_id} not found. Use list_past_runs() to see available runs."})

    info = _results[run_id]
    result = info["result"]

    if not (hasattr(result, "x") and hasattr(result.x, "array")):
        return _fmt({"error": "Result does not have extractable field values."})

    field_name = _infer_field_name(info["simulation"])
    summary = summarize_field(result.x.array, field_name)

    return _fmt({
        "run_id": run_id,
        "simulation": info["simulation"],
        "bounds": info["bounds"],
        "parameters": info["parameters"],
        "summary": summary,
    })


# -- Dynamic dispatch tools --------------------------------------------------

@mcp.tool()
def list_operations(
    category: str | None = None,
    search: str | None = None,
) -> str:
    """List all available dtcc-core operations.

    Browse the full catalog of operations including builder functions
    (terrain, buildings, pointcloud, mesh), IO (load/save), datasets
    (point_cloud, buildings, terrain), and reproject utilities.

    Use describe_operation() to see the full parameter schema for any
    operation, then run_operation() to execute it.

    Args:
        category: Filter by category. Options: "builder", "io",
            "datasets", "reproject". Omit to list all.
        search: Free-text search across operation names, descriptions,
            and tags. E.g. "terrain", "pointcloud", "mesh".

    Returns a JSON array of operations with name, category, and description.
    """
    from .registry import list_operations as _list_ops

    try:
        ops = _list_ops(category=category, search=search)
    except Exception as exc:
        return _fmt({"error": f"Failed to list operations: {exc}"})
    return _fmt(ops)


@mcp.tool()
def describe_operation(name: str) -> str:
    """Get the full parameter schema for a dtcc-core operation.

    Shows all parameters including their types, defaults, and whether
    they accept object references (from previous run_operation results).

    Args:
        name: Operation name as returned by list_operations()
            (e.g. "builder.build_terrain_raster", "datasets.point_cloud").

    Returns a JSON object with the operation's full schema.
    """
    from .registry import get_operation as _get_op

    try:
        op = _get_op(name)
    except KeyError as exc:
        return _fmt({"error": str(exc)})
    return _fmt(op.to_dict())


@mcp.tool()
def run_operation(
    name: str,
    params: dict[str, Any] | None = None,
    label: str | None = None,
) -> str:
    """Execute a dtcc-core operation and store the result.

    Runs any registered operation (builder function, dataset download,
    IO operation, etc.). Results are stored in the object store and can
    be referenced by ID in subsequent operations.

    For parameters that accept dtcc-core objects (marked is_object_ref
    in describe_operation), pass the object ID string from a previous
    run_operation result.

    Args:
        name: Operation name (e.g. "builder.build_terrain_raster",
            "datasets.point_cloud").
        params: Parameters as a JSON object. Object-reference params
            should use the ID string from a previous result.
        label: Optional human-readable label for this result.

    Returns a JSON object with result_id(s), operation name, and a
    summary of the result (statistics, counts — never raw data).
    """
    from .dispatcher import run_operation as _dispatch

    try:
        result = _dispatch(
            name=name,
            params=params,
            store=_object_store,
            label=label or "",
        )
    except Exception as exc:
        return _fmt({"error": f"Operation failed: {exc}"})
    return _fmt(result)


@mcp.tool()
def list_objects(limit: int = 20) -> str:
    """List objects stored in memory from previous operations.

    Shows all dtcc-core objects (PointCloud, Mesh, Raster, etc.) that
    have been created by run_operation(). Use object IDs to pass
    results between operations in multi-step pipelines.

    Args:
        limit: Maximum number of objects to return (most recent first).

    Returns a JSON array of stored objects with id, type, source
    operation, label, and memory size.
    """
    objects = _object_store.list(limit=limit)
    return _fmt({
        "num_objects": len(_object_store),
        "total_memory_mb": round(_object_store.total_bytes / (1024 * 1024), 2),
        "objects": objects,
    })


@mcp.tool()
def inspect_object(object_id: str) -> str:
    """Get a detailed summary of a stored object.

    Returns type-specific statistics: point counts, mesh info,
    raster dimensions, elevation stats, etc. Never returns raw
    array data — only human-readable summaries.

    Args:
        object_id: The object ID from run_operation() or list_objects().

    Returns a JSON summary of the object's contents.
    """
    try:
        obj = _object_store.get(object_id)
    except KeyError:
        return _fmt({"error": f"Object '{object_id}' not found. Use list_objects() to see available objects."})

    summary = serialize(obj)
    summary["object_id"] = object_id
    return _fmt(summary)


# -- Visualization -----------------------------------------------------------

@mcp.tool()
def render_object(
    object_id: str,
    width: int = 1200,
    height: int = 800,
) -> str:
    """Render a stored object as a PNG screenshot.

    Creates an offscreen 3D visualization of any dtcc-core geometry object
    (Mesh, PointCloud, City, Building, Raster, etc.) and saves it as a PNG.

    Supported types: Mesh, PointCloud, City, Raster, Building, Surface,
    MultiSurface, VolumeMesh, RoadNetwork, Bounds, LineString,
    MultiLineString, and lists of Buildings.

    Args:
        object_id: The object ID from run_operation() or list_objects().
        width: Image width in pixels (default 1200).
        height: Image height in pixels (default 800).

    Returns a JSON object with the image file path, or an error message.
    """
    from .renderer import render_to_file, SUPPORTED_TYPES

    try:
        obj = _object_store.get(object_id)
    except KeyError:
        return _fmt({"error": f"Object '{object_id}' not found. Use list_objects() to see available objects."})

    type_name = type(obj).__name__
    if type_name not in SUPPORTED_TYPES and not isinstance(obj, list):
        return _fmt({
            "error": f"Unsupported type for rendering: {type_name}. "
                     f"Supported: {', '.join(sorted(SUPPORTED_TYPES - {'list'}))}",
        })

    image_path = render_to_file(
        obj=obj,
        type_name=type_name,
        label=object_id,
        width=width,
        height=height,
    )

    if image_path is None:
        return _fmt({"error": "Rendering failed. Check that dtcc-viewer is installed and the object has geometry."})

    return _fmt({
        "object_id": object_id,
        "type": type_name,
        "image_path": image_path,
        "width": width,
        "height": height,
    })


# -- Object management -------------------------------------------------------

@mcp.tool()
def delete_object(object_id: str) -> str:
    """Delete a stored object from memory.

    Frees the memory used by the object. Use list_objects() to see what's
    stored before deleting.

    Args:
        object_id: The object ID to delete.

    Returns confirmation with the deleted object's type and label,
    or an error if not found.
    """
    if object_id not in _object_store:
        return _fmt({"error": f"Object '{object_id}' not found. Use list_objects() to see available objects."})

    entry = _object_store._objects[object_id]
    type_name = entry["type"]
    label = entry["label"]
    _object_store.delete(object_id)

    return _fmt({
        "deleted": object_id,
        "type": type_name,
        "label": label,
    })


@mcp.tool()
def get_field_names(object_id: str) -> str:
    """Get available field/data names from a stored object.

    Useful for discovering what data is attached to an object before
    running spatial queries or exports.

    Args:
        object_id: The object ID from run_operation() or list_objects().

    Returns a JSON object with field names, units, and counts.
    """
    try:
        obj = _object_store.get(object_id)
    except KeyError:
        return _fmt({"error": f"Object '{object_id}' not found. Use list_objects() to see available objects."})

    type_name = type(obj).__name__
    fields: list[dict[str, Any]] = []

    if type_name == "SensorCollection":
        seen: dict[tuple[str, str], int] = {}
        for station in obj.stations():
            for geom in station.geometry.values():
                for f in getattr(geom, "fields", []):
                    key = (f.name, f.unit)
                    seen[key] = seen.get(key, 0) + 1
        fields = [{"name": k[0], "unit": k[1], "count": v} for k, v in seen.items()]

    elif type_name in ("Mesh", "PointCloud", "Surface", "VolumeMesh"):
        for f in getattr(obj, "fields", []):
            fields.append({
                "name": f.name,
                "unit": f.unit,
                "count": len(f.values) if hasattr(f.values, "__len__") else 0,
            })

    elif type_name == "Raster":
        data = getattr(obj, "data", None)
        shape = list(data.shape) if data is not None else []
        fields = [{
            "name": "data",
            "unit": "",
            "count": int(data.size) if data is not None else 0,
            "shape": shape,
        }]

    elif type_name == "City":
        if getattr(obj, "buildings", []):
            fields.append({"name": "buildings", "unit": "", "count": len(obj.buildings)})
        if hasattr(obj, "has_terrain") and obj.has_terrain():
            fields.append({"name": "terrain", "unit": "", "count": 1})
        if getattr(obj, "trees", []):
            fields.append({"name": "trees", "unit": "", "count": len(obj.trees)})

    else:
        for f in getattr(obj, "fields", []):
            fields.append({
                "name": f.name,
                "unit": f.unit,
                "count": len(f.values) if hasattr(f.values, "__len__") else 0,
            })

    return _fmt({"object_id": object_id, "type": type_name, "fields": fields})


# -- Export ------------------------------------------------------------------

# Type → (save_function_name, allowed_formats)
_EXPORT_DISPATCH = {
    "PointCloud":  ("save_pointcloud",   {"csv", "las", "laz", "json"}),
    "Mesh":        ("save_mesh",         {"obj", "ply", "stl", "vtk", "vtu", "gltf"}),
    "VolumeMesh":  ("save_volume_mesh",  {"obj", "ply", "stl", "vtk", "vtu"}),
    "Raster":      ("save_raster",       {"csv", "tif", "png", "jpg"}),
    "City":        ("save_city",         {"json"}),
}


@mcp.tool()
def export_object(
    object_id: str,
    format: str,
    filepath: str | None = None,
) -> str:
    """Export a stored object to a file.

    Supported types and formats:
    - PointCloud: csv, las, laz, json
    - Mesh: obj, ply, stl, vtk, vtu, gltf
    - VolumeMesh: obj, ply, stl, vtk, vtu
    - Raster: csv, tif, png, jpg
    - City: json
    - SensorCollection: csv

    Args:
        object_id: The object ID from run_operation() or list_objects().
        format: Output format (e.g. "csv", "obj", "ply", "json").
        filepath: Optional output file path. If omitted, saves to
            /tmp/dtcc_exports/<object_id>.<format>.

    Returns a JSON object with the exported file path, or an error.
    """
    import os

    try:
        obj = _object_store.get(object_id)
    except KeyError:
        return _fmt({"error": f"Object '{object_id}' not found. Use list_objects() to see available objects."})

    type_name = type(obj).__name__
    fmt = format.lower().lstrip(".")

    # SensorCollection: manual CSV export
    if type_name == "SensorCollection":
        if fmt != "csv":
            return _fmt({"error": f"SensorCollection only supports csv format, got '{fmt}'."})

        import csv as csv_mod

        if filepath is None:
            os.makedirs("/tmp/dtcc_exports", exist_ok=True)
            filepath = f"/tmp/dtcc_exports/{object_id}.csv"

        with open(filepath, "w", newline="") as fh:
            writer = csv_mod.writer(fh)
            writer.writerow(["station", "x", "y", "z", "field", "value", "unit"])
            for i, station in enumerate(obj.stations()):
                for geom in station.geometry.values():
                    if not hasattr(geom, "x"):
                        continue
                    for field in getattr(geom, "fields", []):
                        val = field.values[0] if len(field.values) > 0 else ""
                        writer.writerow([
                            station.attributes.get("station_name", f"station_{i}"),
                            round(geom.x, 2), round(geom.y, 2), round(geom.z, 2),
                            field.name, val, field.unit,
                        ])

        return _fmt({"object_id": object_id, "type": type_name, "format": fmt, "filepath": filepath})

    # Standard dtcc-core types
    if type_name not in _EXPORT_DISPATCH:
        return _fmt({"error": f"Export not supported for type '{type_name}'."})

    save_func_name, allowed_formats = _EXPORT_DISPATCH[type_name]

    if fmt not in allowed_formats:
        return _fmt({
            "error": f"Format '{fmt}' not supported for {type_name}. "
                     f"Allowed: {', '.join(sorted(allowed_formats))}."
        })

    if filepath is None:
        os.makedirs("/tmp/dtcc_exports", exist_ok=True)
        filepath = f"/tmp/dtcc_exports/{object_id}.{fmt}"

    from dtcc_core import io as dtcc_io

    save_func = getattr(dtcc_io, save_func_name)

    try:
        save_func(obj, filepath)
    except Exception as exc:
        return _fmt({"error": f"Export failed: {exc}"})

    return _fmt({"object_id": object_id, "type": type_name, "format": fmt, "filepath": filepath})


# -- Rich text output --------------------------------------------------------

@mcp.tool()
def object_to_text(object_id: str, format: str = "markdown") -> str:
    """Get a rich text representation of a stored object.

    Returns formatted markdown with tables and statistics, suitable
    for direct display. Unlike inspect_object() which returns JSON,
    this produces human-readable formatted text.

    Args:
        object_id: The object ID from run_operation() or list_objects().
        format: Output format — currently only "markdown" is supported.

    Returns a markdown-formatted string (not JSON-wrapped).
    """
    if format != "markdown":
        return _fmt({"error": f"Unsupported format '{format}'. Only 'markdown' is supported."})

    try:
        obj = _object_store.get(object_id)
    except KeyError:
        return _fmt({"error": f"Object '{object_id}' not found. Use list_objects() to see available objects."})

    from .serializers import to_markdown

    return to_markdown(obj)


# -- Spatial queries ---------------------------------------------------------

@mcp.tool()
def spatial_query(
    object_id: str,
    query_type: str,
    params: dict[str, Any],
) -> str:
    """Perform spatial queries on stored objects.

    Supported queries:
    - filter_by_bounds: Keep points/stations within a bounding box.
      Params: {bounds: [xmin, ymin, xmax, ymax]}
      Applies to: SensorCollection, PointCloud
    - filter_by_value: Keep stations matching a field condition.
      Params: {field: str, op: ">"|"<"|">="|"<="|"==", value: float}
      Applies to: SensorCollection
    - nearest: Find the nearest station to a point.
      Params: {x: float, y: float}
      Applies to: SensorCollection
    - buildings_by_height: Filter buildings by height range.
      Params: {min_height?: float, max_height?: float}
      Applies to: City

    Args:
        object_id: The object ID from run_operation() or list_objects().
        query_type: Type of spatial query (see above).
        params: Query-specific parameters.

    Returns filtered results. For filter queries, a new object is stored
    and its ID returned. For nearest, station info is returned directly.
    """
    import numpy as np

    try:
        obj = _object_store.get(object_id)
    except KeyError:
        return _fmt({"error": f"Object '{object_id}' not found. Use list_objects() to see available objects."})

    type_name = type(obj).__name__

    # -- filter_by_bounds ---------------------------------------------------
    if query_type == "filter_by_bounds":
        bounds = params.get("bounds")
        if not bounds or len(bounds) != 4:
            return _fmt({"error": "filter_by_bounds requires 'bounds': [xmin, ymin, xmax, ymax]."})
        xmin, ymin, xmax, ymax = bounds

        if type_name == "SensorCollection":
            from dtcc_core.model.object.sensor_collection import SensorCollection as SC

            filtered = SC()
            count = 0
            for station in obj.stations():
                for geom in station.geometry.values():
                    if hasattr(geom, "x"):
                        if xmin <= geom.x <= xmax and ymin <= geom.y <= ymax:
                            filtered.add_station(station)
                            count += 1
                        break
            new_id = _object_store.store(
                filtered, source_op="spatial_query.filter_by_bounds",
                label=f"filtered from {object_id}",
            )
            return _fmt({
                "new_object_id": new_id, "type": "SensorCollection",
                "count": count, "query": "filter_by_bounds", "bounds": bounds,
            })

        if type_name == "PointCloud":
            from dtcc_core.model.geometry.pointcloud import PointCloud as PC

            pts = obj.points
            mask = (
                (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax)
                & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
            )
            filtered = PC()
            filtered.points = pts[mask]
            for attr in ("classification", "intensity", "return_number", "num_returns"):
                arr = getattr(obj, attr, None)
                if isinstance(arr, np.ndarray) and len(arr) == len(pts):
                    setattr(filtered, attr, arr[mask])
            new_id = _object_store.store(
                filtered, source_op="spatial_query.filter_by_bounds",
                label=f"filtered from {object_id}",
            )
            return _fmt({
                "new_object_id": new_id, "type": "PointCloud",
                "count": int(mask.sum()), "original_count": len(pts),
                "query": "filter_by_bounds", "bounds": bounds,
            })

        return _fmt({
            "error": f"filter_by_bounds not supported for type '{type_name}'. "
                     "Use SensorCollection or PointCloud."
        })

    # -- filter_by_value ----------------------------------------------------
    if query_type == "filter_by_value":
        if type_name != "SensorCollection":
            return _fmt({"error": "filter_by_value only supported for SensorCollection."})

        import operator

        field_name = params.get("field")
        op = params.get("op")
        value = params.get("value")
        if not all([field_name, op, value is not None]):
            return _fmt({"error": "filter_by_value requires 'field', 'op', and 'value'."})
        if op not in (">", "<", ">=", "<=", "=="):
            return _fmt({"error": f"Invalid operator '{op}'. Use >, <, >=, <=, or ==."})

        ops = {">": operator.gt, "<": operator.lt, ">=": operator.ge,
               "<=": operator.le, "==": operator.eq}
        cmp = ops[op]

        from dtcc_core.model.object.sensor_collection import SensorCollection as SC

        filtered = SC()
        count = 0
        for station in obj.stations():
            matched = False
            for geom in station.geometry.values():
                for f in getattr(geom, "fields", []):
                    if f.name == field_name and len(f.values) > 0:
                        if cmp(float(f.values[0]), float(value)):
                            matched = True
                        break
                break
            if matched:
                filtered.add_station(station)
                count += 1

        new_id = _object_store.store(
            filtered, source_op="spatial_query.filter_by_value",
            label=f"filtered from {object_id}",
        )
        return _fmt({
            "new_object_id": new_id, "type": "SensorCollection",
            "count": count, "query": "filter_by_value",
            "field": field_name, "op": op, "value": value,
        })

    # -- nearest ------------------------------------------------------------
    if query_type == "nearest":
        if type_name != "SensorCollection":
            return _fmt({"error": "nearest only supported for SensorCollection."})

        x = params.get("x")
        y = params.get("y")
        if x is None or y is None:
            return _fmt({"error": "nearest requires 'x' and 'y'."})

        best_dist = float("inf")
        best_info = None
        for i, station in enumerate(obj.stations()):
            for geom in station.geometry.values():
                if hasattr(geom, "x"):
                    d = (geom.x - x) ** 2 + (geom.y - y) ** 2
                    if d < best_dist:
                        best_dist = d
                        fields_info = [
                            {
                                "name": f.name,
                                "value": float(f.values[0]) if len(f.values) > 0 else None,
                                "unit": f.unit,
                            }
                            for f in getattr(geom, "fields", [])
                        ]
                        best_info = {
                            "station": station.attributes.get("station_name", f"station_{i}"),
                            "x": round(geom.x, 2),
                            "y": round(geom.y, 2),
                            "z": round(geom.z, 2),
                            "distance": round(d ** 0.5, 2),
                            "fields": fields_info,
                        }
                    break

        if best_info is None:
            return _fmt({"error": "No stations found in SensorCollection."})
        return _fmt({"query": "nearest", "result": best_info})

    # -- buildings_by_height ------------------------------------------------
    if query_type == "buildings_by_height":
        if type_name != "City":
            return _fmt({"error": "buildings_by_height only supported for City."})

        min_h = params.get("min_height", 0)
        max_h = params.get("max_height", float("inf"))

        buildings = getattr(obj, "buildings", [])
        filtered = [
            b for b in buildings
            if min_h <= (b.height if getattr(b, "height", None) else 0) <= max_h
        ]

        new_id = _object_store.store(
            filtered, source_op="spatial_query.buildings_by_height",
            label=f"filtered from {object_id}",
        )
        return _fmt({
            "new_object_id": new_id, "type": "list[Building]",
            "count": len(filtered), "original_count": len(buildings),
            "query": "buildings_by_height",
            "min_height": min_h,
            "max_height": max_h if max_h != float("inf") else None,
        })

    return _fmt({
        "error": f"Unknown query_type '{query_type}'. "
                 "Supported: filter_by_bounds, filter_by_value, nearest, buildings_by_height."
    })


# -- Internal helpers --------------------------------------------------------

def _infer_field_name(simulation_name: str) -> str:
    """Map simulation name to a human-readable field name."""
    mapping = {
        "urban_heat_simulation": "temperature",
        "air_quality_field": "concentration",
    }
    return mapping.get(simulation_name, "field")


def main():
    mcp.run()


if __name__ == "__main__":
    main()
