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

mcp = FastMCP("dtcc-agent")

# In-memory store for simulation results so the agent can refer back
# to previous runs (e.g. for comparison). Keyed by run_id.
_results: dict[str, dict[str, Any]] = {}


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
