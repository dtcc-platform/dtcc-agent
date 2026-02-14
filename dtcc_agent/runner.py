"""Simulation runner: direct in-process access to dtcc-sim simulations.

Wraps the dtcc_core dataset registry to discover, configure, and run
simulations. Results stay in memory as dolfinx Function objects — no
file I/O needed for the agent to read them.

Usage:
    from dtcc_agent.runner import list_simulations, run

    sims = list_simulations()
    result = run("urban_heat_simulation", bounds=[...], parameters={...})
    values = result.x.array  # numpy array of field values
"""

from __future__ import annotations

import json
from typing import Any

# Import dtcc_sim.datasets to trigger auto-registration of simulation
# datasets into the dtcc_core registry. This must happen before we
# query the registry.
import dtcc_sim.datasets  # noqa: F401

from dtcc_core.datasets.registry import list_datasets as _list_all, get_dataset

# Names of datasets that are simulations (as opposed to data fetchers
# like "buildings", "point_cloud", etc.). We tag them explicitly so
# the LLM only sees runnable simulations in the list_simulations tool.
_SIMULATION_NAMES = {
    "urban_heat_simulation",
    "air_quality_field",
}


def list_simulations() -> list[dict[str, str]]:
    """Return metadata for all registered simulation datasets.

    Returns
    -------
    list of dicts, each with keys: name, description
    """
    all_datasets = _list_all()
    result = []
    for name, ds in all_datasets.items():
        if name in _SIMULATION_NAMES:
            result.append({
                "name": name,
                "description": getattr(ds, "description", ""),
            })
    return result


def get_schema(name: str) -> dict[str, Any]:
    """Return the JSON schema for a simulation's parameters.

    Parameters
    ----------
    name : str
        Simulation name (e.g. "urban_heat_simulation").

    Returns
    -------
    dict — the JSON Schema from the Pydantic ArgsModel.
    """
    ds = get_dataset(name)
    return ds.show_options()


def run(
    name: str,
    bounds: list[float],
    parameters: dict[str, Any] | None = None,
) -> Any:
    """Run a simulation and return the result object.

    Parameters
    ----------
    name : str
        Simulation name.
    bounds : list[float]
        Bounding box [minx, miny, maxx, maxy] in EPSG:3006.
    parameters : dict, optional
        Simulation-specific parameters. See get_schema() for valid keys.

    Returns
    -------
    For urban_heat_simulation: dolfinx.fem.Function
        Access values via result.x.array (numpy array)
    For air_quality_field: dolfinx.fem.Function
        Access values via result.x.array (numpy array)

    Raises
    ------
    KeyError
        If simulation name is not registered.
    Exception
        Propagated from the underlying simulation.
    """
    ds = get_dataset(name)

    kwargs: dict[str, Any] = {"bounds": bounds}
    if parameters:
        kwargs.update(parameters)

    return ds(**kwargs)


def get_buildings(
    bounds: list[float],
    source: str = "LM",
    smallest_building_size: float = 15.0,
    max_buildings: int = 100,
) -> dict[str, Any]:
    """Fetch buildings in a bounding box and return a JSON-friendly summary.

    Parameters
    ----------
    bounds : list[float]
        [minx, miny, maxx, maxy] in EPSG:3006.
    source : str
        Data source: "LM" (Lantmäteriet) or "OSM" (OpenStreetMap).
    smallest_building_size : float
        Minimum footprint area in m² to include.
    max_buildings : int
        Max number of per-building details to return.

    Returns
    -------
    dict with keys: bounds, crs, source, num_buildings, buildings,
    height_stats, total_footprint_area_m2
    """
    import numpy as np

    ds = get_dataset("buildings")
    buildings = ds(
        bounds=bounds,
        source=source,
        smallest_building_size=smallest_building_size,
    )

    details = []
    heights = []

    for i, b in enumerate(buildings):
        h = b.height
        if h is not None and h > 0:
            heights.append(h)

        # Compute footprint area from lod0 vertices
        footprint_area = None
        num_vertices = 0
        if b.lod0 is not None:
            verts = b.lod0.vertices
            num_vertices = len(verts)
            if num_vertices >= 3:
                v = np.array(verts)
                x, y = v[:, 0], v[:, 1]
                footprint_area = round(
                    0.5 * abs(float(
                        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
                    )),
                    1,
                )

        ground_height = b.attributes.get("ground_height")

        if i < max_buildings:
            detail = {
                "id": str(b.id),
                "height_m": round(h, 1) if h else None,
                "ground_height_m": round(ground_height, 1) if ground_height else None,
                "footprint_vertices": num_vertices,
            }
            if footprint_area is not None:
                detail["footprint_area_m2"] = footprint_area
            details.append(detail)

    heights_arr = np.array(heights) if heights else np.array([0.0])
    total_area = sum(d.get("footprint_area_m2", 0) for d in details)

    return {
        "bounds": bounds,
        "crs": "EPSG:3006",
        "source": source,
        "num_buildings": len(buildings),
        "buildings": details,
        "truncated": len(buildings) > max_buildings,
        "height_stats": {
            "min_m": round(float(heights_arr.min()), 1),
            "max_m": round(float(heights_arr.max()), 1),
            "mean_m": round(float(heights_arr.mean()), 1),
            "median_m": round(float(np.median(heights_arr)), 1),
        },
        "total_footprint_area_m2": round(total_area, 1),
    }
