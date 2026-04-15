"""Simulation runner for local or remote dtcc-sim simulations.

In mini-service mode, configured with DTCC_REMOTE_SERVICES or
DTCC_SIM_SERVICE_URL, simulations are delegated through dtcc-core's remote
dataset protocol. Without a remote service configured, this module falls back
to direct in-process dtcc-sim access.

Usage:
    from dtcc_agent.runner import list_simulations, run

    sims = list_simulations()
    result = run("urban_heat_simulation", bounds=[...], parameters={...})
    values = result.x.array  # numpy array of field values
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

# Names of datasets that are simulations (as opposed to data fetchers
# like "buildings", "point_cloud", etc.). We tag them explicitly so
# the LLM only sees runnable simulations in the list_simulations tool.
_SIMULATION_NAMES = {
    "urban_heat_simulation",
    "air_quality_field",
}


@dataclass
class RemoteSimulationResult:
    """Metadata returned when a simulation runs in the dtcc-sim mini-service."""

    simulation: str
    bounds: list[float]
    parameters: dict[str, Any]
    base_url: str
    task_id: str | None
    result_file: str | None
    size_bytes: int | None
    output_format: str | None
    content_type: str | None = None
    status: str = "completed"
    remote: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_REGISTERED_REMOTE_SERVICES: set[str] = set()


def _remote_services() -> list[str]:
    """Return configured remote service URLs."""
    explicit = os.getenv("DTCC_SIM_SERVICE_URL", "").strip()
    if explicit:
        return [explicit.rstrip("/")]

    return [
        url.strip().rstrip("/")
        for url in os.getenv("DTCC_REMOTE_SERVICES", "").split(",")
        if url.strip()
    ]


def _remote_base_url() -> str | None:
    """Return the primary configured dtcc-sim service URL, if any."""
    services = _remote_services()
    return services[0] if services else None


def _ensure_remote_services_registered() -> None:
    """Register configured remote services using dtcc-core's shared protocol."""
    from dtcc_core.datasets import register_remote_service

    for url in _remote_services():
        if url in _REGISTERED_REMOTE_SERVICES:
            continue
        registered = register_remote_service(url)
        if registered:
            _REGISTERED_REMOTE_SERVICES.add(url)


def _list_all_remote():
    _ensure_remote_services_registered()
    from dtcc_core.datasets.registry import list_datasets as _list_all

    return _list_all()


def _get_remote_dataset(name: str):
    _ensure_remote_services_registered()
    from dtcc_core.datasets.registry import get_dataset

    ds = get_dataset(name)
    if not getattr(ds, "base_url", None):
        raise KeyError(name)
    return ds


def _list_all_local():
    # Import dtcc_sim.datasets lazily so the plain Python mini-service can start
    # without FEniCSx/dtcc-sim installed. Direct mode still works when those
    # packages are available in the current environment.
    import dtcc_sim.datasets  # noqa: F401
    from dtcc_core.datasets.registry import list_datasets as _list_all

    return _list_all()


def _get_local_dataset(name: str):
    import dtcc_sim.datasets  # noqa: F401
    from dtcc_core.datasets.registry import get_dataset

    return get_dataset(name)


def _get_core_dataset(name: str):
    from dtcc_core.datasets.registry import get_dataset

    return get_dataset(name)


def list_simulations() -> list[dict[str, str]]:
    """Return metadata for all registered simulation datasets.

    Returns
    -------
    list of dicts, each with keys: name, description
    """
    base_url = _remote_base_url()
    if base_url:
        all_datasets = _list_all_remote()
        return [
            {
                "name": name,
                "description": getattr(ds, "description", ""),
            }
            for name, ds in all_datasets.items()
            if name in _SIMULATION_NAMES and getattr(ds, "base_url", None)
        ]

    all_datasets = _list_all_local()
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
    base_url = _remote_base_url()
    if base_url:
        ds = _get_remote_dataset(name)
        return ds.show_options()

    ds = _get_local_dataset(name)
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
    kwargs: dict[str, Any] = {"bounds": bounds}
    if parameters:
        kwargs.update(parameters)

    base_url = _remote_base_url()
    if base_url:
        ds = _get_remote_dataset(name)
        formats = getattr(ds, "supported_formats", None) or ["bin"]
        requested_format = kwargs.setdefault("format", formats[0])
        remote_info: dict[str, Any] = {}

        # Use dtcc-core's RemoteDatasetDescriptor implementation instead of
        # carrying a second copy of the submit/status/result protocol here.
        validated = ds.validate(dict(kwargs)) if hasattr(ds, "validate") else kwargs
        result = ds.build(validated, remote_info_callback=remote_info.update)
        if isinstance(result, tuple) and len(result) >= 3:
            data, output_format, content_type = result[:3]
            size_bytes = len(data) if isinstance(data, bytes) else None
        else:
            output_format = requested_format
            content_type = None
            size_bytes = None

        return RemoteSimulationResult(
            simulation=name,
            bounds=bounds,
            parameters=kwargs,
            base_url=getattr(ds, "base_url", base_url),
            task_id=remote_info.get("remote_task_id"),
            result_file=None,
            size_bytes=size_bytes,
            output_format=output_format,
            content_type=content_type,
        )

    ds = _get_local_dataset(name)
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

    ds = _get_core_dataset("buildings")
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
