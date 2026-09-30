"""Render a stored dtcc-core object to a PNG with matplotlib's Agg backend.

No OpenGL and no display, so it runs the same in a worker thread, on macOS
and in the container (U9). Surfaces, footprints and lines are drawn in plan
view; meshes and point clouds in 3D; rasters as an image. Large geometry is
subsampled so a render stays quick.

This is the M1a renderer. Whether the chat page draws geometry itself is D1/D2
in the rebuild plan (M3).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logger = logging.getLogger(__name__)

MAX_POINTS = 50_000
MAX_FACES = 20_000
MAX_PIXELS = 2400


def _sample(rows: np.ndarray, limit: int) -> np.ndarray:
    if len(rows) <= limit:
        return rows
    return rows[:: int(np.ceil(len(rows) / limit))]


def _xy(vertices: Any) -> np.ndarray:
    return np.asarray(vertices, dtype=float)[:, :2]


# -- Plan view: each returns (polygons, lines, points) in x/y ---------------

def _surface_polygons(surface: Any) -> list[np.ndarray]:
    return [_xy(surface.vertices)] if len(surface.vertices) else []


def _building_polygons(building: Any) -> list[np.ndarray]:
    footprint = building.get_footprint()
    return _surface_polygons(footprint) if footprint is not None else []


def _plan(obj: Any, type_name: str) -> tuple[list, list, np.ndarray | None]:
    polygons: list[np.ndarray] = []
    lines: list[np.ndarray] = []
    points = None
    if type_name == "list":
        for item in obj:
            item_type = type(item).__name__
            if item_type == "Building":
                polygons += _building_polygons(item)
            elif item_type == "Surface":
                polygons += _surface_polygons(item)
    elif type_name in ("City", "BuildingCollection"):
        for building in obj.buildings:
            polygons += _building_polygons(building)
    elif type_name == "Building":
        polygons = _building_polygons(obj)
    elif type_name == "Surface":
        polygons = _surface_polygons(obj)
    elif type_name == "MultiSurface":
        for surface in obj.surfaces:
            polygons += _surface_polygons(surface)
    elif type_name == "FootprintCollection":
        for surface in obj.footprints:
            polygons += _surface_polygons(surface)
    elif type_name == "LineString":
        lines = [_xy(obj.vertices)]
    elif type_name == "MultiLineString":
        lines = [_xy(line.vertices) for line in obj.linestrings]
    elif type_name == "RoadNetwork":
        vertices = _xy(obj.vertices)
        lines = [vertices[list(edge)] for edge in np.asarray(obj.edges, dtype=int)]
    elif type_name == "Bounds":
        x0, y0, x1, y1 = obj.xmin, obj.ymin, obj.xmax, obj.ymax
        polygons = [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])]
    elif type_name == "SensorCollection":
        coords = [
            (geom.x, geom.y)
            for station in obj.stations()
            for geom in station.geometry.values()
            if hasattr(geom, "x")
        ]
        points = np.array(coords, dtype=float) if coords else None
    return polygons, lines, points


def _draw_plan(fig: Figure, obj: Any, type_name: str) -> bool:
    polygons, lines, points = _plan(obj, type_name)
    if not polygons and not lines and points is None:
        return False
    ax = fig.add_subplot()
    if polygons:
        ax.add_collection(PolyCollection(
            polygons, facecolor="#9db4c0", edgecolor="#2b3a42", linewidth=0.4,
        ))
    if lines:
        ax.add_collection(LineCollection(lines, colors="#2b3a42", linewidth=0.8))
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], s=12, color="#c0392b", zorder=3)
    ax.autoscale_view()
    ax.set_aspect("equal")
    # SWEREF 99 coordinates are 6-7 digits; show them whole, not as an offset.
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return True


# -- 3D and image -----------------------------------------------------------

def _draw_mesh(fig: Figure, vertices: np.ndarray, faces: np.ndarray) -> bool:
    if not len(vertices) or not len(faces):
        return False
    faces = _sample(np.asarray(faces, dtype=int)[:, :3], MAX_FACES)
    triangles = vertices[faces]
    heights = triangles[:, :, 2].mean(axis=1)
    span = np.ptp(heights) or 1.0
    colors = colormaps["viridis"]((heights - heights.min()) / span)
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, linewidths=0))
    _fit_3d(ax, vertices)
    return True


def _draw_points(fig: Figure, points: np.ndarray) -> bool:
    if not len(points):
        return False
    points = _sample(points, MAX_POINTS)
    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], s=0.5,
               cmap="viridis", linewidths=0)
    _fit_3d(ax, points)
    return True


def _draw_raster(fig: Figure, raster: Any) -> bool:
    data = np.asarray(raster.data, dtype=float)
    if data.ndim != 2 or not data.size:
        return False
    if raster.nodata is not None:
        data = np.ma.masked_equal(data, raster.nodata)
    ax = fig.add_subplot()
    image = ax.imshow(data, cmap="viridis")
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_axis_off()
    return True


def _fit_3d(ax: Any, xyz: np.ndarray) -> None:
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2] if hi[2] > lo[2] else lo[2] + 1)
    ax.set_box_aspect(np.maximum(hi - lo, (hi - lo).max() / 20 or 1.0))
    ax.set_axis_off()


def _draw(fig: Figure, obj: Any, type_name: str) -> bool:
    if type_name == "Mesh":
        return _draw_mesh(fig, np.asarray(obj.vertices, dtype=float), obj.faces)
    if type_name == "VolumeMesh":
        return _draw_points(fig, np.asarray(obj.vertices, dtype=float))
    if type_name == "PointCloud":
        return _draw_points(fig, np.asarray(obj.points, dtype=float))
    if type_name == "Raster":
        return _draw_raster(fig, obj)
    return _draw_plan(fig, obj, type_name)


SUPPORTED_TYPES = {
    "Mesh", "VolumeMesh", "PointCloud", "Raster", "City", "BuildingCollection",
    "FootprintCollection", "Building", "Surface",
    "MultiSurface", "LineString", "MultiLineString", "RoadNetwork", "Bounds",
    "SensorCollection", "list",
}


def render_to_file(obj: Any, type_name: str, filepath: Path,
                   width: int = 1200, height: int = 800) -> bool:
    """Draw `obj` into a PNG at `filepath`. False when there is nothing to draw."""
    width = max(100, min(width, MAX_PIXELS))
    height = max(100, min(height, MAX_PIXELS))
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    try:
        if not _draw(fig, obj, type_name):
            return False
        fig.savefig(filepath, format="png")
    except Exception as exc:
        logger.warning("Rendering %s failed: %s", type_name, exc)
        return False
    return True
