"""Bridge between ObjectStore objects and dtcc-viewer for offscreen rendering.

Provides render_to_file() which takes a dtcc-core object, adds it to a
dtcc-viewer Scene, and captures a PNG screenshot via an offscreen OpenGL window.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

# Type name → (scene_method_name, extra_kwargs)
_DISPATCH_TABLE: dict[str, tuple[str, dict[str, Any]]] = {
    "Mesh": ("add_mesh", {}),
    "PointCloud": ("add_pointcloud", {"size": 0.5}),
    "City": ("add_city", {}),
    "Raster": ("add_raster", {}),
    "Building": ("add_building", {}),
    "Surface": ("add_surface", {}),
    "MultiSurface": ("add_multisurface", {}),
    "VolumeMesh": ("add_volume_mesh", {}),
    "RoadNetwork": ("add_roadnetwork", {}),
    "Bounds": ("add_bounds", {}),
    "LineString": ("add_linestring", {}),
    "MultiLineString": ("add_multilinestring", {}),
    "SensorCollection": ("add_sensor_collection", {}),
}

SUPPORTED_TYPES = set(_DISPATCH_TABLE.keys()) | {"list"}


def render_to_file(
    obj: Any,
    type_name: str,
    label: str = "object",
    width: int = 1200,
    height: int = 800,
    filepath: str | None = None,
) -> str | None:
    """Render a dtcc-core object to a PNG file.

    Parameters
    ----------
    obj : Any
        A dtcc-core object (Mesh, PointCloud, City, etc.) or a list of
        Building objects.
    type_name : str
        The type name of the object (e.g. "Mesh", "PointCloud").
    label : str
        Human-readable label used as the scene object name.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    filepath : str, optional
        Output PNG path. If not provided, a temp file is created.

    Returns
    -------
    str or None
        The file path of the saved PNG, or None on failure.
    """
    try:
        from dtcc_viewer.opengl.window import Window
        from dtcc_viewer.opengl.scene import Scene
    except ImportError:
        logger.warning("dtcc-viewer is not installed — cannot render screenshots.")
        return None

    if filepath is None:
        tmp_dir = tempfile.mkdtemp(prefix="dtcc_screenshots_")
        filepath = os.path.join(tmp_dir, f"{label}.png")

    try:
        # Window must be created FIRST — it initializes the GLFW/OpenGL
        # context that Scene.__init__() needs (calls glGetIntegerv).
        window = Window(width, height, visible=False)
    except Exception as exc:
        logger.warning(f"Failed to create OpenGL window: {exc}")
        return None

    scene = Scene()

    # Handle list of Buildings
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            item_type = type(item).__name__
            if item_type in _DISPATCH_TABLE:
                method_name, kwargs = _DISPATCH_TABLE[item_type]
                getattr(scene, method_name)(f"{label}_{i}", item, **kwargs)
            else:
                logger.warning(f"Unsupported type in list: {item_type}")
        if len(scene.wrappers) == 0:
            logger.warning("No renderable objects in list.")
            return None
    elif type_name in _DISPATCH_TABLE:
        method_name, kwargs = _DISPATCH_TABLE[type_name]
        getattr(scene, method_name)(label, obj, **kwargs)
    else:
        logger.warning(f"Unsupported type for rendering: {type_name}")
        return None

    try:
        success = window.screenshot(scene, filepath, width, height)
        if success:
            return filepath
        return None
    except Exception as exc:
        logger.warning(f"Screenshot failed: {exc}")
        return None
