"""Spatial cropping for cached dtcc-core objects."""

from __future__ import annotations

import logging
from copy import copy, deepcopy
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def crop_to_bounds(obj: Any, bounds: list[float]) -> Any:
    """Crop an object to the given [xmin, ymin, xmax, ymax] bounds.

    Supports PointCloud (filters points array) and a Core
    BuildingCollection (keeps the buildings Core itself would load for
    these bounds). Returns other types unchanged, including a Core City,
    whose buildings cannot be replaced.

    Detection uses duck-typing: any object with a ``points`` numpy
    array is treated as a PointCloud; any object with a ``buildings``
    list is treated as a City.
    """
    # Duck-type: PointCloud has a numpy .points array with shape (N, 3+)
    pts = getattr(obj, "points", None)
    if isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] >= 2:
        return _crop_pointcloud(obj, bounds)

    # Duck-type: City has a .buildings list
    if hasattr(obj, "buildings") and isinstance(getattr(obj, "buildings"), list):
        return _crop_city(obj, bounds)

    # Unknown type — return as-is
    type_name = type(obj).__name__
    logger.debug("No cropping for type %s, returning as-is", type_name)
    return obj


def _crop_pointcloud(pc: Any, bounds: list[float]) -> Any:
    """Filter PointCloud points to those within bounds."""
    xmin, ymin, xmax, ymax = bounds
    pts = pc.points
    mask = (
        (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
        (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
    )

    cropped = deepcopy(pc)
    cropped.points = pts[mask]

    # Filter all matching per-point arrays
    for attr in ("classification", "intensity", "return_number", "num_returns"):
        arr = getattr(pc, attr, None)
        if isinstance(arr, np.ndarray) and len(arr) == len(pts):
            setattr(cropped, attr, arr[mask])

    logger.info("Cropped PointCloud: %d → %d points", len(pts), mask.sum())
    return cropped


# Core's footprint loader (io.footprints.load) keeps a footprint only when it
# lies wholly inside the bounds shrunk by this many metres.
FOOTPRINT_EDGE_DISTANCE = 2.0

# The source feature a building was split from. Core copies the feature's
# properties onto every part, but only takes Building.id from a property
# named "id", which neither source has, so parts get unrelated random ids.
SOURCE_ID_ATTRIBUTES = ("objektidentitet", "osm_id")  # LM, OSM


def _source_bounds(b: Any) -> list[float] | None:
    """The extent of the source feature Core tested for ``b`` (dtcc-core#133).

    Core tests the whole source feature, then splits it into parts, drops
    parts under the size limit and repairs invalid outlines, and records the
    feature's extent from before all that on every building it produced.
    """
    value = b.attributes.get("source_bounds")
    return value if value is not None and len(value) == 4 else None


def _judged_with(b: Any) -> Any:
    """The buildings ``b`` is kept or dropped with.

    With source_bounds, a building is judged alone: parts of one feature share
    the same bounds, so they fall together anyway, and LM gives several
    separate features one objektidentitet, which Core tests one by one (#49).
    Without it, parts of one source feature are grouped by its id.
    """
    if _source_bounds(b) is not None:
        return id(b)
    for name in SOURCE_ID_ATTRIBUTES:
        value = b.attributes.get(name)
        if value is not None:
            return (name, value)
    return b.id or id(b)  # no id at all must not group unrelated buildings


def _tested_shape(b: Any) -> Any:
    """The shape Core's containment test runs on, or None if there is none.

    A shape lies inside an axis-aligned rectangle exactly when its bounding
    box does, so testing the source feature's bounding box is Core's own
    test. A building without source_bounds (cached before #133) falls back to
    its unsimplified LOD0 outline: footprint() would simplify it by 1 cm.
    """
    from dtcc_core.model import GeometryType
    from shapely.geometry import box

    if (source_bounds := _source_bounds(b)) is not None:
        return box(*source_bounds)
    outline = b.flatten_geometry(GeometryType.LOD0)
    return outline.to_polygon(simplify=0.0) if outline is not None else None


def _crop_city(city: Any, bounds: list[float]) -> Any:
    """Keep the buildings a fresh Core download of ``bounds`` would keep.

    Core keeps a source feature only when it lies wholly inside the bounds
    shrunk by FOOTPRINT_EDGE_DISTANCE, for LM and OSM alike, so a cropped
    cache hit applies the same test (_tested_shape) to the same buildings
    (_judged_with) and counts what a fresh download counts. A building
    without an outline is dropped, as Core's size filter drops it.
    """
    from dtcc_core.io.vector_utils import create_bounds_filter
    from dtcc_core.model import Bounds, GeometryType

    inside = create_bounds_filter(
        Bounds(*bounds), buffer=-FOOTPRINT_EDGE_DISTANCE, strategy="contains",
    )
    whole = {}  # judged together -> every one of them inside
    for b in city.buildings:
        shape = _tested_shape(b)
        ok = (b.flatten_geometry(GeometryType.LOD0) is not None
              and shape is not None and not shape.is_empty
              and inside["strategy"](inside["geometry"], shape))
        group = _judged_with(b)
        whole[group] = whole.get(group, True) and ok
    kept = [b for b in city.buildings if whole[_judged_with(b)]]

    # Shallow copy: callers crop a freshly loaded cache object nothing else
    # holds, so sharing its buildings is safe and skips copying geometry.
    cropped = copy(city)
    try:
        cropped.buildings = kept
    except AttributeError:  # a Core City derives buildings; it has no setter
        logger.debug("Cannot crop %s, returning as-is", type(city).__name__)
        return city
    logger.info("Cropped City: %d → %d buildings",
                len(city.buildings), len(kept))
    return cropped
