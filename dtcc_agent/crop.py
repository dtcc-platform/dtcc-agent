"""Spatial cropping for cached dtcc-core objects."""

from __future__ import annotations

import logging
from copy import copy, deepcopy
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def crop_to_bounds(obj: Any, bounds: list[float]) -> Any:
    """Crop an object to the given [xmin, ymin, xmax, ymax] bounds.

    Supports PointCloud (filters points array) and a Core City or
    BuildingCollection (keeps the buildings Core itself would load for
    these bounds). Returns unknown types unchanged.

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


def _crop_city(city: Any, bounds: list[float]) -> Any:
    """Keep the buildings a fresh Core download of ``bounds`` would keep.

    Core keeps a footprint only when it lies wholly inside the bounds shrunk
    by FOOTPRINT_EDGE_DISTANCE, for LM and OSM alike, so a cropped cache hit
    applies the same test and counts what a fresh download counts. A
    building without a footprint is dropped, as Core's size filter drops it.
    """
    from dtcc_core.io.vector_utils import create_bounds_filter
    from dtcc_core.model import Bounds

    inside = create_bounds_filter(
        Bounds(*bounds), buffer=-FOOTPRINT_EDGE_DISTANCE, strategy="contains",
    )
    kept = []
    for b in city.buildings:
        footprint = b.footprint()
        if footprint is not None and inside["strategy"](
            inside["geometry"], footprint.to_polygon(simplify=0.0),
        ):
            kept.append(b)

    # Shallow copy: callers crop a freshly loaded cache object nothing else
    # holds, so sharing its buildings is safe and skips copying geometry.
    cropped = copy(city)
    cropped.buildings = kept
    logger.info("Cropped City: %d → %d buildings",
                len(city.buildings), len(kept))
    return cropped
