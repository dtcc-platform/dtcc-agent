"""Spatial cropping for cached dtcc-core objects."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def crop_to_bounds(obj: Any, bounds: list[float]) -> Any:
    """Crop an object to the given [xmin, ymin, xmax, ymax] bounds.

    Supports PointCloud (filters points array) and City (filters
    buildings by footprint centroid). Returns unknown types unchanged.

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


def _crop_city(city: Any, bounds: list[float]) -> Any:
    """Filter City buildings whose footprint centroid is within bounds."""
    xmin, ymin, xmax, ymax = bounds
    cropped = deepcopy(city)

    if not hasattr(city, "buildings"):
        return cropped

    kept = []
    for b in city.buildings:
        footprint = getattr(b, "footprint", None)
        if footprint is not None and hasattr(footprint, "centroid"):
            cx, cy = footprint.centroid.x, footprint.centroid.y
            if xmin <= cx <= xmax and ymin <= cy <= ymax:
                kept.append(b)
        else:
            kept.append(b)  # keep if we can't determine position

    cropped.buildings = kept
    logger.info("Cropped City: %d → %d buildings",
                len(city.buildings), len(kept))
    return cropped
