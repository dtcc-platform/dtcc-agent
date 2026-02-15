"""In-memory object store for dtcc-core objects.

Stores intermediate results (PointCloud, Mesh, Raster, etc.) with short
hex IDs so that multi-step pipelines can reference previous outputs.
Thread-safe via a lock; LRU eviction keeps memory bounded.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import numpy as np


def _estimate_bytes(obj: Any) -> int:
    """Estimate memory usage of a dtcc-core object."""
    total = 0
    # Handle raw numpy arrays directly
    if isinstance(obj, np.ndarray):
        return max(obj.nbytes, 64)
    # Walk attributes looking for numpy arrays
    for attr_name in ("points", "classification", "intensity", "return_number",
                      "num_returns", "vertices", "faces", "cells", "markers",
                      "normals", "data"):
        arr = getattr(obj, attr_name, None)
        if isinstance(arr, np.ndarray):
            total += arr.nbytes
    # For objects with children (City, etc.), recurse
    children = getattr(obj, "children", None)
    if isinstance(children, dict):
        for child_list in children.values():
            for child in child_list:
                total += _estimate_bytes(child)
    # For objects with geometry dict
    geometry = getattr(obj, "geometry", None)
    if isinstance(geometry, dict):
        for geom in geometry.values():
            total += _estimate_bytes(geom)
    return max(total, 64)  # minimum 64 bytes for the object itself


class ObjectStore:
    """Thread-safe in-memory store for dtcc-core objects.

    Parameters
    ----------
    max_bytes : int
        Maximum total memory before LRU eviction kicks in.
        Default 2 GB.
    """

    def __init__(self, max_bytes: int = 2 * 1024**3):
        self._lock = threading.Lock()
        self._objects: dict[str, dict[str, Any]] = {}
        self._max_bytes = max_bytes
        self._total_bytes = 0

    def store(self, obj: Any, source_op: str = "", label: str = "") -> str:
        """Store an object and return its short hex ID."""
        obj_id = uuid.uuid4().hex[:8]
        nbytes = _estimate_bytes(obj)
        entry = {
            "object": obj,
            "type": type(obj).__name__,
            "source_op": source_op,
            "label": label,
            "created": time.time(),
            "last_accessed": time.time(),
            "nbytes": nbytes,
        }
        with self._lock:
            self._total_bytes += nbytes
            self._objects[obj_id] = entry
            self._evict_if_needed()
        return obj_id

    def get(self, obj_id: str) -> Any:
        """Retrieve an object by ID. Raises KeyError if not found."""
        with self._lock:
            if obj_id not in self._objects:
                raise KeyError(f"Object '{obj_id}' not found in store")
            self._objects[obj_id]["last_accessed"] = time.time()
            return self._objects[obj_id]["object"]

    def delete(self, obj_id: str) -> None:
        """Remove an object by ID."""
        with self._lock:
            if obj_id in self._objects:
                self._total_bytes -= self._objects[obj_id]["nbytes"]
                del self._objects[obj_id]

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return summaries of stored objects, most recent first."""
        with self._lock:
            entries = sorted(
                self._objects.items(),
                key=lambda kv: kv[1]["created"],
                reverse=True,
            )
            result = []
            for obj_id, entry in entries[:limit]:
                result.append({
                    "id": obj_id,
                    "type": entry["type"],
                    "source_op": entry["source_op"],
                    "label": entry["label"],
                    "created": entry["created"],
                    "nbytes": entry["nbytes"],
                })
        return result

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._objects)

    def __contains__(self, obj_id: str) -> bool:
        return obj_id in self._objects

    def _evict_if_needed(self) -> None:
        """Evict least-recently-accessed objects until under budget.
        Must be called with lock held."""
        while self._total_bytes > self._max_bytes and self._objects:
            # Find LRU entry
            lru_id = min(
                self._objects,
                key=lambda k: self._objects[k]["last_accessed"],
            )
            self._total_bytes -= self._objects[lru_id]["nbytes"]
            del self._objects[lru_id]
