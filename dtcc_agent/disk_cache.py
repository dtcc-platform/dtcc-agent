"""Persistent disk cache for dtcc-core objects.

Stores pickled objects on disk with a JSON metadata index.
Supports spatial containment lookup for datasets and exact
hash lookup for builders. Thread-safe via a lock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path("/tmp/dtcc_cache")
CACHE_TTL_HOURS = 168        # 7 days
CACHE_MAX_SIZE_GB = 10

CACHE_ALLOWLIST = frozenset({
    "datasets.point_cloud",
    "datasets.buildings",
    "builder.build_terrain_raster",
    "builder.build_terrain_surface_mesh",
    "builder.build_city_surface_mesh",
    "builder.raster.slope_aspect",
    "builder.pc_filter.classification_filter",
})


class DiskCache:
    """Persistent disk cache with JSON index and pickle storage."""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self._lock = threading.Lock()
        self._cache_dir = cache_dir
        self._objects_dir = cache_dir / "objects"
        self._index_path = cache_dir / "index.json"
        self._objects_dir.mkdir(parents=True, exist_ok=True)

        # Load or create index
        if self._index_path.exists():
            with open(self._index_path) as f:
                self._index: list[dict[str, Any]] = json.load(f)
            logger.info("Disk cache loaded: %d entries from %s",
                        len(self._index), cache_dir)
        else:
            self._index = []
            logger.info("Disk cache initialized (empty) at %s", cache_dir)

    def _save_index(self) -> None:
        """Write index to disk. Must be called with lock held."""
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def store(
        self,
        obj: Any,
        operation: str,
        category: str,
        params_hash: str,
        bounds: list[float] | None = None,
        source: str | None = None,
        object_type: str = "",
    ) -> str:
        """Pickle an object to disk and add an index entry."""
        cache_id = uuid.uuid4().hex[:8]
        pkl_path = self._objects_dir / f"{cache_id}.pkl"

        with open(pkl_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_bytes = pkl_path.stat().st_size

        entry = {
            "cache_id": cache_id,
            "operation": operation,
            "category": category,
            "params_hash": params_hash,
            "bounds": bounds,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "size_bytes": size_bytes,
            "object_type": object_type,
        }

        with self._lock:
            self._index.append(entry)
            self._save_index()

        logger.info("Cached %s result %s (%.1f MB)",
                     operation, cache_id, size_bytes / 1e6)
        return cache_id

    @staticmethod
    def _bounds_contain(outer: list[float], inner: list[float]) -> bool:
        """Check if outer bounds fully contain inner bounds (EPSG:3006)."""
        return (outer[0] <= inner[0] and outer[1] <= inner[1] and
                outer[2] >= inner[2] and outer[3] >= inner[3])

    def dataset_lookup(
        self,
        operation: str,
        source: str,
        params_hash: str,
        requested_bounds: list[float],
    ) -> tuple[str, list[float]] | None:
        """Find a cached dataset whose bounds contain the requested bounds.

        Returns (cache_id, cached_bounds) or None.
        Prefers the smallest containing entry to minimize cropping.
        """
        now = datetime.now()
        candidates = []

        with self._lock:
            for entry in self._index:
                if entry["operation"] != operation:
                    continue
                if entry.get("source") != source:
                    continue
                if entry["params_hash"] != params_hash:
                    continue
                if entry.get("bounds") is None:
                    continue
                # TTL check
                cached_time = datetime.fromisoformat(entry["timestamp"])
                age_hours = (now - cached_time).total_seconds() / 3600
                if age_hours > CACHE_TTL_HOURS:
                    continue
                # Containment check
                if self._bounds_contain(entry["bounds"], requested_bounds):
                    # Score by area (smaller is better — less cropping)
                    area = ((entry["bounds"][2] - entry["bounds"][0]) *
                            (entry["bounds"][3] - entry["bounds"][1]))
                    candidates.append((area, entry["cache_id"], entry["bounds"]))

        if not candidates:
            logger.debug("Disk cache miss: %s (no containing bounds)", operation)
            return None

        # Pick smallest containing area
        candidates.sort()
        best = candidates[0]
        logger.info("Disk cache hit: %s, cache_id=%s, cached_bounds=%s",
                    operation, best[1], best[2])
        return best[1], best[2]

    def builder_lookup(
        self,
        operation: str,
        params_hash: str,
    ) -> str | None:
        """Find a cached builder result by exact operation + params hash.

        Returns cache_id or None.
        """
        now = datetime.now()
        with self._lock:
            for entry in self._index:
                if entry["operation"] != operation:
                    continue
                if entry["params_hash"] != params_hash:
                    continue
                # TTL check
                cached_time = datetime.fromisoformat(entry["timestamp"])
                age_hours = (now - cached_time).total_seconds() / 3600
                if age_hours > CACHE_TTL_HOURS:
                    continue

                logger.info("Disk cache hit: %s, cache_id=%s", operation, entry["cache_id"])
                return entry["cache_id"]

        logger.debug("Disk cache miss: %s (no matching hash)", operation)
        return None

    def cleanup(self) -> int:
        """Remove expired entries and enforce disk budget. Returns count removed."""
        now = datetime.now()
        removed = 0

        with self._lock:
            surviving = []
            for entry in self._index:
                cached_time = datetime.fromisoformat(entry["timestamp"])
                age_hours = (now - cached_time).total_seconds() / 3600
                if age_hours > CACHE_TTL_HOURS:
                    # Delete pickle file
                    pkl_path = self._objects_dir / f"{entry['cache_id']}.pkl"
                    pkl_path.unlink(missing_ok=True)
                    removed += 1
                else:
                    surviving.append(entry)

            # Enforce disk budget: evict oldest first
            total_bytes = sum(e.get("size_bytes", 0) for e in surviving)
            max_bytes = CACHE_MAX_SIZE_GB * 1024**3
            surviving.sort(key=lambda e: e["timestamp"])
            while total_bytes > max_bytes and surviving:
                oldest = surviving.pop(0)
                pkl_path = self._objects_dir / f"{oldest['cache_id']}.pkl"
                pkl_path.unlink(missing_ok=True)
                total_bytes -= oldest.get("size_bytes", 0)
                removed += 1

            self._index = surviving
            self._save_index()

        if removed:
            logger.info("Disk cache cleanup: removed %d entries", removed)
        return removed

    def load(self, cache_id: str) -> Any:
        """Load a pickled object by cache_id."""
        pkl_path = self._objects_dir / f"{cache_id}.pkl"
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
