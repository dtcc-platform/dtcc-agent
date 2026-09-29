"""Persistent disk cache for dtcc-core objects.

Stores pickled objects on disk with a JSON metadata index.
Supports spatial containment lookup for datasets and exact
hash lookup for builders. Thread-safe via a lock, and safe for several
processes sharing one directory via a file lock on the index.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import pickle
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def _default_cache_dir() -> Path:
    # Per user, never a shared /tmp path: loading a pickle runs code, so the
    # directory must be one only this user can write.
    return Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache") / "dtcc_agent"


CACHE_DIR = Path(os.getenv("DTCC_AGENT_CACHE_DIR") or _default_cache_dir())
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


def content_fingerprint(obj_metadata: dict) -> str:
    """Compute a stable fingerprint from ObjectStore metadata.

    Used to generate cache keys for builder operations where
    input objects are referenced by transient IDs.
    """
    key_parts = {
        "type": obj_metadata.get("type", ""),
        "source_op": obj_metadata.get("source_op", ""),
        "nbytes": obj_metadata.get("nbytes", 0),
        "label": obj_metadata.get("label", ""),
    }
    canonical = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def canonical_params_hash(
    operation: str,
    params: dict,
    object_fingerprints: dict = None,
) -> str:
    """Compute a stable hash for operation parameters.

    For builder operations, object-ref params are replaced with
    their content fingerprints. Bounds are excluded (handled
    separately for datasets via containment).
    """
    canonical = dict(sorted(params.items()))

    # Replace object-ref param values with fingerprints
    if object_fingerprints:
        for key, fingerprint in object_fingerprints.items():
            if key in canonical:
                canonical[key] = f"__fp:{fingerprint}"

    # Remove bounds (handled by containment for datasets)
    canonical.pop("bounds", None)

    payload = json.dumps({"op": operation, "params": canonical},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class CacheDirError(RuntimeError):
    """The cache directory could hold pickles planted by another user."""


def _private_dir(path: Path) -> None:
    """Create ``path`` for this user only, or refuse one others can write."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    st = path.stat()
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise CacheDirError(
            f"Cache directory {path} is owned by another user. Loading its "
            "pickles would run their code; set DTCC_AGENT_CACHE_DIR to a "
            "directory you own.")
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise CacheDirError(
            f"Cache directory {path} is writable by other users. Loading its "
            f"pickles would run their code; run `chmod go-w {path}` or set "
            "DTCC_AGENT_CACHE_DIR to a private directory.")


class DiskCache:
    """Persistent disk cache with JSON index and pickle storage."""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self._lock = threading.Lock()
        self._cache_dir = cache_dir
        self._objects_dir = cache_dir / "objects"
        self._index_path = cache_dir / "index.json"
        self._lock_path = cache_dir / "index.lock"
        _private_dir(cache_dir)
        _private_dir(self._objects_dir)

        self._index: list[dict[str, Any]] = []
        self._index_version: tuple[int, int] | None = None
        with self._lock:
            self._reload_if_changed()
        logger.info("Disk cache at %s: %d entries", cache_dir, len(self._index))

        # Clean up expired entries on startup
        self.cleanup()

    def _version(self) -> tuple[int, int] | None:
        try:
            st = self._index_path.stat()
        except FileNotFoundError:
            return None
        return st.st_mtime_ns, st.st_size

    def _reload_if_changed(self) -> None:
        """Pick up entries other processes wrote. Must be called with lock held."""
        version = self._version()
        if version == self._index_version:
            return
        try:
            with open(self._index_path) as f:
                self._index = json.load(f)
        except FileNotFoundError:
            self._index = []
        except json.JSONDecodeError:
            logger.warning("Disk cache index %s is unreadable; starting empty",
                           self._index_path)
            self._index = []
        self._index_version = version

    def _save_index(self) -> None:
        """Write index to disk atomically. Must be called with lock held."""
        tmp = self._index_path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(self._index, f, indent=2)
        os.replace(tmp, self._index_path)
        self._index_version = self._version()

    @contextmanager
    def _editing_index(self):
        """Read-modify-write the index under a lock every process honours, so
        one process never overwrites entries another has added."""
        with self._lock, open(self._lock_path, "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                self._index_version = None  # always re-read under the lock
                self._reload_if_changed()
                yield
                self._save_index()
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

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

        with self._editing_index():
            self._index.append(entry)

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
            self._reload_if_changed()
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
            self._reload_if_changed()
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

        with self._editing_index():
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

        if removed:
            logger.info("Disk cache cleanup: removed %d entries", removed)
        return removed

    def load(self, cache_id: str) -> Any:
        """Load a pickled object by cache_id."""
        pkl_path = self._objects_dir / f"{cache_id}.pkl"
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
