"""Persistent disk cache for dtcc-core objects.

Stores pickled objects on disk with a JSON metadata index.
Supports spatial containment lookup for datasets and exact
hash lookup for builders. Thread-safe via a lock, and safe for several
processes sharing one directory via a file lock on the index.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
import pickle
import re
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



def _core_build() -> str:
    """The installed dtcc-core: the commit pip installed it from, as the
    contract workflow checks, else its version (an editable or local install,
    whose edits this cannot see)."""
    dist = importlib.metadata.distribution("dtcc-core")
    try:
        return json.loads(dist.read_text("direct_url.json") or "{}")["vcs_info"]["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return dist.version


# Every entry records what wrote it. Pickles of Core objects need the Core that
# made them, and entry fields change with this module, so an entry with another
# stamp is never served and cleanup removes it: after an upgrade the cache
# starts cold. Bump "schema" when the entry format or cached payloads change.
CACHE_STAMP = {"schema": 1, "core": _core_build()}

# Only downloads keyed by their bounds and parameters. Builder results are not
# cached until their keys are correct (U2, #11): see builder_calls.
CACHE_ALLOWLIST = frozenset({
    "datasets.point_cloud",
    "datasets.buildings",
})


def canonical_params_hash(operation: str, params: dict) -> str:
    """A stable hash of a dataset download's parameters. Bounds are excluded:
    a download is found by containment of its bounds instead."""
    canonical = {k: v for k, v in sorted(params.items()) if k != "bounds"}
    payload = json.dumps({"op": operation, "params": canonical},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class CacheDirError(RuntimeError):
    """The cache directory could hold pickles planted by another user."""


_CACHE_ID = re.compile(r"[0-9a-f]{8}")
_PRIVATE_FILE = 0o600


def _fresh(entry: Any, now: datetime) -> bool:
    """Whether an index entry may be served: written by this schema and Core,
    well formed, and younger than the TTL. Anything else is a cold miss."""
    try:
        age = now - datetime.fromisoformat(entry["timestamp"])
        return (entry.get("stamp") == CACHE_STAMP
                and bool(_CACHE_ID.fullmatch(entry["cache_id"]))
                and age.total_seconds() / 3600 <= CACHE_TTL_HOURS)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _others_can_write(st: os.stat_result) -> bool:
    return bool(st.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


def _refuse(path: Path, why: str) -> None:
    raise CacheDirError(
        f"Cache path {path} is {why}. The cache loads pickles, which run code, "
        "so nobody else may be able to change it: fix the permissions "
        "(chmod go-w) or set DTCC_AGENT_CACHE_DIR to a directory you own.")


def _check_owner(path: Path, st: os.stat_result, trusted: tuple[int, ...]) -> None:
    if hasattr(os, "getuid") and st.st_uid not in trusted:
        _refuse(path, "owned by another user")


def _private_dir(path: Path) -> Path:
    """Make ``path`` a directory only this user controls, or refuse it, and
    return its real path, which the cache then uses so a symlink swapped
    after this check cannot redirect it.

    Missing directories are created 0700 one at a time, so no process ever
    sees one open. Parents may be shared only the way /tmp is (sticky), else
    whoever can write one could swap the cache for their own. Every entry
    already in it must be private too, and none may be a symlink: the cache
    never makes one, and a link's target could sit where others can write.
    """
    missing = [path, *(a for a in path.parents if not a.exists())]
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
    uid = os.getuid() if hasattr(os, "getuid") else None
    real = path.resolve(strict=True)

    for parent in real.parents:
        st = parent.stat()
        _check_owner(parent, st, (0, uid))
        if _others_can_write(st) and not st.st_mode & stat.S_ISVTX:
            _refuse(parent, "writable by other users")

    st = real.stat()
    _check_owner(real, st, (uid,))
    if _others_can_write(st):
        _refuse(real, "writable by other users")
    for child in real.iterdir():
        try:
            st = child.lstat()
        except FileNotFoundError:
            continue  # removed by another process's cleanup or index write
        if stat.S_ISLNK(st.st_mode):
            _refuse(child, "a symlink")
        _check_owner(child, st, (uid,))
        if _others_can_write(st):
            _refuse(child, "writable by other users")
    return real


def _open_private(path: Path, flags: int):
    """Open for writing as 0600 whatever the umask."""
    fd = os.open(path, flags | os.O_WRONLY | os.O_CREAT, _PRIVATE_FILE)
    return os.fdopen(fd, "ab" if flags & os.O_APPEND else "wb")


class DiskCache:
    """Persistent disk cache with JSON index and pickle storage."""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self._lock = threading.Lock()
        self._cache_dir = cache_dir = _private_dir(cache_dir)
        self._objects_dir = _private_dir(cache_dir / "objects")
        self._index_path = cache_dir / "index.json"
        self._lock_path = cache_dir / "index.lock"

        self._index: list[dict[str, Any]] = []
        self._index_version: tuple[int, int, int] | None = None
        with self._lock:
            self._reload_if_changed()
        logger.info("Disk cache at %s: %d entries", cache_dir, len(self._index))

        # Clean up expired entries on startup
        self.cleanup()

    def _version(self) -> tuple[int, int, int] | None:
        try:
            st = self._index_path.stat()
        except FileNotFoundError:
            return None
        return st.st_ino, st.st_mtime_ns, st.st_size  # os.replace: new inode

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
        if not isinstance(self._index, list):
            logger.warning("Disk cache index %s is not a list; starting empty",
                           self._index_path)
            self._index = []
        self._index_version = version

    def _save_index(self) -> None:
        """Write index to disk atomically. Must be called with lock held."""
        tmp = self._index_path.with_suffix(f".{os.getpid()}.tmp")
        with _open_private(tmp, os.O_TRUNC) as f:
            f.write(json.dumps(self._index, indent=2).encode())
        os.replace(tmp, self._index_path)
        self._index_version = self._version()

    @contextmanager
    def _editing_index(self):
        """Read-modify-write the index under a lock every process honours, so
        one process never overwrites entries another has added."""
        with self._lock, _open_private(self._lock_path, os.O_APPEND) as lock_file:
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

        with _open_private(pkl_path, os.O_EXCL) as f:
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
            "stamp": CACHE_STAMP,
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
                if not _fresh(entry, now):
                    continue
                if entry.get("operation") != operation:
                    continue
                if entry.get("source") != source:
                    continue
                if entry.get("params_hash") != params_hash:
                    continue
                if entry.get("bounds") is None:
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

    def cleanup(self) -> int:
        """Remove entries that may not be served (expired, another stamp,
        malformed) and enforce the disk budget. Returns count removed."""
        now = datetime.now()
        removed = 0

        with self._editing_index():
            surviving = []
            for entry in self._index:
                if _fresh(entry, now):
                    surviving.append(entry)
                    continue
                cache_id = entry.get("cache_id") if isinstance(entry, dict) else None
                if isinstance(cache_id, str) and _CACHE_ID.fullmatch(cache_id):
                    (self._objects_dir / f"{cache_id}.pkl").unlink(missing_ok=True)
                removed += 1

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
        if not _CACHE_ID.fullmatch(cache_id):
            raise ValueError(f"Not a cache id: {cache_id!r}")
        pkl_path = self._objects_dir / f"{cache_id}.pkl"
        fd = os.open(pkl_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as f:
            return pickle.load(f)
