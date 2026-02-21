# Persistent Disk Cache for Downloads & Builders

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Avoid redundant dataset downloads and deterministic builder re-computation by persisting results to disk with spatial containment-based lookup for datasets and provenance-hash lookup for builders.

**Architecture:** A `DiskCache` class in `dtcc_agent/disk_cache.py` stores pickled objects on disk alongside a JSON metadata index. The dispatcher checks the cache before running any allowlisted operation. Dataset lookups use spatial containment (cached bounds fully contain requested bounds) with cropping to the requested area. Builder lookups use exact provenance hashing (operation + content fingerprints of inputs + other params).

**Tech Stack:** pickle (serialization), JSON (index), hashlib (cache keys), threading.Lock (concurrency)

---

## Retained from prior plan

- Allowlist approach (safe, incremental)
- Observability/logging (hit/miss/expired/not-eligible)
- `cache_hit` boolean in dispatcher responses
- Config flags for enable/disable
- Regression test requirements

## Dropped from prior plan

- Layer A (LLM answer cache) — low hit rate, not the user's priority
- In-memory-only storage — defeats the purpose across restarts
- 30-minute TTL — too short for downloaded geodata

## Changed from prior plan

| Aspect | Prior plan | This plan |
|--------|-----------|-----------|
| Storage | In-memory dict, process lifetime | Persistent disk (pickle + JSON index) |
| Dataset matching | Exact parameter hash | Spatial containment (cached bounds contain requested) |
| Cropping | Not mentioned | Crop cached data to requested bounds |
| TTL | 30 minutes | 7 days |
| Disk budget | None | 10 GB configurable |
| Builder key | Underspecified fingerprinting | Content fingerprint from ObjectStore metadata |

---

## Storage Layout

```
/tmp/dtcc_cache/
├── index.json              # metadata index, loaded into memory on startup
└── objects/
    ├── a1b2c3d4.pkl        # pickled dtcc-core objects
    ├── e5f6g7h8.pkl
    └── ...
```

### Index Entry Schema

```python
{
    "cache_id": "a1b2c3d4",
    "operation": "datasets.point_cloud",
    "category": "datasets",            # "datasets" or "builder"
    "params_hash": "fa3b...",          # SHA-256 of canonical non-bounds params
    "bounds": [319700, 6399500, 320200, 6400000],  # EPSG:3006, null for non-spatial
    "source": "LM",                    # dataset source, null for builders
    "timestamp": "2026-02-21T14:30:00",
    "size_bytes": 52428800,
    "object_type": "PointCloud",
}
```

### Cache Key Logic

**Datasets:** match on `operation` + `source` + non-bounds params hash + bounds containment (cached bounds fully contain requested bounds).

**Builders:** match on `operation` + `params_hash` where object-ref params are replaced by their content fingerprints. Content fingerprint = SHA-256 of (object_type + source_op + bounds + point/vertex count). For objects loaded from disk cache, reuse the `cache_id` as fingerprint.

---

## Allowlist (v1)

Dataset ops:
- `datasets.point_cloud`
- `datasets.buildings`

Deterministic builder ops:
- `builder.build_terrain_raster`
- `builder.build_terrain_surface_mesh`
- `builder.build_city_surface_mesh`
- `builder.raster.slope_aspect`
- `builder.pc_filter.classification_filter`

---

## Constants

```python
CACHE_DIR = Path("/tmp/dtcc_cache")
CACHE_TTL_HOURS = 168            # 7 days
CACHE_MAX_SIZE_GB = 10           # total disk budget
CACHE_ALLOWLIST = {
    "datasets.point_cloud",
    "datasets.buildings",
    "builder.build_terrain_raster",
    "builder.build_terrain_surface_mesh",
    "builder.build_city_surface_mesh",
    "builder.raster.slope_aspect",
    "builder.pc_filter.classification_filter",
}
```

---

## Implementation Tasks

### Task 1: DiskCache core — store and index

**Files:**
- Create: `dtcc_agent/disk_cache.py`
- Test: `tests/test_disk_cache.py` (create)

**Step 1: Write failing test**

```python
# tests/test_disk_cache.py
"""Tests for persistent disk cache."""

import pickle
import tempfile
from pathlib import Path

from dtcc_agent.disk_cache import DiskCache


def test_store_creates_pickle_and_index_entry():
    with tempfile.TemporaryDirectory() as td:
        cache = DiskCache(cache_dir=Path(td))
        obj = {"fake": "pointcloud", "points": [1, 2, 3]}

        cache_id = cache.store(
            obj=obj,
            operation="datasets.point_cloud",
            category="datasets",
            params_hash="abc123",
            bounds=[319700, 6399500, 320200, 6400000],
            source="LM",
            object_type="PointCloud",
        )

        # Pickle file exists
        pkl_path = Path(td) / "objects" / f"{cache_id}.pkl"
        assert pkl_path.exists()

        # Index has one entry
        assert len(cache._index) == 1
        entry = cache._index[0]
        assert entry["cache_id"] == cache_id
        assert entry["operation"] == "datasets.point_cloud"
        assert entry["bounds"] == [319700, 6399500, 320200, 6400000]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_disk_cache.py::test_store_creates_pickle_and_index_entry -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtcc_agent.disk_cache'`

**Step 3: Write minimal implementation**

```python
# dtcc_agent/disk_cache.py
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

    def load(self, cache_id: str) -> Any:
        """Load a pickled object by cache_id."""
        pkl_path = self._objects_dir / f"{cache_id}.pkl"
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_disk_cache.py::test_store_creates_pickle_and_index_entry -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dtcc_agent/disk_cache.py tests/test_disk_cache.py
git commit -m "feat: add DiskCache core with store, load, and JSON index"
```

---

### Task 2: Dataset lookup with containment check

**Files:**
- Modify: `dtcc_agent/disk_cache.py`
- Test: `tests/test_disk_cache.py`

**Step 1: Write failing tests**

```python
def test_dataset_lookup_hit_on_exact_bounds():
    with tempfile.TemporaryDirectory() as td:
        cache = DiskCache(cache_dir=Path(td))
        cache.store(
            obj="fake_pointcloud",
            operation="datasets.point_cloud",
            category="datasets",
            params_hash="src-LM",
            bounds=[319700, 6399500, 320200, 6400000],
            source="LM",
            object_type="PointCloud",
        )
        result = cache.dataset_lookup(
            operation="datasets.point_cloud",
            source="LM",
            params_hash="src-LM",
            requested_bounds=[319700, 6399500, 320200, 6400000],
        )
        assert result is not None
        cache_id, cached_bounds = result
        assert cached_bounds == [319700, 6399500, 320200, 6400000]


def test_dataset_lookup_hit_on_containing_bounds():
    with tempfile.TemporaryDirectory() as td:
        cache = DiskCache(cache_dir=Path(td))
        # Cache a larger area
        cache.store(
            obj="fake_pointcloud",
            operation="datasets.point_cloud",
            category="datasets",
            params_hash="src-LM",
            bounds=[319000, 6399000, 321000, 6401000],
            source="LM",
            object_type="PointCloud",
        )
        # Request a smaller area within it
        result = cache.dataset_lookup(
            operation="datasets.point_cloud",
            source="LM",
            params_hash="src-LM",
            requested_bounds=[319700, 6399500, 320200, 6400000],
        )
        assert result is not None


def test_dataset_lookup_miss_on_non_containing_bounds():
    with tempfile.TemporaryDirectory() as td:
        cache = DiskCache(cache_dir=Path(td))
        cache.store(
            obj="fake_pointcloud",
            operation="datasets.point_cloud",
            category="datasets",
            params_hash="src-LM",
            bounds=[319700, 6399500, 320200, 6400000],
            source="LM",
            object_type="PointCloud",
        )
        # Request a different area
        result = cache.dataset_lookup(
            operation="datasets.point_cloud",
            source="LM",
            params_hash="src-LM",
            requested_bounds=[330000, 6410000, 330500, 6410500],
        )
        assert result is None


def test_dataset_lookup_miss_on_different_source():
    with tempfile.TemporaryDirectory() as td:
        cache = DiskCache(cache_dir=Path(td))
        cache.store(
            obj="fake_pointcloud",
            operation="datasets.point_cloud",
            category="datasets",
            params_hash="src-LM",
            bounds=[319700, 6399500, 320200, 6400000],
            source="LM",
            object_type="PointCloud",
        )
        result = cache.dataset_lookup(
            operation="datasets.point_cloud",
            source="OSM",
            params_hash="src-OSM",
            requested_bounds=[319700, 6399500, 320200, 6400000],
        )
        assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_disk_cache.py -v -k "dataset_lookup"`
Expected: FAIL — `AttributeError: 'DiskCache' object has no attribute 'dataset_lookup'`

**Step 3: Write implementation**

Add to `DiskCache`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_disk_cache.py -v -k "dataset_lookup"`
Expected: All PASS

**Step 5: Commit**

```bash
git add dtcc_agent/disk_cache.py tests/test_disk_cache.py
git commit -m "feat: add dataset_lookup with spatial containment check"
```

---

### Task 3: Builder lookup with exact hash

**Files:**
- Modify: `dtcc_agent/disk_cache.py`
- Test: `tests/test_disk_cache.py`

**Step 1: Write failing tests**

```python
def test_builder_lookup_hit_on_matching_hash():
    with tempfile.TemporaryDirectory() as td:
        cache = DiskCache(cache_dir=Path(td))
        cache.store(
            obj="fake_raster",
            operation="builder.build_terrain_raster",
            category="builder",
            params_hash="hash-abc",
            object_type="Raster",
        )
        result = cache.builder_lookup(
            operation="builder.build_terrain_raster",
            params_hash="hash-abc",
        )
        assert result is not None


def test_builder_lookup_miss_on_different_hash():
    with tempfile.TemporaryDirectory() as td:
        cache = DiskCache(cache_dir=Path(td))
        cache.store(
            obj="fake_raster",
            operation="builder.build_terrain_raster",
            category="builder",
            params_hash="hash-abc",
            object_type="Raster",
        )
        result = cache.builder_lookup(
            operation="builder.build_terrain_raster",
            params_hash="hash-xyz",
        )
        assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_disk_cache.py -v -k "builder_lookup"`
Expected: FAIL — `AttributeError`

**Step 3: Write implementation**

Add to `DiskCache`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_disk_cache.py -v -k "builder_lookup"`
Expected: All PASS

**Step 5: Commit**

```bash
git add dtcc_agent/disk_cache.py tests/test_disk_cache.py
git commit -m "feat: add builder_lookup with exact hash matching"
```

---

### Task 4: TTL expiry and disk budget eviction

**Files:**
- Modify: `dtcc_agent/disk_cache.py`
- Test: `tests/test_disk_cache.py`

**Step 1: Write failing tests**

```python
from unittest.mock import patch
from datetime import timedelta


def test_dataset_lookup_miss_when_expired():
    with tempfile.TemporaryDirectory() as td:
        cache = DiskCache(cache_dir=Path(td))
        cache.store(
            obj="fake",
            operation="datasets.point_cloud",
            category="datasets",
            params_hash="src-LM",
            bounds=[319700, 6399500, 320200, 6400000],
            source="LM",
            object_type="PointCloud",
        )
        future = datetime.now() + timedelta(hours=169)
        with patch("dtcc_agent.disk_cache.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.fromisoformat = datetime.fromisoformat
            result = cache.dataset_lookup(
                operation="datasets.point_cloud",
                source="LM",
                params_hash="src-LM",
                requested_bounds=[319700, 6399500, 320200, 6400000],
            )
        assert result is None


def test_cleanup_removes_expired_entries():
    with tempfile.TemporaryDirectory() as td:
        cache = DiskCache(cache_dir=Path(td))
        cache.store(
            obj="old_data",
            operation="datasets.point_cloud",
            category="datasets",
            params_hash="src-LM",
            bounds=[319700, 6399500, 320200, 6400000],
            source="LM",
            object_type="PointCloud",
        )
        assert len(cache._index) == 1

        future = datetime.now() + timedelta(hours=169)
        with patch("dtcc_agent.disk_cache.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.fromisoformat = datetime.fromisoformat
            removed = cache.cleanup()

        assert removed == 1
        assert len(cache._index) == 0
```

**Step 2: Run tests to verify the cleanup test fails**

Run: `python -m pytest tests/test_disk_cache.py -v -k "cleanup"`
Expected: FAIL — `AttributeError: 'DiskCache' has no attribute 'cleanup'`

**Step 3: Write implementation**

Add to `DiskCache`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_disk_cache.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add dtcc_agent/disk_cache.py tests/test_disk_cache.py
git commit -m "feat: add cleanup with TTL expiry and disk budget eviction"
```

---

### Task 5: Content fingerprinting helper

**Files:**
- Modify: `dtcc_agent/disk_cache.py`
- Test: `tests/test_disk_cache.py`

**Step 1: Write failing test**

```python
from dtcc_agent.disk_cache import content_fingerprint


def test_content_fingerprint_stable_for_same_metadata():
    meta_a = {"type": "PointCloud", "source_op": "datasets.point_cloud",
              "nbytes": 1000, "label": "test"}
    meta_b = {"type": "PointCloud", "source_op": "datasets.point_cloud",
              "nbytes": 1000, "label": "test"}
    assert content_fingerprint(meta_a) == content_fingerprint(meta_b)


def test_content_fingerprint_differs_for_different_metadata():
    meta_a = {"type": "PointCloud", "source_op": "datasets.point_cloud",
              "nbytes": 1000, "label": "area_a"}
    meta_b = {"type": "PointCloud", "source_op": "datasets.point_cloud",
              "nbytes": 2000, "label": "area_b"}
    assert content_fingerprint(meta_a) != content_fingerprint(meta_b)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_disk_cache.py -v -k "fingerprint"`
Expected: FAIL — `ImportError`

**Step 3: Write implementation**

Add as a module-level function in `disk_cache.py`:

```python
def content_fingerprint(obj_metadata: dict[str, Any]) -> str:
    """Compute a stable fingerprint from ObjectStore metadata.

    Used to generate cache keys for builder operations where
    input objects are referenced by transient IDs.
    """
    # Use type + source_op + size + label as fingerprint inputs
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
    params: dict[str, Any],
    object_fingerprints: dict[str, str] | None = None,
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_disk_cache.py -v -k "fingerprint"`
Expected: All PASS

**Step 5: Commit**

```bash
git add dtcc_agent/disk_cache.py tests/test_disk_cache.py
git commit -m "feat: add content_fingerprint and canonical_params_hash helpers"
```

---

### Task 6: Spatial cropping utilities

**Files:**
- Create: `dtcc_agent/crop.py`
- Test: `tests/test_crop.py` (create)

**Step 1: Write failing test**

```python
# tests/test_crop.py
"""Tests for spatial cropping of cached objects."""

import numpy as np
import pytest

from dtcc_agent.crop import crop_to_bounds


def test_crop_pointcloud_filters_points():
    """PointCloud-like object: only points within bounds are kept."""
    class FakePC:
        def __init__(self):
            self.points = np.array([
                [100.0, 200.0, 5.0],
                [150.0, 250.0, 6.0],
                [300.0, 400.0, 7.0],  # outside
            ])
            self.classification = np.array([1, 2, 3])

    pc = FakePC()
    cropped = crop_to_bounds(pc, [90, 190, 200, 300])

    assert len(cropped.points) == 2
    assert len(cropped.classification) == 2
    assert cropped.points[0][0] == 100.0


def test_crop_returns_original_if_unknown_type():
    """Unknown types are returned as-is (no cropping)."""
    obj = {"data": 123}
    result = crop_to_bounds(obj, [0, 0, 100, 100])
    assert result is obj
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_crop.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# dtcc_agent/crop.py
"""Spatial cropping for cached dtcc-core objects."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def crop_to_bounds(obj: Any, bounds: list[float]) -> Any:
    """Crop an object to the given [xmin, ymin, xmax, ymax] bounds.

    Supports PointCloud (filters points array) and returns unknown
    types unchanged.
    """
    type_name = type(obj).__name__

    if type_name == "PointCloud" and hasattr(obj, "points"):
        return _crop_pointcloud(obj, bounds)

    if type_name == "City" and hasattr(obj, "buildings"):
        return _crop_city(obj, bounds)

    # Unknown type — return as-is
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
        # Use footprint centroid if available
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_crop.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add dtcc_agent/crop.py tests/test_crop.py
git commit -m "feat: add spatial cropping for PointCloud and City"
```

---

### Task 7: Dispatcher integration

**Files:**
- Modify: `dtcc_agent/dispatcher.py`
- Modify: `dtcc_agent/server.py` (pass cache to dispatcher)
- Test: `tests/test_dispatcher.py` (extend)

**Step 1: Write failing test**

```python
# Append to tests/test_dispatcher.py
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from dtcc_agent.disk_cache import DiskCache
from dtcc_agent.object_store import ObjectStore


def test_dataset_cache_hit_skips_download():
    """Cached dataset result is loaded from disk instead of re-downloading."""
    with tempfile.TemporaryDirectory() as td:
        cache = DiskCache(cache_dir=Path(td))
        store = ObjectStore()

        # Pre-populate cache with a fake result
        import numpy as np
        fake_pc = MagicMock()
        fake_pc.__class__.__name__ = "PointCloud"
        fake_pc.points = np.array([[319800, 6399600, 5.0]])
        fake_pc.classification = np.array([1])

        cache.store(
            obj=fake_pc,
            operation="datasets.point_cloud",
            category="datasets",
            params_hash=cache.canonical_params_hash(
                "datasets.point_cloud", {"source": "LM"}
            ),
            bounds=[319700, 6399500, 320200, 6400000],
            source="LM",
            object_type="PointCloud",
        )

        # Attempt run_operation — should hit cache, not call dataset
        with patch("dtcc_agent.dispatcher.get_operation") as mock_get_op:
            mock_op = MagicMock()
            mock_op.category = "datasets"
            mock_op.name = "datasets.point_cloud"
            mock_get_op.return_value = mock_op

            result = run_operation(
                "datasets.point_cloud",
                {"bounds": [319700, 6399500, 320200, 6400000], "source": "LM"},
                store,
                cache=cache,
            )

        assert "error" not in result
        assert result.get("cache_hit") is True
        # The dataset callable should NOT have been called
        mock_op._callable.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dispatcher.py::test_dataset_cache_hit_skips_download -v`
Expected: FAIL — `run_operation() got unexpected keyword argument 'cache'`

**Step 3: Modify dispatcher**

In `dtcc_agent/dispatcher.py`, update `run_operation` signature and add cache logic:

```python
from .disk_cache import (
    DiskCache, CACHE_ALLOWLIST,
    content_fingerprint, canonical_params_hash,
)
from .crop import crop_to_bounds


def run_operation(
    name: str,
    params: dict[str, Any] | None,
    store: ObjectStore,
    label: str = "",
    cache: DiskCache | None = None,
) -> dict[str, Any]:
    params = params or {}
    try:
        op = get_operation(name)
    except KeyError as exc:
        return {"error": str(exc)}

    # --- Disk cache check ---
    if cache and name in CACHE_ALLOWLIST:
        hit = _check_cache(name, op.category, params, store, cache)
        if hit is not None:
            hit["cache_hit"] = True
            return hit

    # Datasets are special
    if op.category == "datasets":
        result = _run_dataset(op, params, store, label)
    else:
        result = _run_function(op, params, store, label)

    # --- Store in disk cache on success ---
    if cache and name in CACHE_ALLOWLIST and "error" not in result:
        _populate_cache(name, op.category, params, result, store, cache)

    return result
```

Add the cache helper functions:

```python
def _check_cache(
    name: str,
    category: str,
    params: dict[str, Any],
    store: ObjectStore,
    cache: DiskCache,
) -> dict[str, Any] | None:
    """Check disk cache. Returns dispatcher-format response or None."""
    if category == "datasets":
        bounds = params.get("bounds")
        if not bounds:
            return None
        source = params.get("source", "LM")
        non_bounds = {k: v for k, v in params.items() if k != "bounds"}
        ph = canonical_params_hash(name, non_bounds)
        result = cache.dataset_lookup(name, source, ph, bounds)
        if result is None:
            return None
        cache_id, cached_bounds = result
        obj = cache.load(cache_id)
        # Crop if cached bounds are larger than requested
        if cached_bounds != bounds:
            obj = crop_to_bounds(obj, bounds)
        obj_id = store.store(obj, source_op=name, label="(cached)")
        return {
            "operation": name,
            "result_id": obj_id,
            "label": "(cached)",
            "summary": serialize(obj),
        }
    else:
        # Builder: compute fingerprints for object-ref inputs
        fingerprints = _compute_fingerprints(params, store)
        ph = canonical_params_hash(name, params, fingerprints)
        cache_id = cache.builder_lookup(name, ph)
        if cache_id is None:
            return None
        obj = cache.load(cache_id)
        obj_id = store.store(obj, source_op=name, label="(cached)")
        return {
            "operation": name,
            "result_id": obj_id,
            "label": "(cached)",
            "summary": serialize(obj),
        }


def _compute_fingerprints(
    params: dict[str, Any],
    store: ObjectStore,
) -> dict[str, str]:
    """Compute content fingerprints for object-ref params."""
    fingerprints = {}
    for key, value in params.items():
        if isinstance(value, str) and value in store:
            meta = None
            for entry in store.list(limit=500):
                if entry["id"] == value:
                    meta = entry
                    break
            if meta:
                fingerprints[key] = content_fingerprint(meta)
    return fingerprints


def _populate_cache(
    name: str,
    category: str,
    params: dict[str, Any],
    result: dict[str, Any],
    store: ObjectStore,
    cache: DiskCache,
) -> None:
    """Store a successful operation result in the disk cache."""
    result_id = result.get("result_id")
    if not result_id:
        return  # tuple/primitive results — skip for v1
    try:
        obj = store.get(result_id)
    except KeyError:
        return

    if category == "datasets":
        bounds = params.get("bounds")
        source = params.get("source", "LM")
        non_bounds = {k: v for k, v in params.items() if k != "bounds"}
        ph = canonical_params_hash(name, non_bounds)
        bounds_list = list(bounds) if bounds else None
    else:
        fingerprints = _compute_fingerprints(params, store)
        ph = canonical_params_hash(name, params, fingerprints)
        bounds_list = None
        source = None

    cache.store(
        obj=obj,
        operation=name,
        category=category,
        params_hash=ph,
        bounds=bounds_list,
        source=source,
        object_type=type(obj).__name__,
    )
```

**Step 4: Update `server.py` to pass cache to dispatcher**

In `dtcc_agent/server.py`, add near the top after `_object_store`:

```python
from .disk_cache import DiskCache

_disk_cache = DiskCache()
```

Then in every call to `run_operation(...)` in `server.py`, add `cache=_disk_cache`:

```python
# In the run_operation MCP tool handler:
result = dispatcher.run_operation(name, parameters, _object_store,
                                  label=label or "", cache=_disk_cache)
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dispatcher.py -v`
Expected: All PASS (both new and existing tests)

**Step 6: Commit**

```bash
git add dtcc_agent/dispatcher.py dtcc_agent/server.py tests/test_dispatcher.py
git commit -m "feat: integrate disk cache into dispatcher and server"
```

---

### Task 8: Cache reload on startup

**Files:**
- Modify: `dtcc_agent/disk_cache.py`
- Test: `tests/test_disk_cache.py`

**Step 1: Write failing test**

```python
def test_cache_survives_restart():
    with tempfile.TemporaryDirectory() as td:
        # First instance stores data
        cache1 = DiskCache(cache_dir=Path(td))
        cache1.store(
            obj="persistent_data",
            operation="datasets.point_cloud",
            category="datasets",
            params_hash="src-LM",
            bounds=[319700, 6399500, 320200, 6400000],
            source="LM",
            object_type="PointCloud",
        )

        # Second instance (simulating restart) loads index
        cache2 = DiskCache(cache_dir=Path(td))
        assert len(cache2._index) == 1
        result = cache2.dataset_lookup(
            operation="datasets.point_cloud",
            source="LM",
            params_hash="src-LM",
            requested_bounds=[319700, 6399500, 320200, 6400000],
        )
        assert result is not None
        obj = cache2.load(result[0])
        assert obj == "persistent_data"
```

**Step 2: Run test (should pass — index loading is already in `__init__`)**

Run: `python -m pytest tests/test_disk_cache.py::test_cache_survives_restart -v`
Expected: PASS (this tests existing behavior)

**Step 3: Commit**

```bash
git add tests/test_disk_cache.py
git commit -m "test: verify disk cache persistence across restarts"
```

---

### Task 9: Observability logging

**Files:**
- Modify: `dtcc_agent/dispatcher.py`

All logging is already included in the implementation (Task 7). Verify the logs exist:

- `disk_cache.py`: logs cache hit/miss at INFO/DEBUG level
- `dispatcher.py`: `_check_cache` returns `cache_hit: True` in response
- `crop.py`: logs cropping statistics

**Step 1: Review log output**

Run: `python -m pytest tests/ -v --log-cli-level=DEBUG -k "cache" 2>&1 | grep -i "cache"`
Expected: See hit/miss/crop log lines

**Step 2: Commit (if any logging tweaks needed)**

```bash
git add dtcc_agent/dispatcher.py
git commit -m "chore: finalize cache observability logging"
```

---

### Task 10: Final verification

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 2: Run cleanup on startup**

Add to `DiskCache.__init__` after loading the index:

```python
self.cleanup()  # Remove expired entries on startup
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete persistent disk cache for downloads and builders"
```

---

## Observability Summary

| Event | Log level | Example |
|-------|-----------|---------|
| Cache hit (dataset) | INFO | `Disk cache hit: datasets.point_cloud, cache_id=a1b2c3d4` |
| Cache hit (builder) | INFO | `Disk cache hit: builder.build_terrain_raster, cache_id=e5f6g7h8` |
| Cache miss | DEBUG | `Disk cache miss: datasets.point_cloud (no containing bounds)` |
| Cache store | INFO | `Cached datasets.point_cloud result a1b2c3d4 (52.4 MB)` |
| Cropping | INFO | `Cropped PointCloud: 1000000 → 450000 points` |
| Cleanup | INFO | `Disk cache cleanup: removed 3 entries` |
| Response flag | — | `"cache_hit": true` in dispatcher response dict |

---

## Risks and Mitigations

- **Stale data:** 7-day TTL + deterministic allowlist only. Non-deterministic ops excluded.
- **Pickle security:** Cache dir is `/tmp/dtcc_cache`, same trust boundary as the process.
- **Disk space:** 10 GB budget with oldest-first eviction on `cleanup()`.
- **Large pickles:** Point clouds can be 100s of MB. The disk budget and cleanup handle this.
- **Thread safety:** All index operations under `threading.Lock`.
- **Fingerprint collisions:** Conservative fingerprint uses type + source_op + nbytes + label. Low collision risk for real usage.

---

## Acceptance Criteria

- Point cloud download is skipped on cache hit for same or containing bounds.
- Cached point cloud is cropped to requested bounds.
- Builder results are cached and replayed for equivalent inputs.
- Cache survives process restart.
- Expired entries are cleaned up on startup.
- Disk usage stays within budget.
- All existing and new tests pass.
- `cache_hit: true` flag present in cached responses.
