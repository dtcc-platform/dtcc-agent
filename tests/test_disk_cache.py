"""Tests for persistent disk cache."""

import json
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

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


# --- Content fingerprinting helpers ---

from dtcc_agent.disk_cache import content_fingerprint, canonical_params_hash


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


def test_canonical_params_hash_stable():
    h1 = canonical_params_hash("builder.build_terrain_raster",
                                {"cell_size": 2.0, "ground_only": True})
    h2 = canonical_params_hash("builder.build_terrain_raster",
                                {"ground_only": True, "cell_size": 2.0})
    assert h1 == h2


def test_canonical_params_hash_replaces_fingerprints():
    h1 = canonical_params_hash("builder.build_terrain_raster",
                                {"pc": "obj-id-1", "cell_size": 2.0},
                                {"pc": "fp-abc"})
    h2 = canonical_params_hash("builder.build_terrain_raster",
                                {"pc": "obj-id-2", "cell_size": 2.0},
                                {"pc": "fp-abc"})
    # Same fingerprint, different obj IDs — should produce same hash
    assert h1 == h2


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


# --- get_buildings cache path ---


def test_get_buildings_answers_for_a_smaller_area_inside_a_cached_one(monkeypatch, tmp_path):
    """A containing cache hit is cropped to the requested area before it is
    summarised, not returned whole with its bounds relabelled (#39)."""
    import numpy as np
    from dtcc_core.datasets.buildings import BuildingCollection
    from dtcc_core.model import Building, GeometryType, Surface

    import dtcc_agent.runner as runner
    import dtcc_agent.server as server

    def building(x, y, height):
        b = Building()
        square = [[x, y, 0], [x + 10, y, 0], [x + 10, y + 10, 0], [x, y + 10, 0]]
        b.add_geometry(Surface(vertices=np.array(square, float)), GeometryType.LOD0)
        b.height = height
        return b

    downloads = []

    def fetch(**kwargs):
        downloads.append(kwargs)
        return BuildingCollection([building(0, 0, 10.0), building(500, 500, 30.0)])

    monkeypatch.setattr(runner, "fetch_buildings", fetch)
    monkeypatch.setattr(server, "_disk_cache", DiskCache(cache_dir=tmp_path))

    whole = json.loads(server.get_buildings(bounds=[-100, -100, 600, 600]))
    part = json.loads(server.get_buildings(bounds=[-50, -50, 100, 100], max_buildings=5))

    assert whole["num_buildings"] == 2
    assert part["num_buildings"] == 1
    assert part["bounds"] == [-50, -50, 100, 100]
    assert part["height_stats"]["max_m"] == 10.0
    assert len(part["buildings"]) == 1
    assert len(downloads) == 1  # the sub-area and another max_buildings reuse the download
