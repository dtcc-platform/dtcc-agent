"""Tests for persistent disk cache."""

import json
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

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
    from dtcc_core.datasets.buildings import BuildingCollection

    import dtcc_agent.runner as runner
    import dtcc_agent.server as server

    downloads = []

    def fetch(**kwargs):
        downloads.append(kwargs)
        return BuildingCollection([_core_building(0, 0, 10.0), _core_building(500, 500, 30.0)])

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

    other_source = json.loads(server.get_buildings(bounds=[-50, -50, 100, 100], source="OSM"))
    assert other_source["source"] == "OSM"
    assert [d["source"] for d in downloads] == ["LM", "OSM"]


def _core_building(x, y, height):
    import numpy as np
    from dtcc_core.model import Building, GeometryType, Surface

    b = Building()
    square = [[x, y, 0], [x + 10, y, 0], [x + 10, y + 10, 0], [x, y + 10, 0]]
    b.add_geometry(Surface(vertices=np.array(square, float)), GeometryType.LOD0)
    b.height = height
    return b


def _isolated_get_buildings(monkeypatch, tmp_path, fetch):
    import dtcc_agent.runner as runner
    import dtcc_agent.server as server

    cache = DiskCache(cache_dir=tmp_path)
    monkeypatch.setattr(runner, "fetch_buildings", fetch)
    monkeypatch.setattr(server, "_disk_cache", cache)
    return server, cache


def test_get_buildings_answers_exact_bounds_from_cache_without_cropping(monkeypatch, tmp_path):
    from dtcc_core.datasets.buildings import BuildingCollection

    import dtcc_agent.crop as crop

    downloads = []

    def fetch(**kwargs):
        downloads.append(kwargs)
        return BuildingCollection([_core_building(0, 0, 10.0), _core_building(20, 20, 20.0)])

    server, _ = _isolated_get_buildings(monkeypatch, tmp_path, fetch)
    server.get_buildings(bounds=[-100, -100, 100, 100])

    crops = []
    monkeypatch.setattr(crop, "crop_to_bounds", lambda obj, b: crops.append(b) or obj)
    again = json.loads(server.get_buildings(bounds=[-100, -100, 100, 100], max_buildings=1))

    assert len(downloads) == 1 and crops == []
    assert again["num_buildings"] == 2
    assert len(again["buildings"]) == 1 and again["truncated"] is True


def test_get_buildings_reports_a_failed_download_and_caches_nothing(monkeypatch, tmp_path):
    def fetch(**kwargs):
        raise RuntimeError("LM unreachable")

    server, cache = _isolated_get_buildings(monkeypatch, tmp_path, fetch)
    result = json.loads(server.get_buildings(bounds=[0, 0, 100, 100]))

    assert result == {"error": "Failed to fetch buildings: LM unreachable"}
    assert cache.dataset_lookup(
        "datasets.buildings", "LM",
        canonical_params_hash("datasets.buildings", {"source": "LM"}),
        [0, 0, 100, 100],
    ) is None


def test_get_buildings_answers_when_the_cache_cannot_be_written(monkeypatch, tmp_path):
    from dtcc_core.datasets.buildings import BuildingCollection

    server, cache = _isolated_get_buildings(
        monkeypatch, tmp_path, lambda **kw: BuildingCollection([_core_building(0, 0, 10.0)]))

    def broken_store(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cache, "store", broken_store)
    result = json.loads(server.get_buildings(bounds=[-50, -50, 50, 50]))

    assert result["num_buildings"] == 1


def test_get_buildings_downloads_again_when_a_cached_entry_cannot_be_read(monkeypatch, tmp_path):
    from dtcc_core.datasets.buildings import BuildingCollection

    downloads = []

    def fetch(**kwargs):
        downloads.append(kwargs)
        return BuildingCollection([_core_building(0, 0, 10.0)])

    server, cache = _isolated_get_buildings(monkeypatch, tmp_path, fetch)
    server.get_buildings(bounds=[-50, -50, 50, 50])

    def unreadable(cache_id):
        raise pickle.UnpicklingError("corrupt")

    monkeypatch.setattr(cache, "load", unreadable)
    result = json.loads(server.get_buildings(bounds=[-50, -50, 50, 50]))

    assert len(downloads) == 2
    assert result["num_buildings"] == 1


def test_get_buildings_summarises_an_empty_area(monkeypatch, tmp_path):
    from dtcc_core.datasets.buildings import BuildingCollection

    server, _ = _isolated_get_buildings(monkeypatch, tmp_path, lambda **kw: BuildingCollection([]))
    result = json.loads(server.get_buildings(bounds=[0, 0, 10, 10]))

    assert result["num_buildings"] == 0
    assert result["buildings"] == [] and result["truncated"] is False
    assert result["height_stats"]["max_m"] is None
    assert result["total_footprint_area_m2"] == 0


def test_get_buildings_reuses_a_download_the_dispatcher_cached(monkeypatch, tmp_path):
    """run_operation("datasets.buildings") usually leaves source out; its
    entry must still serve get_buildings, which always names it."""
    from dtcc_core.datasets.buildings import BuildingCollection

    import dtcc_agent.dispatcher as dispatcher
    from dtcc_agent.object_store import ObjectStore

    def fetch(**kwargs):
        raise AssertionError("should answer from the dispatcher's cache entry")

    server, cache = _isolated_get_buildings(monkeypatch, tmp_path, fetch)
    store = ObjectStore()
    downloaded = BuildingCollection([_core_building(0, 0, 10.0), _core_building(500, 500, 30.0)])
    result_id = store.store(downloaded, source_op="datasets.buildings")
    dispatcher._populate_cache(
        "datasets.buildings", "datasets", {"bounds": [-100, -100, 600, 600]},
        {"result_id": result_id}, store, cache,
    )

    result = json.loads(server.get_buildings(bounds=[-50, -50, 100, 100]))

    assert result["num_buildings"] == 1


def test_get_buildings_downloads_again_when_a_larger_cached_area_cannot_be_cropped(monkeypatch, tmp_path):
    """A plain list (an older Core's return type) has no crop. Reusing it
    whole would answer for the cached area, so it counts as a miss (#39)."""
    from dtcc_core.datasets.buildings import BuildingCollection

    from dtcc_agent.dispatcher import store_dataset

    downloads = []

    def fetch(**kwargs):
        downloads.append(kwargs)
        return BuildingCollection([_core_building(0, 0, 10.0)])

    server, cache = _isolated_get_buildings(monkeypatch, tmp_path, fetch)
    store_dataset(
        "datasets.buildings", {"bounds": [-100, -100, 600, 600], "source": "LM"},
        [_core_building(0, 0, 10.0), _core_building(500, 500, 30.0)], cache,
    )

    result = json.loads(server.get_buildings(bounds=[-50, -50, 100, 100]))

    assert result["num_buildings"] == 1
    assert len(downloads) == 1


# --- where the cache lives, and who may write it ---


def test_the_default_cache_dir_is_per_user_not_shared_tmp(monkeypatch, tmp_path):
    import importlib

    import dtcc_agent.disk_cache as disk_cache

    monkeypatch.delenv("DTCC_AGENT_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    try:
        assert importlib.reload(disk_cache).CACHE_DIR == tmp_path / "xdg" / "dtcc_agent"
    finally:
        monkeypatch.undo()
        importlib.reload(disk_cache)


def test_a_new_cache_dir_is_private(tmp_path):
    import stat

    DiskCache(cache_dir=tmp_path / "fresh")

    assert stat.S_IMODE((tmp_path / "fresh").stat().st_mode) == 0o700


def test_the_cache_refuses_a_dir_others_can_write(tmp_path):
    """Loading a pickle runs code; anyone who can write the dir could plant one."""
    from dtcc_agent.disk_cache import CacheDirError

    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)

    with pytest.raises(CacheDirError, match="writable by other users"):
        DiskCache(cache_dir=shared)


def test_the_cache_refuses_a_dir_another_user_owns(monkeypatch, tmp_path):
    import os

    from dtcc_agent.disk_cache import CacheDirError

    monkeypatch.setattr(os, "getuid", lambda: os.stat(tmp_path).st_uid + 1)

    with pytest.raises(CacheDirError, match="owned by another user"):
        DiskCache(cache_dir=tmp_path)


# --- several processes sharing one cache dir ---


def test_two_caches_on_one_dir_keep_each_others_entries(tmp_path):
    """Each process used to rewrite index.json from its own copy, dropping
    entries another process had added since it started."""
    a = DiskCache(cache_dir=tmp_path)
    b = DiskCache(cache_dir=tmp_path)

    a.store(obj=1, operation="datasets.buildings", category="datasets",
            params_hash="h", bounds=[0, 0, 10, 10], source="LM")
    b.store(obj=2, operation="datasets.buildings", category="datasets",
            params_hash="h", bounds=[100, 100, 110, 110], source="LM")

    assert len(DiskCache(cache_dir=tmp_path)._index) == 2
    hit = a.dataset_lookup("datasets.buildings", "LM", "h", [101, 101, 109, 109])
    assert hit is not None and a.load(hit[0]) == 2


def test_the_cache_refuses_a_dir_whose_parent_others_can_write(tmp_path):
    """Whoever can write the parent can swap the whole cache dir for theirs."""
    from dtcc_agent.disk_cache import CacheDirError

    parent = tmp_path / "parent"
    parent.mkdir()
    parent.chmod(0o775)

    with pytest.raises(CacheDirError, match="writable by other users"):
        DiskCache(cache_dir=parent / "cache")


def test_a_sticky_shared_parent_like_tmp_is_fine(tmp_path):
    parent = tmp_path / "tmp"
    parent.mkdir()
    parent.chmod(0o1777)

    DiskCache(cache_dir=parent / "cache")


def test_the_cache_refuses_a_pickle_others_can_write(tmp_path):
    """A private dir does not help if one of its pickles is writable."""
    from dtcc_agent.disk_cache import CacheDirError

    cache = DiskCache(cache_dir=tmp_path / "c")
    cache_id = cache.store(obj=1, operation="op", category="builder", params_hash="h")
    (tmp_path / "c" / "objects" / f"{cache_id}.pkl").chmod(0o666)

    with pytest.raises(CacheDirError, match="writable by other users"):
        DiskCache(cache_dir=tmp_path / "c")


def test_cache_files_are_created_private(tmp_path):
    import os
    import stat

    old = os.umask(0o002)
    try:
        cache = DiskCache(cache_dir=tmp_path / "c")
        cache_id = cache.store(obj=1, operation="op", category="builder", params_hash="h")
    finally:
        os.umask(old)

    for name in ("index.json", "index.lock", f"objects/{cache_id}.pkl"):
        assert stat.S_IMODE((tmp_path / "c" / name).stat().st_mode) == 0o600, name


def test_load_refuses_a_cache_id_that_is_not_one(tmp_path):
    cache = DiskCache(cache_dir=tmp_path / "c")

    with pytest.raises(ValueError):
        cache.load("../../elsewhere")


def test_missing_parents_are_created_private_whatever_the_umask(tmp_path):
    """A fresh ~/.cache made under umask 002 was 0775 and failed the check."""
    import os
    import stat

    old = os.umask(0o002)
    try:
        DiskCache(cache_dir=tmp_path / "home" / ".cache" / "dtcc_agent")
    finally:
        os.umask(old)

    assert stat.S_IMODE((tmp_path / "home" / ".cache").stat().st_mode) == 0o700


def test_a_symlinked_cache_dir_is_used_through_its_real_path(tmp_path):
    """Once checked, the cache never follows the link again, so swapping it
    cannot redirect pickle loads."""
    real = tmp_path / "real"
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    real.mkdir(mode=0o700)

    cache = DiskCache(cache_dir=link)

    assert cache._cache_dir == real.resolve()


def test_a_private_symlink_inside_the_cache_is_accepted(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir(mode=0o700)
    (tmp_path / "c").mkdir(mode=0o700)
    (tmp_path / "c" / "objects").symlink_to(storage, target_is_directory=True)

    cache = DiskCache(cache_dir=tmp_path / "c")

    assert cache._objects_dir == storage.resolve()


def test_startup_ignores_a_file_removed_while_it_scans(monkeypatch, tmp_path):
    """Another process's cleanup or index write can remove a file mid-scan."""
    from pathlib import Path

    (tmp_path / "c").mkdir(mode=0o700)
    ghost = tmp_path / "c" / "index.123.tmp"
    real_iterdir = Path.iterdir

    def iterdir_with_a_ghost(self):
        yield from real_iterdir(self)
        if self.name == "c":
            yield ghost  # listed, then gone

    monkeypatch.setattr(Path, "iterdir", iterdir_with_a_ghost)

    DiskCache(cache_dir=tmp_path / "c")
