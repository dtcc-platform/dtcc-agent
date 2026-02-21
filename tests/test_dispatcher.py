"""Tests for the dispatcher module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dtcc_agent.object_store import ObjectStore
from dtcc_agent.dispatcher import run_operation, _resolve_bounds
from dtcc_agent.disk_cache import DiskCache, canonical_params_hash

try:
    from dtcc_core.model.geometry.bounds import Bounds
    HAS_DTCC = True
except ImportError:
    HAS_DTCC = False


class TestResolveBounds:
    @pytest.mark.skipif(not HAS_DTCC, reason="dtcc_core not installed")
    def test_4_element_list(self):
        result = _resolve_bounds([1.0, 2.0, 3.0, 4.0])
        assert hasattr(result, "xmin")
        assert result.xmin == 1.0
        assert result.ymin == 2.0
        assert result.xmax == 3.0
        assert result.ymax == 4.0

    @pytest.mark.skipif(not HAS_DTCC, reason="dtcc_core not installed")
    def test_6_element_list(self):
        result = _resolve_bounds([1.0, 2.0, 0.0, 3.0, 4.0, 10.0])
        assert result.zmin == 0.0
        assert result.zmax == 10.0

    def test_non_bounds_passthrough(self):
        result = _resolve_bounds("hello")
        assert result == "hello"

    def test_wrong_length_passthrough(self):
        result = _resolve_bounds([1.0, 2.0])
        assert result == [1.0, 2.0]


class TestRunOperation:
    def test_unknown_operation(self):
        store = ObjectStore()
        result = run_operation("nonexistent.op", {}, store)
        assert "error" in result

    def test_object_ref_resolution(self):
        """Test that string IDs get resolved from the object store."""
        store = ObjectStore()
        arr = np.array([1.0, 2.0, 3.0])
        obj_id = store.store(arr, source_op="test")
        retrieved = store.get(obj_id)
        np.testing.assert_array_equal(retrieved, arr)


class _FakePointCloud:
    """Picklable stand-in for a dtcc-core PointCloud."""
    def __init__(self, points):
        self.points = points


class TestCacheIntegration:
    def test_dataset_cache_hit_skips_download(self):
        """Cached dataset result is loaded from disk instead of re-downloading."""
        with tempfile.TemporaryDirectory() as td:
            cache = DiskCache(cache_dir=Path(td))
            store = ObjectStore()

            # Pre-populate cache with a fake point cloud
            fake_pc = _FakePointCloud(
                points=np.array([[319800.0, 6399600.0, 5.0]])
            )

            non_bounds = {"source": "LM"}
            ph = canonical_params_hash("datasets.point_cloud", non_bounds)

            cache.store(
                obj=fake_pc,
                operation="datasets.point_cloud",
                category="datasets",
                params_hash=ph,
                bounds=[319700, 6399500, 320200, 6400000],
                source="LM",
                object_type="PointCloud",
            )

            # Mock the registry lookup so it returns a datasets-category op
            mock_op = MagicMock()
            mock_op.category = "datasets"
            mock_op.name = "datasets.point_cloud"

            with patch("dtcc_agent.dispatcher.get_operation", return_value=mock_op):
                result = run_operation(
                    "datasets.point_cloud",
                    {"bounds": [319700, 6399500, 320200, 6400000], "source": "LM"},
                    store,
                    cache=cache,
                )

            assert "error" not in result
            assert result.get("cache_hit") is True
            assert "result_id" in result
            # The dataset callable should NOT have been called
            mock_op._callable.assert_not_called()
