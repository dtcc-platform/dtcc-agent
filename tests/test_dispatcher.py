"""Tests for the dispatcher module."""

import numpy as np
import pytest

from dtcc_agent.object_store import ObjectStore
from dtcc_agent.dispatcher import run_operation, _resolve_bounds

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
