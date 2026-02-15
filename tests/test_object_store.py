"""Tests for the in-memory object store."""

import numpy as np

from dtcc_agent.object_store import ObjectStore, _estimate_bytes


class TestEstimateBytes:
    def test_numpy_arrays(self):
        class FakePC:
            points = np.zeros((100, 3), dtype=np.float64)
            classification = np.zeros(100, dtype=np.uint8)
        nbytes = _estimate_bytes(FakePC())
        assert nbytes >= 100 * 3 * 8 + 100

    def test_plain_object(self):
        assert _estimate_bytes("hello") == 64  # minimum

    def test_dict_object(self):
        assert _estimate_bytes({"key": "value"}) == 64


class TestObjectStore:
    def test_store_and_get(self):
        store = ObjectStore()
        arr = np.ones((50, 3))
        obj_id = store.store(arr, source_op="test", label="my_array")
        assert len(obj_id) == 8
        retrieved = store.get(obj_id)
        np.testing.assert_array_equal(retrieved, arr)

    def test_get_missing_raises(self):
        store = ObjectStore()
        try:
            store.get("nonexist")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_delete(self):
        store = ObjectStore()
        obj_id = store.store(np.zeros(10), source_op="test")
        assert obj_id in store
        store.delete(obj_id)
        assert obj_id not in store
        assert len(store) == 0

    def test_list_ordering(self):
        store = ObjectStore()
        id1 = store.store("first", source_op="op1")
        id2 = store.store("second", source_op="op2")
        items = store.list()
        # Most recent first
        assert items[0]["id"] == id2
        assert items[1]["id"] == id1

    def test_list_limit(self):
        store = ObjectStore()
        for i in range(5):
            store.store(f"item_{i}", source_op="op")
        items = store.list(limit=2)
        assert len(items) == 2

    def test_contains(self):
        store = ObjectStore()
        obj_id = store.store("data", source_op="test")
        assert obj_id in store
        assert "fake_id" not in store

    def test_total_bytes_tracking(self):
        store = ObjectStore()
        arr = np.zeros((100, 3), dtype=np.float64)
        store.store(arr, source_op="test")
        assert store.total_bytes >= arr.nbytes

    def test_lru_eviction(self):
        # Each array is 10 * 8 = 80 bytes, but _estimate_bytes returns max(nbytes, 64)
        # so each is 80 bytes. Limit 200 allows 2 objects but not 3.
        store = ObjectStore(max_bytes=200)
        id1 = store.store(np.zeros(10, dtype=np.float64), source_op="op1")
        id2 = store.store(np.zeros(10, dtype=np.float64), source_op="op2")
        assert len(store) == 2
        # Access id1 so id2 becomes LRU
        store.get(id1)
        # This third store should trigger eviction of id2 (LRU)
        id3 = store.store(np.zeros(10, dtype=np.float64), source_op="op3")
        assert id2 not in store, "LRU object should have been evicted"
        assert store.total_bytes <= 200

    def test_list_entry_fields(self):
        store = ObjectStore()
        obj_id = store.store(np.zeros(5), source_op="test_op", label="my_label")
        items = store.list()
        assert len(items) == 1
        entry = items[0]
        assert entry["id"] == obj_id
        assert entry["type"] == "ndarray"
        assert entry["source_op"] == "test_op"
        assert entry["label"] == "my_label"
        assert "created" in entry
        assert "nbytes" in entry
