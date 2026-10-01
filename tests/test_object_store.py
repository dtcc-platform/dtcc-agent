"""Tests for the in-memory object store."""

import json

import numpy as np

from dtcc_agent.object_store import MemoryBudget, ObjectStore, _estimate_bytes


class TestEstimateBytes:
    def test_numpy_arrays(self):
        class FakePC:
            def __init__(self):
                self.points = np.zeros((100, 3), dtype=np.float64)
                self.classification = np.zeros(100, dtype=np.uint8)
        nbytes = _estimate_bytes(FakePC())
        assert nbytes >= 100 * 3 * 8 + 100

    def test_plain_object(self):
        assert _estimate_bytes("hello") == 64  # minimum

    def test_a_dict_counts_its_contents(self):
        small = _estimate_bytes({"key": "value"})
        assert _estimate_bytes({"key": "value" * 1000}) > small + 4000

    def test_the_u4_probe_is_counted_within_an_order_of_magnitude(self):
        # #13: a GeoJSON-like dict whose JSON is 64,168 bytes was counted as 64.
        features = [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [319000.0 + i, 6400000.0 + i]},
             "properties": {"name": f"station_{i}", "air_temperature": 21.5 + i / 100}}
            for i in range(413)
        ]
        geojson = {"type": "FeatureCollection", "features": features}
        size = len(json.dumps(geojson))
        assert 60_000 < size < 70_000
        assert size / 10 <= _estimate_bytes(geojson) <= size * 10

    def test_a_shared_array_and_a_view_count_once(self):
        arr = np.zeros(10_000)
        assert _estimate_bytes([arr, arr, arr[:10]]) < arr.nbytes + 1000

    def test_a_simulation_result_counts_its_values(self):
        class Vector:
            def __init__(self):
                self._values = np.zeros(50_000)

            @property
            def array(self):
                return self._values

        class Function:
            __slots__ = ("x",)

            def __init__(self):
                self.x = Vector()

        assert _estimate_bytes(Function()) >= 50_000 * 8

    def test_classes_and_modules_are_not_walked(self):
        assert _estimate_bytes({"np": np, "cls": ObjectStore}) < 1000


class TestObjectStore:
    def test_store_and_get(self):
        store = ObjectStore()
        arr = np.ones((50, 3))
        obj_id = store.store(arr, source_op="test", label="my_array")
        assert obj_id.startswith("obj_") and len(obj_id) == 12
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
        assert items[0]["object_ref"] == id2
        assert items[1]["object_ref"] == id1

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
        assert entry["object_ref"] == obj_id
        assert entry["type"] == "ndarray"
        assert entry["source_op"] == "test_op"
        assert entry["label"] == "my_label"
        assert "created" in entry
        assert "nbytes" in entry


def test_delete_returns_the_removed_entry_and_none_when_absent():
    store = ObjectStore()
    obj_id = store.store([1, 2, 3], source_op="op", label="lbl")

    entry = store.delete(obj_id)

    assert (entry["type"], entry["label"]) == ("list", "lbl")
    assert store.delete(obj_id) is None
    assert store.total_bytes == 0


# -- Shared budget and oversized Objects (T11, #23; U4, #13) -----------------

ARRAY = 80_000  # np.zeros(10_000) of float64


def _arr():
    return np.zeros(10_000)


def test_an_object_larger_than_the_store_is_not_kept_and_nothing_is_evicted():
    store = ObjectStore(max_bytes=3 * ARRAY)
    kept = store.store(_arr())
    assert store.store(np.zeros(40_000)) is None
    assert kept in store
    assert "Too large to keep" in store.not_stored()


def test_the_least_recently_used_object_in_any_store_goes_first():
    budget = MemoryBudget(max_bytes=int(2.5 * ARRAY))
    a = ObjectStore(max_bytes=2 * ARRAY, budget=budget)
    b = ObjectStore(max_bytes=2 * ARRAY, budget=budget)
    old = a.store(_arr())
    newer = b.store(_arr())
    a.get(old)  # now newer is the least recently used
    b.store(_arr())
    assert old in a and newer not in b
    assert budget.total_bytes <= budget.max_bytes


def test_each_store_keeps_its_own_cap_under_a_larger_budget():
    budget = MemoryBudget(max_bytes=10 * ARRAY)
    store = ObjectStore(max_bytes=2 * ARRAY + 1000, budget=budget)
    refs = [store.store(_arr()) for _ in range(3)]
    assert refs[0] not in store and len(store) == 2


def test_clearing_a_store_returns_its_bytes_to_the_budget():
    budget = MemoryBudget(max_bytes=10 * ARRAY)
    a = ObjectStore(max_bytes=5 * ARRAY, budget=budget)
    b = ObjectStore(max_bytes=5 * ARRAY, budget=budget)
    a.store(_arr())
    b.store(_arr())
    a.clear()
    assert len(a) == 0
    assert budget.total_bytes == b.total_bytes
