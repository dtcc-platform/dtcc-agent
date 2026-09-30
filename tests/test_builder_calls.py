"""U2 (#11), decided 2026-09-30: builder results are not cached until their
keys are correct. Each builder call is recorded instead, so the team can see
how often a cache would have hit before building provenance keys."""

import json

import numpy as np
import pytest

from dtcc_agent import builder_calls
from dtcc_agent.disk_cache import CACHE_ALLOWLIST, DiskCache
from dtcc_agent.dispatcher import run_operation
from dtcc_agent.object_store import ObjectStore


def test_only_the_bounds_keyed_downloads_are_cached():
    assert CACHE_ALLOWLIST == {"datasets.point_cloud", "datasets.buildings"}


@pytest.fixture
def log_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DTCC_AGENT_LOG_DIR", str(tmp_path))
    return tmp_path


def _lines(log_dir):
    path = log_dir / "builder_calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _key(store, params):
    return builder_calls.would_be_key("builder.build_terrain_raster", params, store)


def test_a_builder_call_is_recorded_not_cached(log_dir, tmp_path):
    store = ObjectStore()
    cache = DiskCache(cache_dir=tmp_path / "cache")
    pc = store.store(np.zeros((3, 3)), source_op="datasets.point_cloud")

    result = run_operation("builder.pc_filter.classification_filter",
                           {"pc": pc, "classes": [2]}, store, cache=cache)

    assert cache._index == []
    [line] = _lines(log_dir)
    assert line["operation"] == "builder.pc_filter.classification_filter"
    assert line["ok"] is ("error" not in result)
    assert line["seconds"] >= 0 and len(line["key"]) == 16
    assert set(line) == {"at", "operation", "ok", "seconds", "key"}  # a hash, never the parameters
    assert pc not in json.dumps(line)


def test_the_key_tells_areas_apart(log_dir):
    store = ObjectStore()
    pc = store.store(np.zeros((3, 3)), source_op="datasets.point_cloud")

    a = _key(store, {"pc": pc, "cell_size": 2.0, "bounds": [0, 0, 10, 10]})
    b = _key(store, {"pc": pc, "cell_size": 2.0, "bounds": [0, 0, 20, 20]})

    assert a != b


def test_the_key_matches_the_same_inputs_across_sessions(log_dir):
    """What a cache shared across Sessions would have hit (an upper bound: the
    metadata fingerprint can call two different inputs equal)."""
    one, two = ObjectStore(), ObjectStore()
    params = {"cell_size": 2.0, "bounds": [0, 0, 10, 10]}

    a = _key(one, {**params, "pc": one.store(np.zeros((3, 3)), source_op="datasets.point_cloud")})
    b = _key(two, {**params, "pc": two.store(np.zeros((3, 3)), source_op="datasets.point_cloud")})

    assert a == b


def test_nothing_is_recorded_without_a_log_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("DTCC_AGENT_LOG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    builder_calls.record("builder.x", {}, ObjectStore(), 0.1, ok=True)

    assert list(tmp_path.iterdir()) == []
