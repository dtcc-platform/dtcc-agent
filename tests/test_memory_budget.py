"""T11 (#23) at the tool level: oversized results, the Run cap (U4, #13)."""

import json

import numpy as np
import pytest

import dtcc_agent.server as server
from dtcc_agent.dispatcher import _store_and_summarize, _kept
from dtcc_agent.object_store import ObjectStore


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(server, "_local_session",
                        server._Session(objects=ObjectStore(max_bytes=100_000)))
    return server._session()


def test_an_oversized_operation_result_returns_its_summary_without_a_ref():
    store = ObjectStore(max_bytes=100_000)
    result = _store_and_summarize(np.zeros(50_000), "builder.big", store, "")
    assert result["object_ref"] is None
    assert "Too large to keep" in result["not_stored"]
    assert "summary" in result and len(store) == 0


def test_a_tuple_with_one_oversized_part_says_so():
    store = ObjectStore(max_bytes=100_000)
    result = _store_and_summarize((np.zeros(10), np.zeros(50_000)), "op", store, "")
    assert result["object_refs"][0] and result["object_refs"][1] is None
    assert "not_stored" in result


def test_a_result_that_fits_carries_no_note():
    store = ObjectStore(max_bytes=100_000)
    assert _kept(store, store.store(np.zeros(10))) == {}


def test_an_oversized_simulation_result_still_gets_a_run(session):
    run_ref = server._store_result("air_temperature", [0, 0, 1, 1], {}, np.zeros(50_000))
    run = session.results[run_ref]
    assert run["object_ref"] is None
    assert "Too large to keep" in server._kept(run["object_ref"])["not_stored"]


def test_a_session_remembers_its_last_max_runs(session, monkeypatch):
    monkeypatch.setattr(server, "MAX_RUNS", 3)
    runs = [server._store_result("sim", [0, 0, 1, 1], {}, np.zeros(10)) for _ in range(5)]
    assert list(session.results) == runs[2:]


def test_an_oversized_geojson_is_summarised_not_stored(session, tmp_path, monkeypatch):
    features = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [i, i]},
                 "properties": {"name": "x" * 50}} for i in range(2_000)]
    (tmp_path / "big.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}))
    monkeypatch.setenv("SHARED_RESULTS_DIR", str(tmp_path))

    result = json.loads(server.load_geojson("big.geojson"))

    assert result["object_ref"] is None
    assert "Too large to keep" in result["not_stored"]
    assert result["feature_count"] == 2_000
