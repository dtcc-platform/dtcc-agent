"""Simulations as DTCC Engine jobs (#109).

The Engine is mocked at the HTTP layer with httpx.MockTransport, so these run
offline. One test round-trips a real canonical package written by Core, the
same way the Engine's worker writes it, through run_simulation's statistics.
"""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from dtcc_agent import engine, runner, runtime, server

HEAT = "urban_heat_simulation"
BOUNDS = [0.0, 0.0, 100.0, 100.0]
HEAT_DESCRIBE = {"name": HEAT, "description": "Steady urban heat",
                 "args_schema": {"properties": {"bounds": {}, "format": {}}}}


class FakeEngine:
    """Answers the Engine's routes from a script; records every request."""

    def __init__(self, *, submit=None, states=("completed",), package=b"pkg"):
        self.requests: list[httpx.Request] = []
        self.submit = submit or httpx.Response(202, json={"job_id": "j1", "status_url": "/api/v1/jobs/j1"})
        self.states = list(states)
        self.package = package

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/v1/datasets":
            return httpx.Response(200, json={"datasets": {HEAT: HEAT_DESCRIBE, "buildings": {}}})
        if path == "/api/v1/jobs" and request.method == "POST":
            return self.submit
        if path.endswith("/package"):
            return httpx.Response(200, content=self.package)
        if path.startswith("/api/v1/jobs/"):
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            if isinstance(state, httpx.Response):
                return state
            body = {"job_id": "j1", "state": state,
                    "error": {"type": "ValueError"} if state == "failed" else None}
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    def posts(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.method == "POST"]


def _client(fake: FakeEngine, *, timeout: float = 900.0):
    sleeps: list[float] = []
    now = [0.0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    client = engine.Engine("http://engine", "tok", timeout=timeout,
                           transport=httpx.MockTransport(fake),
                           sleep=sleep, clock=lambda: now[0])
    return client, sleeps


# -- The client --------------------------------------------------------------

def test_a_job_is_submitted_once_polled_with_backoff_and_its_format_dropped():
    fake = FakeEngine(states=["queued", "running", "running", "running", "running", "completed"])
    client, sleeps = _client(fake)

    client.submit(HEAT, {"bounds": BOUNDS, "format": "xdmf"})
    job = client.wait("j1")

    assert job["state"] == "completed"
    assert fake.posts() == [{"dataset": HEAT, "parameters": {"bounds": BOUNDS}}]
    assert sleeps == [1.0, 2.0, 4.0, 5.0, 5.0]
    assert fake.requests[0].headers["Authorization"] == "Bearer tok"


def test_a_rejected_token_says_so():
    client, _ = _client(FakeEngine(submit=httpx.Response(401, json={"detail": "Missing or invalid"})))

    with pytest.raises(engine.EngineError, match="rejected the agent's token"):
        client.submit(HEAT, {"bounds": BOUNDS})


def test_refused_parameters_carry_the_engines_detail():
    detail = [{"loc": ["mesh_max_mesh_size"], "type": "float_parsing"}]
    client, _ = _client(FakeEngine(submit=httpx.Response(422, json={"detail": detail})))

    with pytest.raises(engine.EngineError, match="refused the parameters.*float_parsing"):
        client.submit(HEAT, {"bounds": BOUNDS})


def test_an_unconfirmed_submission_is_polled_never_resubmitted():
    unconfirmed = httpx.Response(503, json={"detail": {"message": "may still run", "job_id": "j9"}})
    fake = FakeEngine(submit=unconfirmed)
    client, _ = _client(fake)

    assert client.submit(HEAT, {"bounds": BOUNDS}) == "j9"
    assert len(fake.posts()) == 1


def test_an_unavailable_queue_says_so():
    client, _ = _client(FakeEngine(submit=httpx.Response(503, json={"detail": "queue down"})))

    with pytest.raises(engine.EngineError, match="queue is unavailable"):
        client.submit(HEAT, {"bounds": BOUNDS})


def test_a_failed_job_reports_its_error_type():
    client, _ = _client(FakeEngine(states=["failed"]))

    with pytest.raises(engine.EngineError, match="j1 failed: ValueError"):
        client.wait("j1")


def test_an_expired_job_says_to_run_it_again():
    client, _ = _client(FakeEngine(states=[httpx.Response(404, json={"detail": "unknown"})]))

    with pytest.raises(engine.EngineError, match="unknown or expired"):
        client.wait("j1")


def test_a_briefly_unknown_state_is_waited_out():
    fake = FakeEngine(states=[httpx.Response(503, json={"detail": "redis"}), "completed"])
    client, sleeps = _client(fake)

    assert client.wait("j1")["state"] == "completed" and sleeps == [1.0]


def test_a_job_that_outlasts_the_timeout_names_itself():
    client, sleeps = _client(FakeEngine(states=["running"]), timeout=10.0)

    with pytest.raises(engine.EngineError, match="j1 did not finish within 10 s"):
        client.wait("j1")
    assert sum(sleeps) <= 10.0


def test_an_unreachable_engine_says_where_it_looked():
    def down(request):
        raise httpx.ConnectError("refused")

    client = engine.Engine("http://engine", "tok", transport=httpx.MockTransport(down))
    with pytest.raises(engine.EngineError, match="http://engine is unreachable"):
        client.datasets()


# -- Configuration -----------------------------------------------------------

def test_an_engine_url_without_a_token_stops_the_server_at_start(monkeypatch):
    monkeypatch.setenv("DTCC_ENGINE_URL", "http://engine")
    monkeypatch.delenv("DTCC_ENGINE_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="DTCC_ENGINE_TOKEN"):
        runtime.start()


def test_with_the_engine_configured_dtcc_sim_is_neither_asked_nor_merged(monkeypatch):
    monkeypatch.setenv("DTCC_ENGINE_URL", "http://engine")
    monkeypatch.setenv("DTCC_REMOTE_SERVICES", "http://sim:8001")

    assert runner._remote_services() == []


@pytest.fixture
def on_engine(monkeypatch):
    """The runner configured for an Engine that is a FakeEngine."""
    monkeypatch.setenv("DTCC_ENGINE_URL", "http://engine")
    monkeypatch.setenv("DTCC_ENGINE_TOKEN", "tok")
    monkeypatch.setenv("DTCC_REMOTE_SERVICES", "http://sim:8001")

    def install(fake: FakeEngine):
        client, _ = _client(fake)
        monkeypatch.setattr(engine, "client", lambda: client)
        return client

    return install


def test_simulations_are_listed_and_described_by_the_engine(on_engine):
    on_engine(FakeEngine())

    assert runner.list_simulations() == [{"name": HEAT, "description": "Steady urban heat"}]
    assert runner.get_schema(HEAT) == HEAT_DESCRIBE["args_schema"]
    with pytest.raises(KeyError):
        runner.get_schema("buildings")


def test_the_engine_path_is_used_even_when_dtcc_sim_is_configured(on_engine, monkeypatch):
    monkeypatch.setattr(runner, "_get_remote_dataset",
                        lambda name: pytest.fail("dtcc-sim must not be asked"))
    client = on_engine(FakeEngine())
    monkeypatch.setattr(client, "package", lambda job_id: ("mesh", 3))

    result = runner.run(HEAT, BOUNDS, {"format": "dtcc"})

    assert (result.model, result.task_id, result.output_format) == ("mesh", "j1", "dtccpkg")
    assert "format" not in result.parameters and "model" not in result.as_dict()


# -- A real package, end to end ----------------------------------------------

@pytest.fixture
def heat_package(tmp_path):
    """A canonical .dtccpkg of a VolumeMesh with a temperature field, written
    by Core exactly as the Engine's worker writes one."""
    from dtcc_core.datasets import DatasetBaseArgs, DatasetDescriptor, get_dataset, unregister
    from dtcc_core.model import Field, VolumeMesh

    class ProbeArgs(DatasetBaseArgs):
        pass

    class ProbeDataset(DatasetDescriptor):
        name = "agent_engine_probe"
        description = "dtcc-agent test dataset: a tetrahedron with a temperature field"
        ArgsModel = ProbeArgs

        def build(self, args):
            return VolumeMesh(
                vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
                cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
                fields=[Field(name="temperature", unit="degC", dim=1, association="vertex",
                              values=np.array([[18.], [19.], [20.], [21.]]))],
            )

    try:
        path = tmp_path / "j1.dtccpkg"
        get_dataset("agent_engine_probe")(bounds=BOUNDS).export(path, canonical=True)
        yield path.read_bytes()
    finally:
        unregister("agent_engine_probe")


def test_run_simulation_reports_statistics_from_an_engine_package(on_engine, heat_package, monkeypatch):
    monkeypatch.setattr(server, "_local_session", server._Session())
    on_engine(FakeEngine(states=["queued", "completed"], package=heat_package))

    payload = json.loads(server.run_simulation(HEAT, BOUNDS))

    summary = payload["summary"]
    assert (summary["min"], summary["max"], summary["mean"]) == (18.0, 21.0, 19.5)
    assert summary["unit"] == "degC"
    assert payload["remote_result"]["task_id"] == "j1"
    assert payload["remote_result"]["size_bytes"] == len(heat_package)
