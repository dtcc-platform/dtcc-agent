"""Simulation results carry field statistics (dtcc-sim#8).

dtcc-sim returns a VolumeMesh with a vertex temperature Field. In mini-service
mode the agent asks for Core's native dtcc format, the one output it can read,
and decodes it; an older build that offers only XDMF still answers, without
statistics. Nothing here runs a solver or a service.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dtcc_agent import runner, server

HEAT = "urban_heat_simulation"
BOUNDS = [0.0, 0.0, 100.0, 100.0]


@pytest.fixture
def clean_stores(monkeypatch):
    monkeypatch.setattr(server, "_local_session", server._Session())


def _mesh(values):
    from dtcc_core.model import Field, VolumeMesh

    return VolumeMesh(
        vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        cells=np.array([[0, 1, 2, 3]], dtype=np.int64),
        fields=[Field(name="temperature", unit="degC", dim=1, association="vertex",
                      values=np.array(values, dtype=float).reshape(-1, 1))],
    )


def _native_bytes(mesh, tmp_path) -> bytes:
    from dtcc_core.io import save_model

    path = tmp_path / "result.dtcc"
    save_model(mesh, path)
    return path.read_bytes()


class _RemoteDataset:
    """A dtcc-sim dataset as Core's remote protocol presents it."""

    def __init__(self, formats, payload=b"xdmf-archive"):
        self.supported_formats = formats
        self.base_url = "http://sim"
        self.payload = payload
        self.requested = None

    def validate(self, kwargs):
        return kwargs

    def build(self, validated, remote_info_callback=None):
        self.requested = validated["format"]
        remote_info_callback({"remote_task_id": "t1"})
        if self.requested == "dtcc":
            return self.payload, "dtcc", "application/octet-stream"
        return b"xdmf-archive", "tar.gz", "application/gzip"


def _remote(monkeypatch, dataset):
    monkeypatch.setattr(runner, "_remote_base_url", lambda: "http://sim")
    monkeypatch.setattr(runner, "_get_remote_dataset", lambda name: dataset)


def test_the_runner_asks_for_the_native_format_and_decodes_it(monkeypatch, tmp_path):
    dataset = _RemoteDataset(["xdmf", "dtcc"], _native_bytes(_mesh([18, 19, 20, 21]), tmp_path))
    _remote(monkeypatch, dataset)

    result = runner.run(HEAT, BOUNDS)

    assert dataset.requested == "dtcc"
    np.testing.assert_array_equal(result.model.fields[0].values.ravel(), [18, 19, 20, 21])
    assert "model" not in result.as_dict()
    json.dumps(result.as_dict())


def test_a_build_without_the_native_format_keeps_its_first_format(monkeypatch):
    dataset = _RemoteDataset(["xdmf"])
    _remote(monkeypatch, dataset)

    result = runner.run(HEAT, BOUNDS)

    assert dataset.requested == "xdmf" and result.model is None


def test_run_simulation_reports_temperature_statistics(monkeypatch, tmp_path, clean_stores):
    _remote(monkeypatch, _RemoteDataset(["xdmf", "dtcc"],
                                        _native_bytes(_mesh([18, 19, 20, 21]), tmp_path)))

    payload = json.loads(server.run_simulation(HEAT, BOUNDS))

    summary = payload["summary"]
    assert summary["field_name"] == "temperature" and summary["unit"] == "degC"
    assert (summary["min"], summary["max"], summary["mean"]) == (18.0, 21.0, 19.5)
    assert payload["remote_result"]["output_format"] == "dtcc"
    assert json.loads(server.get_run_summary(payload["run_ref"]))["summary"] == summary
    assert json.loads(server.list_past_runs())[0]["summary"] == summary


def test_compare_scenarios_reports_the_difference(monkeypatch, tmp_path, clean_stores):
    payloads = iter([_native_bytes(_mesh([18, 19, 20, 21]), tmp_path),
                     _native_bytes(_mesh([20, 21, 22, 23]), tmp_path)])

    class _TwoRuns(_RemoteDataset):
        def build(self, validated, remote_info_callback=None):
            self.payload = next(payloads)
            return super().build(validated, remote_info_callback)

    _remote(monkeypatch, _TwoRuns(["xdmf", "dtcc"]))

    payload = json.loads(server.compare_scenarios(
        HEAT, BOUNDS, {}, {"air_temperature": 35}, label_a="baseline", label_b="heatwave"))

    assert payload["baseline"]["mean"] == 19.5 and payload["heatwave"]["mean"] == 21.5
    assert payload["difference"]["mean"] == 2.0
    assert payload["unit"] == "degC"


def test_an_older_sim_build_still_answers_and_says_why_there_are_no_numbers(
        monkeypatch, clean_stores):
    _remote(monkeypatch, _RemoteDataset(["xdmf"]))

    payload = json.loads(server.run_simulation(HEAT, BOUNDS))

    assert "error" not in payload
    assert "tar.gz output" in payload["summary"]["message"]
    assert "dtcc format" in payload["summary"]["message"]


def test_an_in_process_volume_mesh_is_summarised(monkeypatch, clean_stores):
    monkeypatch.setattr(runner, "run", lambda name, bounds, params: _mesh([1, 2, 3, 4]))

    payload = json.loads(server.run_simulation(HEAT, BOUNDS))

    assert payload["summary"]["mean"] == 2.5 and "remote_result" not in payload
