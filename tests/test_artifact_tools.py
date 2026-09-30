"""render_object, export_object and load_geojson use the Session's artifacts (T7, #21)."""

import json

import numpy as np
import pytest

import dtcc_agent.server as server
from dtcc_agent import artifacts

model = pytest.importorskip("dtcc_core.model")


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("DTCC_AGENT_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(server, "_local_session", server._Session(id="sess1"))
    return server._session()


def _no_paths(text, tmp_path):
    assert str(tmp_path) not in text
    assert "/tmp" not in text


def test_render_object_returns_an_image_artifact_in_its_session(session, tmp_path):
    ref = session.objects.store(model.Bounds(xmin=0, ymin=0, xmax=5, ymax=5), source_op="t")

    text = server.render_object(ref, width=300, height=200)

    result = json.loads(text)
    assert result["artifact"]["kind"] == "image"
    assert artifacts.find("sess1", result["artifact"]["name"]) is not None
    _no_paths(text, tmp_path)


def test_render_object_reports_geometry_it_cannot_draw(session):
    ref = session.objects.store(model.Mesh(), source_op="t")
    assert "Nothing to render" in json.loads(server.render_object(ref))["error"]


def test_export_object_returns_a_file_artifact_in_its_session(session, tmp_path):
    mesh = model.Mesh(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float),
                      faces=np.array([[0, 1, 2]]))
    ref = session.objects.store(mesh, source_op="t")

    text = server.export_object(ref, "obj")

    result = json.loads(text)
    assert result["artifact"]["kind"] == "file"
    path = artifacts.find("sess1", result["artifact"]["name"])
    assert path is not None and path.stat().st_size > 0
    assert artifacts.download_name(path.name) == f"{ref}.obj"
    _no_paths(text, tmp_path)


def test_a_failed_export_leaves_no_file_behind(session, monkeypatch):
    import dtcc_core.io

    def boom(obj, path):
        open(path, "w").close()
        raise RuntimeError("disk full")

    monkeypatch.setattr(dtcc_core.io, "save_mesh", boom)
    ref = session.objects.store(model.Mesh(), source_op="t")

    assert "Export failed" in json.loads(server.export_object(ref, "obj"))["error"]
    assert not list((artifacts.root() / "sess1").iterdir())


@pytest.mark.parametrize("name", ["/etc/passwd", "../outside.geojson", "missing.geojson"])
def test_load_geojson_reads_only_the_shared_results_directory(tmp_path, monkeypatch, name):
    shared = tmp_path / "shared"
    shared.mkdir()
    (tmp_path / "outside.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    monkeypatch.setenv("SHARED_RESULTS_DIR", str(shared))

    assert "No GeoJSON file" in json.loads(server.load_geojson(name))["error"]
