"""Session artifact directories and the shared results root (T7, #21; U1, #10)."""

import stat

import pytest

from dtcc_agent import artifacts


@pytest.fixture(autouse=True)
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv("DTCC_AGENT_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setenv("SHARED_RESULTS_DIR", str(shared))
    return tmp_path


def test_a_new_artifact_lands_in_its_sessions_private_directory():
    path = artifacts.new_path("abc123", "obj_1", ".PNG")
    path.write_bytes(b"x")

    assert path.parent == artifacts.root() / "abc123"
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert path.name.endswith("_obj_1.png")
    assert artifacts.describe(path) == {"name": path.name, "kind": "image"}
    assert artifacts.find("abc123", path.name) == path
    assert artifacts.download_name(path.name) == "obj_1.png"


def test_names_are_unguessable_and_unsafe_stems_are_flattened():
    first = artifacts.new_path("s", "../../etc/passwd", ".csv")
    second = artifacts.new_path("s", "../../etc/passwd", ".csv")
    assert first != second
    assert first.parent == second.parent == artifacts.root() / "s"
    assert "/" not in first.name


def test_another_sessions_artifact_is_not_found():
    path = artifacts.new_path("mine", "obj_1", ".csv")
    path.write_text("x")
    assert artifacts.find("theirs", path.name) is None


@pytest.mark.parametrize("name", ["../x.png", "passwd", "a" * 32 + "_x", "", "%2e%2e"])
def test_a_name_that_is_not_an_artifact_name_is_not_found(name):
    assert artifacts.find("s", name) is None


@pytest.mark.parametrize("session_id", ["..", "a/b", "", "x" * 65])
def test_an_invalid_session_id_never_becomes_a_directory(session_id):
    with pytest.raises(ValueError):
        artifacts.new_path(session_id, "obj", ".png")
    assert artifacts.find(session_id, "0" * 32 + "_obj.png") is None


def test_removing_a_session_deletes_its_files():
    path = artifacts.new_path("gone", "obj", ".csv")
    path.write_text("x")
    artifacts.remove_session("gone")
    assert not path.parent.exists()


def test_a_shared_result_resolves_only_inside_the_shared_directory(roots):
    shared = roots / "shared"
    (shared / "run").mkdir()
    (shared / "run" / "t.geojson").write_text("{}")
    (roots / "secret.geojson").write_text("{}")
    (shared / "link.geojson").symlink_to(roots / "secret.geojson")

    assert artifacts.shared_result("run/t.geojson") == (shared / "run" / "t.geojson").resolve()
    assert artifacts.shared_result("../secret.geojson") is None
    assert artifacts.shared_result(str(roots / "secret.geojson")) is None
    assert artifacts.shared_result("link.geojson") is None
    assert artifacts.shared_result("missing.geojson") is None


def test_no_shared_directory_means_no_shared_results(monkeypatch):
    monkeypatch.delenv("SHARED_RESULTS_DIR")
    assert artifacts.shared_result("t.geojson") is None
