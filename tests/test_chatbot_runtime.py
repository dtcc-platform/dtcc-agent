"""The agent runtime (#86): pydantic-ai, and a clear refusal for anything else,
the removed Agent SDK included."""

import json

import pytest

from chatbot import runtime


def test_the_default_runtime_is_pydantic_ai(monkeypatch):
    monkeypatch.delenv("DTCC_AGENT_RUNTIME", raising=False)
    assert runtime.load_runtime().NAME == "pydantic-ai"


def test_an_empty_runtime_means_the_default(monkeypatch):
    # Compose forwards ${DTCC_AGENT_RUNTIME:-pydantic-ai}, but a bare
    # DTCC_AGENT_RUNTIME= in an env file arrives empty.
    monkeypatch.setenv("DTCC_AGENT_RUNTIME", "")
    assert runtime.load_runtime().NAME == "pydantic-ai"


def test_an_unknown_runtime_stops_the_chatbot(monkeypatch):
    monkeypatch.setenv("DTCC_AGENT_RUNTIME", "langchain")
    with pytest.raises(SystemExit, match="DTCC_AGENT_RUNTIME"):
        runtime.load_runtime()


def test_asking_for_the_removed_sdk_runtime_says_it_is_gone(monkeypatch):
    monkeypatch.setenv("DTCC_AGENT_RUNTIME", "sdk")
    with pytest.raises(SystemExit, match="removed"):
        runtime.load_runtime()


# -- The artifact frame -------------------------------

def test_an_artifact_in_a_structured_result_dict_is_found(tmp_path, monkeypatch):
    # pydantic-ai hands a tool's result over already parsed.
    from dtcc_agent import artifacts

    monkeypatch.setenv("DTCC_AGENT_ARTIFACTS_DIR", str(tmp_path))
    session_dir = tmp_path / "s1"
    session_dir.mkdir()
    (session_dir / "abc_obj_1.png").write_bytes(b"png")
    monkeypatch.setattr(artifacts, "find", lambda sid, name: session_dir / name if sid == "s1" else None)

    content = {"result": json.dumps({"artifact": {"name": "abc_obj_1.png", "kind": "image"}})}
    assert runtime.artifact_frame("s1", content) == {"type": "image", "url": "/artifacts/s1/abc_obj_1.png"}
