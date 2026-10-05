# tests/test_chatbot_app.py
"""Smoke tests for the chatbot FastAPI app."""

import json
import sys
import types

import pytest

_mock_chromadb = None
if "chromadb" not in sys.modules:
    _mock_chromadb = types.ModuleType("chromadb")

    class _MockCollection:
        def count(self):
            return 0

        def add(self, **kwargs):
            return None

        def query(self, **kwargs):
            return {"documents": [[]], "distances": [[]]}

    class _MockPersistentClient:
        def __init__(self, path):
            self.path = path

        def get_or_create_collection(self, **kwargs):
            return _MockCollection()

    _mock_chromadb.PersistentClient = _MockPersistentClient
    sys.modules["chromadb"] = _mock_chromadb

from fastapi.testclient import TestClient
from chatbot.app import app

# The chromadb stub exists only for importing chatbot.app. Drop it, and the
# chatbot.memory built on it, so later test modules get the real library.
if _mock_chromadb is not None and sys.modules.get("chromadb") is _mock_chromadb:
    del sys.modules["chromadb"]
    sys.modules.pop("chatbot.memory", None)


def test_index_returns_html():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "DTCC Lurkie" in resp.text


def test_health_returns_ok():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_websocket_session_handshake():
    client = TestClient(app)
    with client.websocket_connect("/chat") as ws:
        # Send init message
        ws.send_json({"session_id": None})
        # Should receive session message
        msg = ws.receive_json()
        assert msg["type"] == "session"
        assert "session_id" in msg


# -- Artifacts (T7, #21) ----------------------------------------------------

import chatbot.app as app_module
from chatbot.runtime import artifact_frame
from chatbot.sessions import SESSION_IDLE_SECONDS
from dtcc_agent import artifacts


@pytest.fixture
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DTCC_AGENT_ARTIFACTS_DIR", str(tmp_path))
    return tmp_path


def _artifact(session_id, stem, suffix, body=b"data"):
    path = artifacts.new_path(session_id, stem, suffix)
    path.write_bytes(body)
    return path


def test_a_live_sessions_image_is_served_inline(artifact_root):
    sid = app_module.sessions.create()
    path = _artifact(sid, "obj_1", ".png", b"\x89PNG")

    resp = TestClient(app).get(f"/artifacts/{sid}/{path.name}")

    assert resp.status_code == 200
    assert resp.content == b"\x89PNG"
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers["cache-control"] == "private, no-store"
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_an_exported_file_downloads_under_its_plain_name(artifact_root):
    sid = app_module.sessions.create()
    path = _artifact(sid, "obj_1", ".csv")

    resp = TestClient(app).get(f"/artifacts/{sid}/{path.name}")

    assert resp.status_code == 200
    assert "attachment" in resp.headers["content-disposition"]
    assert "obj_1.csv" in resp.headers["content-disposition"]


def test_an_artifact_is_not_served_to_another_or_an_unknown_session(artifact_root):
    owner = app_module.sessions.create()
    other = app_module.sessions.create()
    path = _artifact(owner, "obj_1", ".png")
    client = TestClient(app)

    assert client.get(f"/artifacts/{other}/{path.name}").status_code == 404
    # A Session the chatbot no longer knows serves nothing, even if files remain.
    app_module.sessions._sessions.pop(owner)
    assert client.get(f"/artifacts/{owner}/{path.name}").status_code == 404


def test_a_traversal_name_is_not_served(artifact_root):
    sid = app_module.sessions.create()
    assert TestClient(app).get(f"/artifacts/{sid}/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_a_tool_result_with_an_image_artifact_becomes_an_image_frame(artifact_root):
    path = _artifact("s1", "obj_1", ".png")
    content = [{"type": "text", "text": json.dumps({"artifact": artifacts.describe(path)})}]

    assert artifact_frame("s1", content) == {
        "type": "image", "url": f"/artifacts/s1/{path.name}",
    }


def test_a_tool_result_with_a_file_artifact_becomes_a_download_frame(artifact_root):
    path = _artifact("s1", "obj_1", ".csv")
    content = json.dumps({"artifact": artifacts.describe(path)})

    assert artifact_frame("s1", content) == {
        "type": "file", "url": f"/artifacts/s1/{path.name}", "name": "obj_1.csv",
    }


def test_an_artifact_inside_fastmcps_structured_result_is_found(artifact_root):
    # The shape the Claude CLI actually delivers (seen end to end).
    path = _artifact("s1", "obj_1", ".png")
    tool_text = json.dumps({"object_ref": "obj_1", "artifact": artifacts.describe(path)})

    frame = artifact_frame("s1", json.dumps({"result": tool_text}))

    assert frame == {"type": "image", "url": f"/artifacts/s1/{path.name}"}


@pytest.mark.parametrize("content", [
    "not json", json.dumps({"result": "not json"}), json.dumps({"object_ref": "obj_1"}), json.dumps(["x"]), None,
    json.dumps({"artifact": {"name": "0" * 32 + "_x.png", "kind": "image"}}),
])
def test_a_tool_result_without_this_sessions_artifact_sends_no_frame(artifact_root, content):
    assert artifact_frame("s1", content) is None


# -- The turn around the runtime (#86) ------------------------------------------
#
# These hold for any runtime; a fake one stands in. The runtimes' own tests
# are tests/test_chatbot_runtime_pai.py and tests/test_chatbot_runtime_sdk.py.

from types import SimpleNamespace


class _StubMemory:
    def __init__(self):
        self.stored = []

    def retrieve(self, *args):
        return ""

    def store(self, *args):
        self.stored.append(args)


def _fake_runtime(answer):
    """A runtime whose turn is `answer(ws, session, text, turn)`; records each turn."""
    calls = []

    async def run(ws, session, text, turn, memory):
        calls.append((session.id, session.subject, turn.turn_id))
        return await answer(ws, session, text, turn)

    return SimpleNamespace(NAME="fake", answer=run, calls=calls)


def _priced(turn, cost):
    turn.saw_result(SimpleNamespace(num_turns=1, duration_ms=100, duration_api_ms=80,
                                    total_cost_usd=cost, is_error=False, model_usage={"m": {}},
                                    usage={"input_tokens": 1, "output_tokens": 2,
                                           "cache_read_input_tokens": 0,
                                           "cache_creation_input_tokens": 0}))


def _chat(monkeypatch, tmp_path, answer, *, text="hello"):
    """One turn through chat(); returns (frames, answers.jsonl records, the runtime)."""
    runtime = _fake_runtime(answer)
    monkeypatch.setattr(app_module, "ACCESS_CODE", None)
    monkeypatch.setattr(app_module, "_log_dir", tmp_path)
    monkeypatch.setattr(app_module, "memory", _StubMemory())
    monkeypatch.setattr(app_module, "runtime", runtime)
    sid = app_module.sessions.create()
    frames = []
    with TestClient(app).websocket_connect("/chat") as ws:
        ws.send_json({"session_id": sid})
        ws.receive_json()
        ws.send_json({"content": text})
        while (frame := ws.receive_json())["type"] != "done":
            frames.append(frame)
    path = tmp_path / "answers.jsonl"
    records = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    return frames, records, runtime


SECRET_TEXT = "s3cret-in-the-message"


def test_a_turn_writes_one_answer_record_and_sends_it_before_done(monkeypatch, tmp_path):
    async def answer(ws, session, text, turn):
        turn.saw_model("m")
        _priced(turn, 0.02)
        await ws.send_json({"type": "text", "content": "hi"})
        return "hi"

    frames, [record], runtime = _chat(monkeypatch, tmp_path, answer, text=f"my code is {SECRET_TEXT}")
    assert frames[-1] == {"type": "provenance", **record}
    assert record["turn_id"].startswith("turn_") and runtime.calls[0][2] == record["turn_id"]
    assert record["total_cost_usd"] == 0.02 and record["is_error"] is False
    assert SECRET_TEXT not in (tmp_path / "answers.jsonl").read_text()


def test_the_runtime_gets_the_browsers_session_and_its_subject(monkeypatch, tmp_path):
    async def answer(ws, session, text, turn):
        return "hi"

    _, _, runtime = _chat(monkeypatch, tmp_path, answer)
    [(sid, subject, _)] = runtime.calls
    assert app_module.sessions.get(sid) is not None and subject == "anonymous"


def test_an_answer_is_stored_in_memory(monkeypatch, tmp_path):
    async def answer(ws, session, text, turn):
        return "hi"

    _chat(monkeypatch, tmp_path, answer, text="hello")
    assert app_module.memory.stored[0][1:] == ("hello", "hi")


def test_a_turn_that_fails_completely_still_writes_one_record(monkeypatch, tmp_path):
    async def answer(ws, session, text, turn):
        turn.failed(RuntimeError("boom"))
        await ws.send_json({"type": "text", "content": "Sorry, an error occurred."})
        return ""

    frames, [record], _ = _chat(monkeypatch, tmp_path, answer)
    assert record["is_error"] is True and record["error"] == "RuntimeError"
    assert record["usage"] is None
    assert frames[-1]["type"] == "provenance"
    assert app_module.memory.stored == []  # nothing to remember


def test_a_log_that_cannot_be_written_does_not_stop_the_answer(monkeypatch, tmp_path, caplog):
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("")

    async def answer(ws, session, text, turn):
        _priced(turn, 0.01)
        return "hi"

    with caplog.at_level("WARNING"):
        frames, records, _ = _chat(monkeypatch, blocked, answer)
    assert records == [] and frames[-1]["type"] == "provenance"
    assert any("answers.jsonl" in r.getMessage() for r in caplog.records)


def test_new_chat_forgets_the_conversation(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "ACCESS_CODE", None)
    sid = app_module.sessions.create()
    session = app_module.sessions.get(sid)
    session.sdk_session_id, session.history = "sdk-old", ["a message"]
    with TestClient(app).websocket_connect("/chat") as ws:
        ws.send_json({"session_id": sid})
        ws.receive_json()
        ws.send_json({"type": "new_chat"})
        ws.send_json({"content": ""})  # ignored; lets the new_chat be handled first
    assert session.sdk_session_id is None and session.history == []


def test_an_idle_sessions_artifact_is_gone_without_another_chat_opening(artifact_root):
    sid = app_module.sessions.create()
    path = artifacts.new_path(sid, "obj", ".png")
    path.write_bytes(b"png")
    app_module.sessions._sessions[sid].last_active -= SESSION_IDLE_SECONDS + 1

    assert TestClient(app).get(f"/artifacts/{sid}/{path.name}").status_code == 404
    assert not path.exists()


def test_a_message_to_an_expired_session_closes_the_socket_with_4408(monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    async def answer(ws, session, text, turn):
        return "hi"

    runtime = _fake_runtime(answer)
    monkeypatch.setattr(app_module, "memory", _StubMemory())
    monkeypatch.setattr(app_module, "runtime", runtime)
    sid = app_module.sessions.create()

    with TestClient(app).websocket_connect("/chat") as ws:
        ws.send_json({"session_id": sid})
        assert ws.receive_json() == {"type": "session", "session_id": sid}
        app_module.sessions._sessions[sid].last_active -= SESSION_IDLE_SECONDS + 1
        ws.send_json({"content": "hello"})
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()

    assert closed.value.code == 4408
    assert runtime.calls == []  # no agent call for an expired session
    assert app_module.sessions.get(sid) is None  # and it was not revived


# -- Admission (T14, #72) ----------------------------------------------------

CODE = "an-access-code-16+"


@pytest.mark.parametrize("init", [{"session_id": None}, {"session_id": None, "access_code": "wrong-code-0000000"}],
                         ids=["no-code", "wrong-code"])
def test_a_new_chat_without_the_right_code_is_refused_with_4401(monkeypatch, init):
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setattr(app_module, "ACCESS_CODE", CODE)
    before = len(app_module.sessions._sessions)
    with TestClient(app).websocket_connect("/chat") as ws:
        ws.send_json(init)
        assert ws.receive_json() == {"type": "error", "code": "admission_required"}
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 4401
    assert len(app_module.sessions._sessions) == before


def test_the_right_code_opens_a_chat_and_its_id_resumes_without_one(monkeypatch):
    monkeypatch.setattr(app_module, "ACCESS_CODE", CODE)
    client = TestClient(app)
    with client.websocket_connect("/chat") as ws:
        ws.send_json({"session_id": None, "access_code": CODE})
        frame = ws.receive_json()
    assert frame["type"] == "session"
    with client.websocket_connect("/chat") as ws:
        ws.send_json({"session_id": frame["session_id"]})
        assert ws.receive_json() == frame


def test_with_admission_off_a_chat_opens_without_a_code(monkeypatch):
    monkeypatch.setattr(app_module, "ACCESS_CODE", None)
    with TestClient(app).websocket_connect("/chat") as ws:
        ws.send_json({"session_id": None})
        assert ws.receive_json()["type"] == "session"


@pytest.mark.parametrize("code", [CODE, None])
def test_the_page_can_ask_whether_a_code_is_required(monkeypatch, code):
    monkeypatch.setattr(app_module, "ACCESS_CODE", code)
    assert TestClient(app).get("/admission").json() == {"required": code is not None}


def test_the_chatbot_does_not_need_the_agent_sdk():
    # The default runtime runs without the SDK and its bundled CLI (#86).
    import subprocess

    code = ("import sys; sys.modules['claude_agent_sdk'] = None; "
            "import chatbot.app as a; assert a.runtime.NAME == 'pydantic-ai'; "
            "assert 'chatbot.runtime.sdk' not in sys.modules")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__('os').environ, "DTCC_AGENT_RUNTIME": ""})
    assert out.returncode == 0, out.stderr[-500:]
