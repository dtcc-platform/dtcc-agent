# tests/test_chatbot_app.py
"""Smoke tests for the chatbot FastAPI app."""

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

# claude_agent_sdk is an optional dependency that may not be installed.
# The chatbot.app module imports it at module level, so we must provide
# a mock module *before* importing chatbot.app.
_need_mock = "claude_agent_sdk" not in sys.modules
if _need_mock:
    _mock_sdk = types.ModuleType("claude_agent_sdk")
    _mock_sdk.ClaudeSDKClient = MagicMock()
    _mock_sdk.ClaudeAgentOptions = MagicMock()
    _mock_sdk.AssistantMessage = MagicMock()
    _mock_sdk.UserMessage = MagicMock()
    _mock_sdk.ResultMessage = MagicMock()
    _mock_sdk.SystemMessage = MagicMock()
    _mock_sdk.TextBlock = MagicMock()
    _mock_sdk.ThinkingBlock = MagicMock()
    _mock_sdk.ToolUseBlock = MagicMock()
    _mock_sdk.ToolResultBlock = MagicMock()
    sys.modules["claude_agent_sdk"] = _mock_sdk

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

    assert app_module._artifact_frame("s1", content) == {
        "type": "image", "url": f"/artifacts/s1/{path.name}",
    }


def test_a_tool_result_with_a_file_artifact_becomes_a_download_frame(artifact_root):
    path = _artifact("s1", "obj_1", ".csv")
    content = json.dumps({"artifact": artifacts.describe(path)})

    assert app_module._artifact_frame("s1", content) == {
        "type": "file", "url": f"/artifacts/s1/{path.name}", "name": "obj_1.csv",
    }


def test_an_artifact_inside_fastmcps_structured_result_is_found(artifact_root):
    # The shape the Claude CLI actually delivers (seen end to end).
    path = _artifact("s1", "obj_1", ".png")
    tool_text = json.dumps({"object_ref": "obj_1", "artifact": artifacts.describe(path)})

    frame = app_module._artifact_frame("s1", json.dumps({"result": tool_text}))

    assert frame == {"type": "image", "url": f"/artifacts/s1/{path.name}"}


@pytest.mark.parametrize("content", [
    "not json", json.dumps({"result": "not json"}), json.dumps({"object_ref": "obj_1"}), json.dumps(["x"]), None,
    json.dumps({"artifact": {"name": "0" * 32 + "_x.png", "kind": "image"}}),
])
def test_a_tool_result_without_this_sessions_artifact_sends_no_frame(artifact_root, content):
    assert app_module._artifact_frame("s1", content) is None


# -- The browser's session id reaches the MCP server (T5 audit gap) -----------

class _StubMemory:
    def retrieve(self, *args):
        return ""

    def store(self, *args):
        return None


def _fake_client(seen, fail_on_resume):
    class FakeClient:
        def __init__(self, options):
            self.options = options

        async def __aenter__(self):
            seen.append(self.options)
            if fail_on_resume and self.options["sdk_session_id"]:
                raise RuntimeError("context window exceeded")
            return self

        async def __aexit__(self, *exc):
            return False

        async def query(self, text):
            return None

        async def receive_response(self):
            if False:
                yield None

    return FakeClient


@pytest.mark.parametrize("resumed", [False, True], ids=["first-attempt", "fresh-retry"])
def test_chat_builds_every_agent_call_for_the_browsers_session(monkeypatch, resumed):
    seen = []
    monkeypatch.setattr(app_module, "memory", _StubMemory())
    monkeypatch.setattr(app_module, "_build_options",
                        lambda sid, sdk=None, ctx="": {"session_id": sid, "sdk_session_id": sdk})
    monkeypatch.setattr(app_module, "ClaudeSDKClient", _fake_client(seen, fail_on_resume=True))
    sid = app_module.sessions.create()
    if resumed:
        app_module.sessions.set_sdk_session(sid, "sdk-old")

    with TestClient(app).websocket_connect("/chat") as ws:
        ws.send_json({"session_id": sid})
        assert ws.receive_json() == {"type": "session", "session_id": sid}
        ws.send_json({"content": "hello"})
        while ws.receive_json()["type"] != "done":
            pass

    # The resumed call fails and is retried fresh: both carry the same session.
    assert [o["session_id"] for o in seen] == [sid] * (2 if resumed else 1)
    assert seen[-1]["sdk_session_id"] is None


def test_an_idle_sessions_artifact_is_gone_without_another_chat_opening(artifact_root):
    sid = app_module.sessions.create()
    path = artifacts.new_path(sid, "obj", ".png")
    path.write_bytes(b"png")
    app_module.sessions._sessions[sid].last_active -= SESSION_IDLE_SECONDS + 1

    assert TestClient(app).get(f"/artifacts/{sid}/{path.name}").status_code == 404
    assert not path.exists()


def test_a_message_to_an_expired_session_closes_the_socket_with_4408(monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    seen = []
    monkeypatch.setattr(app_module, "memory", _StubMemory())
    monkeypatch.setattr(app_module, "ClaudeSDKClient", _fake_client(seen, fail_on_resume=False))
    sid = app_module.sessions.create()

    with TestClient(app).websocket_connect("/chat") as ws:
        ws.send_json({"session_id": sid})
        assert ws.receive_json() == {"type": "session", "session_id": sid}
        app_module.sessions._sessions[sid].last_active -= SESSION_IDLE_SECONDS + 1
        ws.send_json({"content": "hello"})
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()

    assert closed.value.code == 4408
    assert seen == []  # no agent call for an expired session
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


def test_a_chat_session_is_anonymous_and_its_tool_server_is_told_so(monkeypatch):
    configs = []
    monkeypatch.setattr(app_module, "ACCESS_CODE", None)
    monkeypatch.setattr(app_module, "memory", _StubMemory())
    monkeypatch.setattr(app_module, "ClaudeSDKClient", _fake_client([], fail_on_resume=False))
    monkeypatch.setattr(app_module, "get_mcp_server_config",
                        lambda sid, subject: configs.append((sid, subject)) or {})
    sid = app_module.sessions.create()
    assert app_module.sessions.get(sid).subject == "anonymous"

    with TestClient(app).websocket_connect("/chat") as ws:
        ws.send_json({"session_id": sid})
        ws.receive_json()
        ws.send_json({"content": "hello"})
        while ws.receive_json()["type"] != "done":
            pass

    assert configs == [(sid, "anonymous")]
