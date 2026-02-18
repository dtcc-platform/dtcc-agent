# tests/test_chatbot_app.py
"""Smoke tests for the chatbot FastAPI app."""

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
    _mock_sdk.ResultMessage = MagicMock()
    _mock_sdk.TextBlock = MagicMock()
    _mock_sdk.ToolUseBlock = MagicMock()
    sys.modules["claude_agent_sdk"] = _mock_sdk

from fastapi.testclient import TestClient
from chatbot.app import app


def test_index_returns_html():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "DTCC Assistant" in resp.text


def test_websocket_session_handshake():
    client = TestClient(app)
    with client.websocket_connect("/chat") as ws:
        # Send init message
        ws.send_json({"session_id": None})
        # Should receive session message
        msg = ws.receive_json()
        assert msg["type"] == "session"
        assert "session_id" in msg
