import sys

from chatbot.config import SYSTEM_PROMPT, get_mcp_server_config


def test_system_prompt_mentions_sweden():
    assert "Sweden" in SYSTEM_PROMPT


def test_mcp_server_config_has_command():
    config = get_mcp_server_config("s1")
    assert "dtcc-agent" in config
    dtcc = config["dtcc-agent"]
    assert "command" in dtcc
    assert "args" in dtcc
    assert dtcc["command"] == sys.executable
    assert dtcc["args"] == ["-m", "dtcc_agent"]


def test_mcp_server_config_honors_python_override(monkeypatch):
    monkeypatch.setenv("DTCC_AGENT_PYTHON", "/custom/python")
    config = get_mcp_server_config("s1")
    assert config["dtcc-agent"]["command"] == "/custom/python"


def test_mcp_server_config_carries_the_session_over_http(monkeypatch):
    # ADR-0004: the Session id travels browser -> chatbot -> MCP server.
    monkeypatch.setenv("DTCC_MCP_URL", "http://mcp:8051/mcp")
    config = get_mcp_server_config("s1")
    assert config["dtcc-agent"] == {
        "type": "http",
        "url": "http://mcp:8051/mcp",
        "headers": {"X-DTCC-Session": "s1"},
    }
