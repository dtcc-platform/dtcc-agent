import sys

from chatbot.config import SYSTEM_PROMPT, get_mcp_server_config


def test_system_prompt_mentions_sweden():
    assert "Sweden" in SYSTEM_PROMPT


def test_mcp_server_config_has_command(monkeypatch):
    monkeypatch.delenv("DTCC_MCP_URL", raising=False)
    config = get_mcp_server_config("s1")
    assert "dtcc-agent" in config
    dtcc = config["dtcc-agent"]
    assert "command" in dtcc
    assert "args" in dtcc
    assert dtcc["command"] == sys.executable
    assert dtcc["args"] == ["-m", "dtcc_agent"]


def test_mcp_server_config_honors_python_override(monkeypatch):
    monkeypatch.delenv("DTCC_MCP_URL", raising=False)
    monkeypatch.setenv("DTCC_AGENT_PYTHON", "/custom/python")
    config = get_mcp_server_config("s1")
    assert config["dtcc-agent"]["command"] == "/custom/python"


def test_mcp_server_config_carries_the_session_over_http(monkeypatch):
    # ADR-0004: the Session id travels browser -> chatbot -> MCP server.
    monkeypatch.setenv("DTCC_MCP_URL", "http://mcp:8051/mcp")
    monkeypatch.delenv("DTCC_MCP_SECRET", raising=False)
    config = get_mcp_server_config("s1")
    assert config["dtcc-agent"] == {
        "type": "http",
        "url": "http://mcp:8051/mcp",
        "headers": {"X-DTCC-Session": "s1", "X-DTCC-Subject": "anonymous"},
    }


def test_mcp_server_config_names_the_session_over_stdio(monkeypatch):
    # The stdio server writes artifacts under this Session's directory.
    monkeypatch.delenv("DTCC_MCP_URL", raising=False)
    config = get_mcp_server_config("s1")
    assert config["dtcc-agent"]["env"]["DTCC_AGENT_SESSION"] == "s1"


def test_mcp_http_config_carries_the_secret_and_subject(monkeypatch):
    monkeypatch.setenv("DTCC_MCP_URL", "http://mcp:8051/mcp")
    monkeypatch.setenv("DTCC_MCP_SECRET", "s" * 32)
    headers = get_mcp_server_config("s1", "anonymous")["dtcc-agent"]["headers"]
    assert headers == {
        "X-DTCC-Session": "s1",
        "X-DTCC-Subject": "anonymous",
        "Authorization": "Bearer " + "s" * 32,
    }


def test_mcp_http_config_sends_no_authorization_without_a_secret(monkeypatch):
    monkeypatch.setenv("DTCC_MCP_URL", "http://mcp:8051/mcp")
    monkeypatch.delenv("DTCC_MCP_SECRET", raising=False)
    assert "Authorization" not in get_mcp_server_config("s1")["dtcc-agent"]["headers"]


def test_mcp_stdio_config_names_the_subject(monkeypatch):
    monkeypatch.delenv("DTCC_MCP_URL", raising=False)
    env = get_mcp_server_config("s1", "anonymous")["dtcc-agent"]["env"]
    assert env["DTCC_AGENT_SUBJECT"] == "anonymous"


def test_an_access_code_shorter_than_16_characters_stops_the_chatbot(monkeypatch):
    import pytest
    from chatbot.config import load_access_code

    monkeypatch.setenv("DTCC_AGENT_ACCESS_CODE", "short")
    with pytest.raises(SystemExit, match="DTCC_AGENT_ACCESS_CODE"):
        load_access_code()


def test_no_access_code_disables_admission_with_a_warning(monkeypatch, caplog):
    from chatbot.config import load_access_code

    monkeypatch.delenv("DTCC_AGENT_ACCESS_CODE", raising=False)
    with caplog.at_level("WARNING"):
        assert load_access_code() is None
    assert [r.levelname for r in caplog.records if "DTCC_AGENT_ACCESS_CODE" in r.getMessage()] == ["WARNING"]


def test_a_long_enough_access_code_is_used(monkeypatch):
    from chatbot.config import load_access_code

    monkeypatch.setenv("DTCC_AGENT_ACCESS_CODE", "c" * 16)
    assert load_access_code() == "c" * 16


def test_the_turn_travels_to_the_mcp_server(monkeypatch):
    monkeypatch.setenv("DTCC_MCP_URL", "http://mcp:8051/mcp")
    assert get_mcp_server_config("s1", "anonymous", "turn_1a2b3c4d")["dtcc-agent"]["headers"][
        "X-DTCC-Turn"] == "turn_1a2b3c4d"
    monkeypatch.delenv("DTCC_MCP_URL")
    assert get_mcp_server_config("s1", "anonymous", "turn_1a2b3c4d")["dtcc-agent"]["env"][
        "DTCC_AGENT_TURN"] == "turn_1a2b3c4d"
