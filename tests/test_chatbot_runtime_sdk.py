"""The Agent SDK runtime (#86): M2's agent loop, kept for one milestone as
M3's rollback. Skipped where the `sdk` extra is not installed."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("claude_agent_sdk")

from chatbot.provenance import TurnRecord  # noqa: E402
from chatbot.runtime import sdk  # noqa: E402
from chatbot.sessions import Session  # noqa: E402


class _Socket:
    def __init__(self):
        self.frames = []

    async def send_json(self, frame):
        self.frames.append(frame)


class _Memory:
    def __init__(self):
        self.retrieved = []

    def retrieve(self, query, session_id):
        self.retrieved.append(query)
        return ""


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

    return FakeClient


def _result(cost):
    return SimpleNamespace(num_turns=1, duration_ms=100, duration_api_ms=80, total_cost_usd=cost,
                           is_error=False, model_usage={"m": {}}, session_id="sdk-new",
                           usage={"input_tokens": 1, "output_tokens": 2,
                                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})


@pytest.fixture
def turn_with(monkeypatch):
    """Run one SDK turn on `session` with a fake client and `stream`."""
    def go(session, stream, *, fail_on_resume=False):
        seen, turn_ids = [], []
        monkeypatch.setattr(sdk, "ClaudeSDKClient", _fake_client(seen, fail_on_resume))
        monkeypatch.setattr(sdk, "build_options",
                            lambda s, sdk_id=None, ctx="", turn_id=None: turn_ids.append(turn_id)
                            or {"session_id": s.id, "sdk_session_id": sdk_id})
        monkeypatch.setattr(sdk, "stream_response", stream)
        ws, memory = _Socket(), _Memory()
        turn = TurnRecord("turn_1", session.id, session.subject, memory_context=False)
        text = asyncio.run(sdk.answer(ws, session, "hello", turn, memory))
        return text, turn, seen, turn_ids, ws.frames, memory

    return go


@pytest.mark.parametrize("resumed", [False, True], ids=["first-attempt", "fresh-retry"])
def test_every_agent_call_is_built_for_the_browsers_session(turn_with, resumed):
    async def stream(client, ws, sid, turn):
        return "sdk-new", "hi"

    session = Session(id="s1", sdk_session_id="sdk-old" if resumed else None)
    _, turn, seen, _, _, _ = turn_with(session, stream, fail_on_resume=True)

    # The resumed call fails and is retried fresh: both carry the same session.
    assert [o["session_id"] for o in seen] == ["s1"] * (2 if resumed else 1)
    assert seen[-1]["sdk_session_id"] is None
    assert session.sdk_session_id == "sdk-new" and turn.runtime == "sdk"


def test_a_turn_retried_fresh_keeps_one_turn_id_and_sums_both_attempts(turn_with):
    async def stream(client, ws, sid, turn):
        turn.saw_result(_result(0.01))
        if client.options["sdk_session_id"]:
            raise RuntimeError("context window exceeded")
        return "sdk-new", "hi"

    _, turn, _, turn_ids, _, memory = turn_with(Session(id="s1", sdk_session_id="sdk-old"), stream)
    record = turn.record()
    assert record["retried_fresh"] is True
    assert record["total_cost_usd"] == pytest.approx(0.02)
    assert record["cost_source"] == "sdk total_cost_usd"
    assert len(turn_ids) == 2 and set(turn_ids) == {"turn_1"}


def test_a_fresh_turn_that_fails_says_so(turn_with):
    async def stream(client, ws, sid, turn):
        raise RuntimeError("boom")

    text, turn, _, _, frames, _ = turn_with(Session(id="s1"), stream)
    assert text == "" and turn.error == "RuntimeError"
    assert frames[-1]["type"] == "text" and "error" in frames[-1]["content"]


def test_memory_is_read_only_when_not_resuming(turn_with):
    async def stream(client, ws, sid, turn):
        return "sdk-new", "hi"

    *_, memory = turn_with(Session(id="s1"), stream)
    assert memory.retrieved == ["hello"]
    *_, memory = turn_with(Session(id="s1", sdk_session_id="sdk-old"), stream)
    assert memory.retrieved == []


def test_the_agent_gets_no_built_in_tools_only_the_dtcc_agent_server(monkeypatch):
    # The CLI's own tools (Bash, Read, Edit, Write, Task…) would run inside
    # the chatbot container, next to its credentials. Only ToolSearch stays:
    # the agent loads the dtcc-agent tools through it.
    built = []
    monkeypatch.setattr(sdk, "ClaudeAgentOptions", lambda **kw: built.append(kw) or SimpleNamespace(**kw))
    monkeypatch.setattr(sdk, "get_mcp_server_config", lambda *a: {"dtcc-agent": {}})

    sdk.build_options(Session(id="s1"))

    [options] = built
    assert options["tools"] == ["ToolSearch"]
    assert options["strict_mcp_config"] is True
    assert set(options["mcp_servers"]) == {"dtcc-agent"}


def test_the_agent_runs_the_configured_model_on_bedrock(monkeypatch):
    built = []
    monkeypatch.setattr(sdk, "ClaudeAgentOptions", lambda **kw: built.append(kw) or SimpleNamespace(**kw))
    monkeypatch.setattr(sdk, "get_mcp_server_config", lambda *a: {"dtcc-agent": {}})
    monkeypatch.setenv("DTCC_AGENT_MODEL", "eu.anthropic.claude-sonnet-4-6")

    sdk.build_options(Session(id="s1"))

    [options] = built
    assert options["model"] == "eu.anthropic.claude-sonnet-4-6"
    assert options["env"]["CLAUDE_CODE_USE_BEDROCK"] == "1"


def test_the_session_and_its_subject_reach_the_tool_server(monkeypatch):
    configs = []
    monkeypatch.setattr(sdk, "ClaudeAgentOptions", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(sdk, "get_mcp_server_config",
                        lambda sid, subject, turn_id: configs.append((sid, subject, turn_id)) or {})

    sdk.build_options(Session(id="s1"), turn_id="turn_1")

    assert configs == [("s1", "anonymous", "turn_1")]


def test_the_sdk_runtime_sends_the_m2_prompt_byte_for_byte():
    """The rollback path keeps M2's prompt, seven schemas and all (#87)."""
    from chatbot.provenance import prompt_version

    opts = sdk.build_options(Session(id="s1"))
    assert prompt_version(opts.system_prompt) == "5fdfaaf2e87b"
    assert sdk.M2_PROMPT_VERSION == "5fdfaaf2e87b"
