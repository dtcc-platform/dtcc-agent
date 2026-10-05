"""The pydantic-ai runtime (#86): one agent loop in the chatbot process, on
Bedrock in production, driven here by pydantic-ai's FunctionModel and a stub
toolset so no test needs a model or a network."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic_ai import FunctionToolset
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.usage import RequestUsage

from chatbot.config import SYSTEM_PROMPT
from chatbot.provenance import TurnRecord, prompt_version
from chatbot.runtime import pai
from chatbot.sessions import Session

IMAGE = "abc_obj_1.png"
CATALOGUE = '[{"name": "datasets.buildings", "description": "Download 3D buildings."}]'
REAL_FETCH = pai.fetch_catalogue


@pytest.fixture(autouse=True)
def catalogue(monkeypatch):
    """Every test starts with no catalogue memoised; the fetch returns
    CATALOGUE and counts its calls. `fail` makes the next that-many raise."""
    monkeypatch.setattr(pai, "_catalogue", pai._Memo())
    monkeypatch.delenv("DTCC_AGENT_CATALOGUE", raising=False)
    state = {"fetches": 0, "fail": 0}

    async def fetch(tools, variant):
        state["fetches"] += 1
        await asyncio.sleep(0.01)
        if state["fail"]:
            state["fail"] -= 1
            raise ConnectionError("MCP server down")
        return CATALOGUE

    monkeypatch.setattr(pai, "fetch_catalogue", fetch)
    return state


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
        return "Earlier: the user asked about Lindholmen."


def _tools(calls):
    toolset = FunctionToolset()

    @toolset.tool_plain
    def render_object(ref: str) -> str:
        """Render an object."""
        calls.append(ref)
        return json.dumps({"artifact": {"name": IMAGE, "kind": "image"}})

    return toolset


def _model(seen, *, render=True, fail=0):
    """Asks for render_object once, then answers; records every request.
    `fail` makes the first that-many requests raise."""
    state = {"failures": fail}

    async def stream(messages, info: AgentInfo):
        seen.append((messages, info))
        if state["failures"]:
            state["failures"] -= 1
            raise RuntimeError("context too long")
        if render and not any(p.part_kind == "tool-return" for m in messages for p in m.parts):
            yield {0: DeltaToolCall(name="render_object", json_args='{"ref": "obj_1"}')}
            return
        yield "Here are the buildings."

    return FunctionModel(stream_function=stream, model_name="test-model")


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Run one turn of the pai runtime against `model` and the stub tools."""
    from dtcc_agent import artifacts

    (tmp_path / IMAGE).write_bytes(b"png")
    monkeypatch.setattr(artifacts, "find", lambda sid, name: tmp_path / name)
    calls: list[str] = []
    monkeypatch.setattr(pai, "toolset", lambda session, turn_id: _tools(calls))

    def go(model, session, text="Show the buildings", memory=None):
        ws, turn = _Socket(), TurnRecord("turn_1", session.id, session.subject, memory_context=False)
        memory = memory or _Memory()
        monkeypatch.setattr(pai, "model", lambda: model)
        answer = asyncio.run(pai.answer(ws, session, text, turn, memory))
        return answer, ws.frames, turn, memory

    go.calls = calls
    return go


def test_a_turn_streams_text_tool_and_image_frames(run):
    seen = []
    answer, frames, turn, _ = run(_model(seen), Session(id="s1"))

    assert answer == "Here are the buildings."
    kinds = [f["type"] for f in frames]
    assert kinds == ["tool_call", "image", "text"]
    assert frames[0] == {"type": "tool_call", "name": "render_object", "status": "running"}
    assert frames[1] == {"type": "image", "url": f"/artifacts/s1/{IMAGE}"}
    assert run.calls == ["obj_1"]
    assert turn.tools_called == ["render_object"]
    assert turn.model == "test-model"


def test_a_follow_up_sends_the_earlier_messages_and_new_chat_clears_them(run):
    session = Session(id="s1")
    run(_model([]), session)
    assert session.history

    seen = []
    run(_model(seen, render=False), session, text="And how tall?")
    first_request = seen[0][0]
    texts = [p.content for m in first_request for p in m.parts if p.part_kind == "user-prompt"]
    assert texts == ["Show the buildings", "And how tall?"]

    session.reset_conversation()
    assert session.history == [] and session.sdk_session_id is None


def test_memory_is_read_only_for_a_fresh_conversation(run):
    session = Session(id="s1")
    seen = []
    _, _, turn, memory = run(_model(seen, render=False), session)
    assert memory.retrieved == ["Show the buildings"] and turn.memory_context is True
    # The memory goes in as instructions after the static prompt.
    assert "Earlier: the user asked about Lindholmen." in seen[0][1].instructions

    _, _, turn, memory = run(_model([], render=False), session, text="And?")
    assert memory.retrieved == [] and turn.memory_context is False


def test_a_failed_follow_up_retries_fresh_and_sums_both_attempts(run):
    session = Session(id="s1")
    run(_model([], render=False), session)

    # Attempt 1 completes a tool-call request, then its next request fails
    # (a context too long, say). The retry starts fresh and answers.
    seen, calls = [], [0]

    async def stream(messages, info):
        seen.append(messages)
        calls[0] += 1
        if calls[0] == 1:
            yield {0: DeltaToolCall(name="render_object", json_args='{"ref": "obj_1"}')}
        elif calls[0] == 2:
            raise RuntimeError("context too long")
        else:
            yield "Here are the buildings."

    answer, frames, turn, memory = run(FunctionModel(stream_function=stream, model_name="test-model"),
                                       session, text="Again")
    assert answer == "Here are the buildings."
    assert turn.retried_fresh is True and turn.error is None
    # The retry starts over, and a fresh conversation reads memory.
    retry_prompt = [p.content for m in seen[-1] for p in m.parts if p.part_kind == "user-prompt"]
    assert retry_prompt == ["Again"] and memory.retrieved == ["Again"]
    # The request that completed before the failure still counts.
    assert turn.totals["num_turns"] == 2


def test_a_fresh_turn_that_fails_says_so_and_records_the_error_class(run):
    answer, frames, turn, _ = run(_model([], fail=5), Session(id="s1"))
    assert answer == ""
    assert frames[-1]["type"] == "text" and "error" in frames[-1]["content"]
    assert turn.error == "RuntimeError" and turn.retried_fresh is False


def test_a_turn_that_fails_before_any_model_request_records_no_usage(run, monkeypatch):
    def unreachable(session, turn_id):
        raise ConnectionError("MCP server down")

    monkeypatch.setattr(pai, "toolset", unreachable)
    _, _, turn, _ = run(_model([]), Session(id="s1"))
    record = turn.record()
    assert record["error"] == "ConnectionError"
    assert record["usage"] is None and record["num_turns"] is None and record["cost_source"] is None


def test_two_turns_on_one_session_run_one_after_the_other(run, monkeypatch):
    session = Session(id="s1")
    active, peak = [0], [0]

    async def stream(messages, info):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        await asyncio.sleep(0.05)
        active[0] -= 1
        yield "ok"

    model = FunctionModel(stream_function=stream, model_name="test-model")

    async def both():
        turns = [TurnRecord(f"turn_{i}", "s1", "anonymous", memory_context=False) for i in (1, 2)]
        await asyncio.gather(*(pai.answer(_Socket(), session, "hi", t, _Memory()) for t in turns))

    monkeypatch.setattr(pai, "model", lambda: model)
    asyncio.run(both())
    assert peak[0] == 1
    assert len([m for m in session.history if m.kind == "response"]) == 2


# -- The catalogue in the cached prefix (#87) ------------------------------------

def _parts(seen):
    """The first request's instruction parts, as (text, dynamic)."""
    return [(p.content, p.dynamic) for p in seen[0][1].model_request_parameters.instruction_parts]


def test_instructions_are_base_then_catalogue_static_then_memory_dynamic(run):
    seen = []
    _, _, turn, _ = run(_model(seen, render=False), Session(id="s1"))
    parts = _parts(seen)
    assert parts == [(SYSTEM_PROMPT, False),
                     (f"{pai.CATALOGUE_LINES['summary']}\n\n{CATALOGUE}", False),
                     ("Earlier: the user asked about Lindholmen.", True)]
    assert turn.catalogue_in_prompt is True and turn.catalogue_variant == "summary"
    # The version is the hash of exactly the static text sent, memory left out.
    assert turn.prompt_version == prompt_version("\n\n".join(text for text, _ in parts[:2]))


def test_the_full_variant_says_not_to_look_operations_up(run, monkeypatch):
    monkeypatch.setenv("DTCC_AGENT_CATALOGUE", "full")
    seen = []
    _, _, turn, _ = run(_model(seen, render=False), Session(id="s1"))
    assert _parts(seen)[1][0].startswith(pai.CATALOGUE_LINES["full"])
    assert turn.catalogue_variant == "full"


def test_an_unknown_variant_refuses_to_run(monkeypatch):
    monkeypatch.setenv("DTCC_AGENT_CATALOGUE", "everything")
    with pytest.raises(SystemExit, match="summary, full"):
        pai.catalogue_variant()


def test_prompt_version_changes_with_one_operation_more():
    one = pai.static_instructions("summary", CATALOGUE)
    two = pai.static_instructions("summary", CATALOGUE[:-1] + ', {"name": "datasets.terrain"}]')
    assert prompt_version("\n\n".join(one)) != prompt_version("\n\n".join(two))


def test_without_a_catalogue_the_prompt_says_how_to_find_operations(run, catalogue):
    catalogue["fail"] = 1
    seen = []
    _, _, turn, _ = run(_model(seen, render=False), Session(id="s1"))
    static = [text for text, dynamic in _parts(seen) if not dynamic]
    assert static == [SYSTEM_PROMPT, pai.DISCOVERY_LINE]
    assert not any(line in text for line in pai.CATALOGUE_LINES.values() for text in static)
    assert turn.catalogue_in_prompt is False
    assert turn.prompt_version == prompt_version(f"{SYSTEM_PROMPT}\n\n{pai.DISCOVERY_LINE}")


def test_a_failed_fetch_is_tried_again_next_turn(run, catalogue):
    catalogue["fail"] = 1
    _, _, first, _ = run(_model([], render=False), Session(id="s1"))
    _, _, second, _ = run(_model([], render=False), Session(id="s2"))
    _, _, third, _ = run(_model([], render=False), Session(id="s3"))
    assert (first.catalogue_in_prompt, second.catalogue_in_prompt) == (False, True)
    assert first.error is None  # the turn answered without it
    assert third.catalogue_in_prompt is True and catalogue["fetches"] == 2  # then memoised


def test_concurrent_first_turns_fetch_the_catalogue_once(monkeypatch, catalogue):
    monkeypatch.setattr(pai, "toolset", lambda session, turn_id: _tools([]))
    monkeypatch.setattr(pai, "model", lambda: _model([], render=False))

    async def three():
        turns = [TurnRecord(f"turn_{i}", f"s{i}", "anonymous", memory_context=False) for i in range(3)]
        await asyncio.gather(*(pai.answer(_Socket(), Session(id=t.session_id), "hi", t, _Memory())
                               for t in turns))
        return turns

    turns = asyncio.run(three())
    assert catalogue["fetches"] == 1
    assert all(t.catalogue_in_prompt for t in turns)


# -- Usage and price ------------------------------------------------------------

def test_usage_maps_to_fresh_input_plus_cache_tokens():
    turn = TurnRecord("turn_1", "s1", "anonymous", memory_context=False)
    usage = pai.RunUsage(requests=3, input_tokens=3010, cache_read_tokens=1000,
                         cache_write_tokens=2000, output_tokens=20)
    turn.saw_run(usage, elapsed_ms=1234, cost=0.05, cost_source="genai-prices 0.1.9")
    record = turn.record()
    assert record["usage"] == {"input_tokens": 10, "output_tokens": 20,
                               "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 2000}
    assert record["num_turns"] == 3 and record["duration_ms"] == 1234
    assert record["duration_api_ms"] is None
    assert record["total_cost_usd"] == 0.05 and record["cost_source"] == "genai-prices 0.1.9"


def test_the_record_names_the_pydantic_ai_runtime(run):
    _, _, turn, _ = run(_model([], render=False), Session(id="s1"))
    record = turn.record()
    assert record["runtime"] == "pydantic-ai" and record["provider"] == "bedrock"
    assert record["sdk"].startswith("pydantic-ai-slim ")


# -- Against the real MCP server --------------------------------------------------

@pytest.mark.parametrize("variant", pai.CATALOGUE_VARIANTS)
def test_the_model_sees_exactly_the_dtcc_agent_tools(monkeypatch, tmp_path, variant):
    """The real stdio MCP server through pydantic-ai's MCP client: the model's
    tool list is the server's 22 tools and nothing else, a call works, and the
    first request already carries the catalogue the server lists."""
    monkeypatch.setattr(pai, "fetch_catalogue", REAL_FETCH)
    monkeypatch.setenv("DTCC_AGENT_CATALOGUE", variant)
    monkeypatch.delenv("DTCC_MCP_URL", raising=False)
    monkeypatch.setenv("DTCC_AGENT_ARTIFACTS_DIR", str(tmp_path))
    seen = []

    instructions = []

    async def stream(messages, info: AgentInfo):
        seen.append(sorted(t.name for t in info.function_tools))
        instructions.append(info.instructions)
        if len(seen) == 1:
            yield {0: DeltaToolCall(name="list_objects", json_args="{}")}
            return
        yield "Nothing stored yet."

    monkeypatch.setattr(pai, "model", lambda: FunctionModel(stream_function=stream, model_name="test-model"))
    ws, turn = _Socket(), TurnRecord("turn_1", "s1", "anonymous", memory_context=False)
    answer = asyncio.run(pai.answer(ws, Session(id="s1"), "What do I have?", turn, _Memory()))

    assert answer == "Nothing stored yet.", turn.error
    assert len(seen[0]) == 22 and "run_operation" in seen[0] and "geocode" in seen[0]
    assert not {"Bash", "Read", "Write", "Edit", "ToolSearch"} & set(seen[0])
    assert turn.tools_called == ["list_objects"]
    assert turn.catalogue_in_prompt is True
    assert pai.CATALOGUE_LINES[variant] in instructions[0]
    assert "datasets.point_cloud" in instructions[0]
    # The full variant carries each operation's parameters.
    assert ('"params"' in instructions[0]) is (variant == "full")
