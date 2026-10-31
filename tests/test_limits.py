"""Turn and spend limits (#132), driven by pydantic-ai's FunctionModel: no
test needs Bedrock. Prices are genai-prices' rates for the deployed model."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic_ai import FunctionToolset
from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from chatbot import limits
from chatbot.limits import LimitedModel, LimitReached, SpendLedger
from chatbot.provenance import TurnRecord
from chatbot.runtime import pai
from chatbot.sessions import Session

MODEL = "eu.anthropic.claude-sonnet-5-5"


class _Socket:
    def __init__(self):
        self.frames = []

    async def send_json(self, frame):
        self.frames.append(frame)


class _Memory:
    def retrieve(self, query, session_id):
        return ""


def _answering(text="Done.", *, gate: asyncio.Event | None = None, seen=None):
    async def stream(messages, info: AgentInfo):
        if seen is not None:
            seen.append(info.model_settings)
        if gate is not None:
            await gate.wait()
        yield text

    def whole(messages, info: AgentInfo):
        if seen is not None:
            seen.append(info.model_settings)
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(whole, stream_function=stream, model_name="test-model")


@pytest.fixture
def turn(monkeypatch):
    """Run one turn of the pai runtime against a model."""
    monkeypatch.setattr(pai, "toolset", lambda session, turn_id: FunctionToolset())

    async def go(model, session, text="How many buildings?"):
        monkeypatch.setattr(pai, "model", lambda: model)
        ws, record = _Socket(), TurnRecord("turn_1", session.id, session.subject, memory_context=False)
        answer = await pai.answer(ws, session, text, record, _Memory())
        return answer, ws.frames, record

    return go


# --- The ledger ---------------------------------------------------------------

def test_two_turns_that_together_pass_the_limit_are_not_both_admitted(tmp_path):
    # The issue's example: at $99 recorded, two $2 turns must not both start.
    ledger = SpendLedger(tmp_path / "spend.json", 100.0, month=lambda: "2026-10")
    ledger.spent = 99.0
    ledger.reserve(0.5)
    with pytest.raises(LimitReached, match="spending limit"):
        ledger.reserve(0.6)
    assert ledger.total() == 99.5


def test_settling_replaces_the_worst_case_with_the_cost(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", 10.0, month=lambda: "2026-10")
    key = ledger.reserve(2.0)
    ledger.settle(key, 0.25)
    assert ledger.total() == 0.25 and ledger.held == {}
    # An unknown cost keeps all that was set aside.
    ledger.settle(ledger.reserve(2.0), None)
    assert ledger.total() == 2.25


def test_after_a_restart_with_usage_outstanding_the_limit_still_holds(tmp_path):
    path = tmp_path / "spend.json"
    before = SpendLedger(path, 10.0, month=lambda: "2026-10")
    before.settle(before.reserve(3.0), 3.0)
    before.reserve(6.0)  # the server stops with this request still running

    after = SpendLedger(path, 10.0, month=lambda: "2026-10")
    assert after.total() == 9.0 and after.held == {}
    with pytest.raises(LimitReached):
        after.reserve(1.5)
    assert json.loads(path.read_text())["spent_usd"] == 9.0


def test_a_new_month_starts_from_zero_but_still_counts_what_is_running(tmp_path):
    month = ["2026-10"]
    ledger = SpendLedger(tmp_path / "spend.json", 10.0, month=lambda: month[0])
    ledger.settle(ledger.reserve(8.0), 8.0)
    running = ledger.reserve(1.0)
    month[0] = "2026-11"
    ledger.reserve(8.5)  # last month's 8 no longer counts; the running 1 does
    with pytest.raises(LimitReached):
        ledger.reserve(1.0)
    ledger.settle(running, 1.0)


def test_a_restart_in_a_new_month_still_counts_what_was_set_aside(tmp_path):
    path = tmp_path / "spend.json"
    october = SpendLedger(path, 10.0, month=lambda: "2026-10")
    october.settle(october.reserve(5.0), 5.0)
    october.reserve(4.0)  # still running when the server stops at midnight

    november = SpendLedger(path, 10.0, month=lambda: "2026-11")
    assert november.total() == 4.0  # October's 5 is behind us; the unsettled 4 is not
    with pytest.raises(LimitReached):
        november.reserve(6.5)


def test_a_request_settled_after_midnight_counts_in_the_new_month(tmp_path):
    # Before, it was added to the old month and then dropped at the next reserve.
    month = ["2026-10"]
    ledger = SpendLedger(tmp_path / "spend.json", 100.0, month=lambda: month[0])
    key = ledger.reserve(3.0)
    month[0] = "2026-11"
    ledger.settle(key, 2.0)
    with pytest.raises(LimitReached):
        ledger.reserve(99.0)
    assert ledger.total() == 2.0


def test_an_unreadable_ledger_stops_the_server(tmp_path):
    path = tmp_path / "spend.json"
    path.write_text("{not json")
    with pytest.raises(SystemExit, match="spend ledger"):
        SpendLedger(path, 10.0)


# --- The model gate -----------------------------------------------------------

def test_with_a_spend_limit_every_request_is_capped_at_the_output_limit(tmp_path):
    seen = []
    model = LimitedModel(_answering(seen=seen), MODEL, SpendLedger(tmp_path / "spend.json", 10.0))
    asyncio.run(model_request(model, [ModelRequest.user_text_prompt("hi")],
                              model_settings={"max_tokens": 100_000}))
    asyncio.run(model_request(model, [ModelRequest.user_text_prompt("hi")],
                              model_settings={"max_tokens": 1}))
    assert seen == [{"max_tokens": limits.MAX_OUTPUT_TOKENS}, {"max_tokens": 1}]


def test_without_a_spend_limit_output_is_not_capped():
    # Unset limits change nothing: the cap exists to bound the spend limit's worst case.
    seen = []
    model = LimitedModel(_answering(seen=seen), MODEL)
    asyncio.run(model_request(model, [ModelRequest.user_text_prompt("hi")]))
    asyncio.run(model_request(model, [ModelRequest.user_text_prompt("hi")],
                              model_settings={"max_tokens": 100_000}))
    assert [(s or {}).get("max_tokens") for s in seen] == [None, 100_000]


def test_with_a_spend_limit_the_bedrock_client_does_not_retry(monkeypatch, tmp_path):
    # botocore's retries happen inside one reserved request, so they'd go uncounted.
    monkeypatch.setenv("DTCC_AGENT_MONTHLY_SPEND_USD", "90")
    monkeypatch.setenv("DTCC_AGENT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "x")
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "5")
    pai.model.cache_clear()
    try:
        model = pai.model()
        assert model.wrapped.client.meta.config.retries["total_max_attempts"] == 1
    finally:
        pai.model.cache_clear()


def test_a_request_is_charged_what_it_cost_and_its_worst_case_bounds_that(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", 10.0)
    model = LimitedModel(_answering(), MODEL, ledger)
    messages = [ModelRequest.user_text_prompt("How many buildings are in Lindholmen?")]

    asyncio.run(model_request(model, messages))
    assert ledger.held == {}
    assert 0 < ledger.total() < 0.2  # the worst case of this request, $0.10 or so, was not kept


def test_concurrent_turns_near_the_limit_cannot_pass_it(tmp_path, turn):
    # Room for one request's worst case but not two: of two turns started
    # together, one answers and the other is refused, with nothing overspent.
    ledger = SpendLedger(tmp_path / "spend.json", 1.0)
    gate = asyncio.Event()
    model = LimitedModel(_answering(gate=gate), MODEL, ledger)
    worst = model.worst_case([ModelRequest.user_text_prompt("How many buildings?")],
                             {"max_tokens": limits.MAX_OUTPUT_TOKENS}, _parameters())
    ledger.spent = 1.0 - 1.5 * worst

    async def both():
        first = asyncio.create_task(turn(model, Session(id="a")))
        second = asyncio.create_task(turn(model, Session(id="b")))
        await asyncio.sleep(0.05)
        gate.set()
        return await first, await second

    results = asyncio.run(both())
    answers = sorted(r[0] for r in results)
    assert answers == ["", "Done."]
    refused = next(r for r in results if r[0] == "")
    assert refused[1] == [{"type": "text", "content": limits.SPEND_REACHED}]
    assert refused[2].error == "LimitReached"
    assert ledger.total() <= 1.0


def test_model_requests_beyond_the_concurrency_limit_wait(turn):
    gate = asyncio.Event()
    model = LimitedModel(_answering(gate=gate), MODEL, concurrency=1)

    async def two():
        first = asyncio.create_task(turn(model, Session(id="a")))
        second = asyncio.create_task(turn(model, Session(id="b")))
        await asyncio.sleep(0.05)
        waiting = model._slots.locked()
        gate.set()
        await first, await second
        return waiting

    assert asyncio.run(two()) is True


def test_a_request_bedrock_refused_costs_nothing(tmp_path):
    # Throttling, as pydantic-ai reports a Bedrock ClientError.
    from pydantic_ai.exceptions import ModelHTTPError as ClientError

    async def refuse(messages, info):
        raise ClientError(status_code=429, model_name=MODEL, body="ThrottlingException")
        yield ""  # pragma: no cover

    ledger = SpendLedger(tmp_path / "spend.json", 10.0)
    model = LimitedModel(FunctionModel(stream_function=refuse, model_name="m"), MODEL, ledger)

    async def go():
        async with model.request_stream([ModelRequest.user_text_prompt("hi")], None, _parameters()):
            pass

    with pytest.raises(ClientError):
        asyncio.run(go())
    assert ledger.total() == 0.0


def test_a_request_stopped_midway_keeps_its_worst_case(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", 10.0)
    model = LimitedModel(_answering(gate=asyncio.Event()), MODEL, ledger)
    messages = [ModelRequest.user_text_prompt("hi")]
    worst = model.worst_case(messages, {"max_tokens": limits.MAX_OUTPUT_TOKENS}, _parameters())

    async def go():
        async with model.request_stream(messages, None, _parameters()) as stream:
            async for _ in stream:
                pass

    async def stop():
        task = asyncio.create_task(go())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(stop())
    assert ledger.total() == pytest.approx(worst)


def test_a_4xx_after_bedrock_answered_keeps_the_worst_case(tmp_path):
    # Once the stream is open, billing may have begun: only a refusal before
    # Bedrock answered is free.
    from pydantic_ai.exceptions import ModelHTTPError

    async def midway(messages, info):
        yield "Part of an answer"
        raise ModelHTTPError(status_code=400, model_name=MODEL, body="ValidationException")

    ledger = SpendLedger(tmp_path / "spend.json", 10.0)
    model = LimitedModel(FunctionModel(stream_function=midway, model_name="m"), MODEL, ledger)

    async def go():
        async with model.request_stream([ModelRequest.user_text_prompt("hi")], None, _parameters()) as s:
            async for _ in s:
                pass

    with pytest.raises(ModelHTTPError):
        asyncio.run(go())
    assert ledger.total() > 0


def test_an_error_that_merely_follows_a_4xx_keeps_the_worst_case(tmp_path):
    # A 4xx in an exception's context is not the request being refused.
    from pydantic_ai.exceptions import ModelHTTPError

    async def failing(messages, info):
        try:
            raise ModelHTTPError(status_code=429, model_name=MODEL, body="earlier")
        except ModelHTTPError:
            raise RuntimeError("something else broke")
        yield ""  # pragma: no cover

    ledger = SpendLedger(tmp_path / "spend.json", 10.0)
    model = LimitedModel(FunctionModel(stream_function=failing, model_name="m"), MODEL, ledger)

    async def go():
        async with model.request_stream([ModelRequest.user_text_prompt("hi")], None, _parameters()):
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(go())
    assert ledger.total() > 0


@pytest.mark.parametrize("state", [
    {"month": "2026-10", "spent_usd": -50.0, "held_usd": {}},
    {"month": "2026-10", "spent_usd": float("nan"), "held_usd": {}},
    {"month": "2026-10", "spent_usd": 1.0, "held_usd": {"k": "lots"}},
    {"month": "2026-10"},
])
def test_a_ledger_with_implausible_values_stops_the_server(tmp_path, state):
    path = tmp_path / "spend.json"
    path.write_text(json.dumps(state))
    with pytest.raises(SystemExit, match="unexpected values"):
        SpendLedger(path, 10.0, month=lambda: "2026-10")


def test_the_ledger_leaves_no_temporary_files(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", 10.0)
    ledger.settle(ledger.reserve(1.0), 0.5)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["spend.json"]


def test_a_spend_limit_needs_an_explicit_log_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("DTCC_AGENT_MONTHLY_SPEND_USD", "90")
    monkeypatch.delenv("DTCC_AGENT_LOG_DIR", raising=False)
    with pytest.raises(SystemExit, match="DTCC_AGENT_LOG_DIR"):
        limits.ledger(tmp_path)
    monkeypatch.setenv("DTCC_AGENT_LOG_DIR", str(tmp_path))
    assert limits.ledger(tmp_path).limit == 90.0


def test_a_spend_limit_on_a_model_without_a_price_stops_the_server(tmp_path):
    with pytest.raises(SystemExit, match="no known price"):
        LimitedModel(_answering(), "no-such-model", SpendLedger(tmp_path / "s.json", 10.0))


# --- Turn limits --------------------------------------------------------------

def test_a_chat_past_its_turn_limit_is_refused(monkeypatch, turn):
    monkeypatch.setenv("DTCC_AGENT_TURNS_PER_CHAT", "2")
    session, model = Session(id="s1"), LimitedModel(_answering(), MODEL)

    async def three():
        return [await turn(model, session, f"Question {i}") for i in range(3)]

    results = asyncio.run(three())
    assert [r[0] for r in results] == ["Done.", "Done.", ""]
    assert results[2][1] == [{"type": "text", "content": "This chat has reached its limit of 2 questions."}]
    assert results[2][2].error == "LimitReached"
    # A new chat in the page keeps the session, and its count.
    session.reset_conversation()
    assert asyncio.run(turn(model, session))[0] == ""


def test_a_turn_past_its_deadline_is_stopped(monkeypatch, turn):
    monkeypatch.setenv("DTCC_AGENT_TURN_SECONDS", "1")
    gate = asyncio.Event()  # never set: the model never answers
    answer, frames, record = asyncio.run(turn(LimitedModel(_answering(gate=gate), MODEL), Session(id="s1")))

    assert answer == ""
    assert frames == [{"type": "text", "content": limits.TURN_TIMED_OUT.format(seconds=1)}]
    assert record.error == "TimeoutError" and record.retried_fresh is False


def test_another_timeout_inside_a_turn_is_an_error_not_the_deadline(turn):
    # No deadline set: a TimeoutError from elsewhere (a connection, say) must
    # not be reported as the turn running out of time.
    async def slow(messages, info):
        raise TimeoutError("connect timed out")
        yield ""  # pragma: no cover

    model = LimitedModel(FunctionModel(stream_function=slow, model_name="m"), MODEL)
    answer, frames, record = asyncio.run(turn(model, Session(id="s1")))
    assert frames == [{"type": "text", "content": "Sorry, an error occurred. Check the server logs for details."}]
    assert record.error == "TimeoutError"


@pytest.mark.parametrize("name, value", [
    ("DTCC_AGENT_MONTHLY_SPEND_USD", "lots"),
    ("DTCC_AGENT_MONTHLY_SPEND_USD", "nan"),  # nan compares false: the limit would be off
    ("DTCC_AGENT_MONTHLY_SPEND_USD", "inf"),
    ("DTCC_AGENT_TURN_SECONDS", "nan"),
    ("DTCC_AGENT_TURNS_PER_CHAT", "0"),
    ("DTCC_AGENT_MODEL_CONCURRENCY", "1.5"),
])
def test_a_bad_limit_setting_stops_the_server(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit, match=name):
        limits.describe()


def test_unset_limits_are_off(monkeypatch):
    for name in ("DTCC_AGENT_MONTHLY_SPEND_USD", "DTCC_AGENT_TURNS_PER_CHAT",
                 "DTCC_AGENT_TURN_SECONDS", "DTCC_AGENT_MODEL_CONCURRENCY"):
        monkeypatch.setenv(name, "")
    assert limits.describe() == ("no spend limit, no turn limit, no turn deadline, "
                                 "no model concurrency limit")


def _parameters():
    from pydantic_ai.models import ModelRequestParameters

    return ModelRequestParameters()
