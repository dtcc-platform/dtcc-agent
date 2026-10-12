"""A rejected Bedrock key: the chat says whom to ask, the record keeps the
HTTP status, and a startup probe logs it before any user meets it."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import FunctionModel

from chatbot import provenance
from chatbot.runtime import pai
from chatbot.sessions import Session
from tests.test_chatbot_runtime_pai import _model, run  # noqa: F401 - run is a fixture


def _rejected(seen, status=403):
    """Bedrock refuses the server's credentials: pydantic-ai raises ModelHTTPError."""

    async def stream(messages, info):
        seen.append(messages)
        raise ModelHTTPError(status_code=status, model_name="test-model",
                             body={"message": "The security token included in the request is invalid."})
        yield  # pragma: no cover - makes this an async generator

    return FunctionModel(stream_function=stream, model_name="test-model")


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_tells_the_user_to_ask_the_admin(run, status):  # noqa: F811
    answer, frames, turn, _ = run(_rejected([], status), Session(id="s1"))
    assert answer == "" and frames[-1] == {"type": "text", "content": pai.AUTH_FAILURE}
    record = turn.record()
    assert record["error"] == "ModelHTTPError" and record["http_status"] == status


def test_a_rejected_key_on_a_follow_up_is_not_retried_fresh(run):  # noqa: F811
    session = Session(id="s1")
    run(_model([], render=False), session)

    seen = []
    _, frames, turn, _ = run(_rejected(seen), session, text="And now?")
    assert turn.retried_fresh is False and len(seen) == 1
    assert frames[-1]["content"] == pai.AUTH_FAILURE


def test_another_failure_still_says_sorry_and_records_no_status(run):  # noqa: F811
    _, frames, turn, _ = run(_model([], fail=5), Session(id="s1"))
    assert frames[-1]["content"].startswith("Sorry, an error occurred")
    assert turn.record()["http_status"] is None


def test_the_http_status_is_found_inside_groups_and_causes():
    inner = ModelHTTPError(status_code=401, model_name="m", body=None)
    try:
        raise RuntimeError("wrapped") from inner
    except RuntimeError as wrapped:
        group = ExceptionGroup("tasks", [ValueError("other"), wrapped])
    assert provenance.http_status(group) == 401
    assert provenance.http_status(RuntimeError("plain")) is None


@pytest.mark.parametrize("status, level, words", [
    (403, "ERROR", "REJECTED THE SERVER'S KEY"),
    (500, "WARNING", "Bedrock check failed"),
    (None, "INFO", "Bedrock check passed"),
])
def test_the_startup_probe_logs_what_bedrock_said(monkeypatch, caplog, status, level, words):
    import pydantic_ai.direct

    async def request(model, messages, *, model_settings=None):
        assert model_settings == {"max_tokens": 1}
        if status:
            raise ModelHTTPError(status_code=status, model_name="m", body=None)

    monkeypatch.setattr(pydantic_ai.direct, "model_request", request)
    monkeypatch.setattr(pai, "model", lambda: "a-model")
    with caplog.at_level("INFO", logger="lurkie"):
        asyncio.run(pai.probe())  # never raises
    assert any(r.levelname == level and words in r.getMessage() for r in caplog.records)


def test_tests_never_run_the_real_probe():
    import os

    assert os.environ["DTCC_AGENT_BEDROCK_PROBE"] == "0"


def test_the_http_status_is_found_in_an_implicit_chain():
    """Raised while handling a 403, without `from`: Python sets __context__ only."""
    try:
        try:
            raise ModelHTTPError(status_code=403, model_name="m", body=None)
        except ModelHTTPError:
            raise RuntimeError("while handling")
    except RuntimeError as wrapped:
        assert provenance.http_status(wrapped) == 403
