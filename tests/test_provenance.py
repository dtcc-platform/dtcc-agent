"""Provenance (T33, #73): what each service records first-hand, and the join."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dtcc_agent import provenance, registry
from chatbot import provenance as answers

SECRET = "s3cret-value-do-not-log"


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _record(**overrides):
    call = dict(tool="run_operation", arguments={"name": "datasets.buildings", "params": {}},
                result=json.dumps({"object_ref": "obj_ae95c27e", "cache_hit": True}),
                error=None, seconds=1.23456, session_id="s1", subject="anonymous",
                turn_id="turn_1a2b3c4d")
    call.update(overrides)
    provenance.record_operation(**call)


# -- Server: operations.jsonl -------------------------------------------------

def test_an_operation_line_names_its_turn_and_hashes_its_parameters(tmp_path, monkeypatch):
    monkeypatch.setenv("DTCC_AGENT_LOG_DIR", str(tmp_path))
    _record(arguments={"name": "datasets.buildings", "params": {"token": SECRET}})

    [line] = _lines(tmp_path / "operations.jsonl")
    assert line["turn_id"] == "turn_1a2b3c4d" and line["session_id"] == "s1"
    assert line["subject"] == "anonymous" and line["tool"] == "run_operation"
    assert line["operation"] == "datasets.buildings"
    assert line["params_hash"] == provenance.params_hash(
        {"name": "datasets.buildings", "params": {"token": SECRET}})
    assert len(line["params_hash"]) == 16
    assert line["seconds"] == 1.235 and line["ok"] is True and line["error"] is None
    assert line["cache_hit"] is True and line["object_refs"] == ["obj_ae95c27e"]
    assert line["at"].endswith("Z")
    assert set(line) == {"at", "turn_id", "session_id", "subject", "tool", "operation",
                         "params_hash", "seconds", "ok", "error", "cache_hit",
                         "object_refs", "catalogue"}


@pytest.mark.parametrize("result, error, category", [
    (json.dumps({"error": f"Invalid bounds: {SECRET}"}), None, "Invalid bounds"),
    (json.dumps({"error": f"no category here {SECRET}"}), None, "error"),
    (json.dumps({"error": f"{SECRET}: then text"}), None, "error"),
    (None, ValueError(f"Refused: {SECRET}"), "ValueError"),
], ids=["category", "no-colon", "value-before-colon", "raised"])
def test_a_failed_call_records_its_category_never_its_text(tmp_path, monkeypatch, result, error, category):
    monkeypatch.setenv("DTCC_AGENT_LOG_DIR", str(tmp_path))
    _record(tool="load_geojson", arguments={"name": SECRET}, result=result, error=error)

    [line] = _lines(tmp_path / "operations.jsonl")
    assert line["ok"] is False and line["error"] == category
    assert line["operation"] is None and line["object_refs"] == []
    assert SECRET not in (tmp_path / "operations.jsonl").read_text()


def test_object_refs_keep_a_result_too_large_to_store(tmp_path, monkeypatch):
    monkeypatch.setenv("DTCC_AGENT_LOG_DIR", str(tmp_path))
    _record(result=json.dumps({"object_refs": ["obj_00000001", None]}))
    _record(result="not json")
    first, second = _lines(tmp_path / "operations.jsonl")
    assert first["object_refs"] == ["obj_00000001", None] and first["cache_hit"] is False
    assert second["ok"] is True and second["object_refs"] == []


def test_without_a_log_dir_the_server_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("DTCC_AGENT_LOG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    _record()
    provenance.record_catalogue(133)
    assert list(tmp_path.iterdir()) == []


def test_a_log_that_cannot_be_written_never_fails_the_call(tmp_path, monkeypatch, caplog):
    blocked = tmp_path / "file-not-dir"
    blocked.write_text("")
    monkeypatch.setenv("DTCC_AGENT_LOG_DIR", str(blocked))
    with caplog.at_level("WARNING"):
        _record()
    assert any("operations.jsonl" in r.getMessage() for r in caplog.records)


def test_the_catalogue_is_recorded_once_built_and_named_on_later_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("DTCC_AGENT_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(registry, "_REGISTRY", None)
    _record()  # nothing built yet: the call did not need the catalogue
    monkeypatch.setattr(registry, "_REGISTRY", {f"op{i}": None for i in range(133)})
    provenance.record_catalogue(133)
    _record()

    before, catalogue, after = _lines(tmp_path / "operations.jsonl")
    assert before["catalogue"] is None
    assert catalogue["type"] == "catalogue" and catalogue["operations"] == 133
    assert catalogue["core_commit"] == after["catalogue"]["core_commit"]
    assert after["catalogue"]["operations"] == 133


# -- Join ----------------------------------------------------------------------

def _write(path, lines):
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


@pytest.fixture
def two_turns(tmp_path):
    cat = {"type": "catalogue", "at": "2026-10-02T09:00:00.000Z", "core_commit": "bb95f2f", "operations": 133}
    ops = [{"at": f"2026-10-02T09:00:0{i}.000Z", "turn_id": "turn_aaaaaaaa", "tool": t}
           for i, t in [(3, "render_object"), (1, "geocode"), (2, "run_operation")]]
    _write(tmp_path / "operations.jsonl", [cat, *ops,
                                           {"at": "2026-10-02T09:00:04.000Z", "turn_id": None, "tool": "x"}])
    _write(tmp_path / "answers.jsonl", [
        {"at": "2026-10-02T09:01:00.000Z", "turn_id": "turn_bbbbbbbb", "model": "m"},
        {"at": "2026-10-02T09:00:05.000Z", "turn_id": "turn_aaaaaaaa", "model": "m"},
    ])
    return tmp_path


def test_join_attaches_each_turns_operations_in_order(two_turns):
    first, second = provenance.join_records(two_turns)
    assert first["turn_id"] == "turn_aaaaaaaa"
    assert [op["tool"] for op in first["operations"]] == ["geocode", "run_operation", "render_object"]
    assert second["turn_id"] == "turn_bbbbbbbb" and second["operations"] == []
    # A turn with no operations still names its catalogue revision.
    assert first["catalogue"] == second["catalogue"] == {"core_commit": "bb95f2f", "operations": 133}


def test_join_filters_to_one_turn_and_runs_from_the_command_line(two_turns):
    assert [r["turn_id"] for r in provenance.join_records(two_turns, turn="turn_bbbbbbbb")] == ["turn_bbbbbbbb"]
    out = subprocess.run([sys.executable, "-m", "dtcc_agent.provenance", "join", str(two_turns),
                          "--turn", "turn_aaaaaaaa"], capture_output=True, text=True, check=True).stdout
    [record] = [json.loads(line) for line in out.splitlines()]
    assert len(record["operations"]) == 3


def test_join_of_an_empty_log_dir_is_empty(tmp_path):
    assert provenance.join_records(tmp_path) == []


# -- Chatbot: answers.jsonl ----------------------------------------------------

def _result(**overrides):
    fields = dict(num_turns=3, duration_ms=9000, duration_api_ms=7000, total_cost_usd=0.04,
                  is_error=False, model_usage={"claude-sonnet-4-5-20250929": {}},
                  usage={"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 30,
                         "cache_creation_input_tokens": 40, "service_tier": "standard"})
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _turn():
    return answers.TurnRecord("turn_1a2b3c4d", "s1", "anonymous", memory_context=True)


ANSWER_FIELDS = {"at", "turn_id", "session_id", "subject", "model", "models_used",
                 "prompt_version", "memory_context", "sdk", "tools_called", "num_turns",
                 "duration_ms", "duration_api_ms", "usage", "total_cost_usd", "is_error",
                 "retried_fresh", "error", "runtime", "provider", "cost_source"}


def test_an_answer_record_holds_what_the_agent_reported():
    turn = _turn()
    turn.saw_model("claude-sonnet-4-5-20250929")
    turn.saw_tool("mcp__dtcc-agent__run_operation")
    turn.saw_result(_result())
    record = turn.record()

    assert set(record) == ANSWER_FIELDS
    assert record["model"] == "claude-sonnet-4-5-20250929"
    assert record["models_used"] == ["claude-sonnet-4-5-20250929"]
    assert record["tools_called"] == ["mcp__dtcc-agent__run_operation"]
    assert record["usage"] == {"input_tokens": 10, "output_tokens": 20,
                               "cache_read_input_tokens": 30, "cache_creation_input_tokens": 40}
    assert record["num_turns"] == 3 and record["total_cost_usd"] == 0.04
    assert record["is_error"] is False and record["retried_fresh"] is False and record["error"] is None
    assert record["memory_context"] is True and record["sdk"].startswith("claude-agent-sdk ")
    assert record["prompt_version"] == answers.prompt_version()
    assert (record["runtime"], record["provider"]) == ("sdk", "bedrock")
    assert record["cost_source"] == "sdk total_cost_usd"


def test_a_turn_retried_fresh_sums_both_attempts():
    turn = _turn()
    turn.saw_result(_result())
    turn.retry()
    turn.saw_result(_result(total_cost_usd=0.01, duration_ms=1000))
    record = turn.record()
    assert record["retried_fresh"] is True
    assert record["total_cost_usd"] == pytest.approx(0.05)
    assert record["duration_ms"] == 10_000 and record["num_turns"] == 6
    assert record["usage"]["input_tokens"] == 20


def test_a_turn_that_crashed_before_any_result_still_has_every_field():
    turn = _turn()
    turn.failed(RuntimeError(f"boom {SECRET}"))
    record = turn.record()
    assert set(record) == ANSWER_FIELDS
    assert record["is_error"] is True and record["error"] == "RuntimeError"
    assert record["model"] is None and record["usage"] is None and record["total_cost_usd"] is None
    assert SECRET not in json.dumps(record)


def test_editing_the_system_prompt_changes_its_version():
    from chatbot.config import SYSTEM_PROMPT

    assert len(answers.prompt_version()) == 12
    assert answers.prompt_version(SYSTEM_PROMPT + " ") != answers.prompt_version(SYSTEM_PROMPT)
