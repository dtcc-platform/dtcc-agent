"""The measurement harness (T34, #74): questions, the run, the report."""

import asyncio
import json
import socket
import threading
import time

import pytest

from eval import measure


def _prov(cost=0.1, out=20, turn="turn_00000001"):
    return {"turn_id": turn, "model": "claude-sonnet-4-5", "models_used": ["claude-sonnet-4-5"],
            "prompt_version": "5fdfaaf2e87b", "sdk": "claude-agent-sdk 0.2.163",
            "runtime": "sdk", "provider": "bedrock", "cost_source": "sdk total_cost_usd",
            "tools_called": ["a", "b"], "total_cost_usd": cost, "is_error": False,
            "usage": {"input_tokens": 10, "output_tokens": out, "cache_read_input_tokens": 100,
                      "cache_creation_input_tokens": 0}}


def _row(qid, run, latency, status="ok", **prov):
    return {"question_id": qid, "run": run, "cache": "cold" if run == 1 else "warm",
            "status": status, "latency_s": latency, "error": None if status == "ok" else status,
            "provenance": _prov(**prov) if status == "ok" else None}


# -- Questions -----------------------------------------------------------------

def test_the_committed_question_set_is_valid_and_covers_the_plan():
    questions = measure.load_questions()
    assert 8 <= len(questions) <= 12
    tags = {t for q in questions for t in q["tags"]}
    assert {"geocode", "cache-subarea", "terrain-raster", "render", "export", "discovery",
            "builder-chain", "expected-refusal", "expected-not-stored", "simulation"} <= tags
    assert any(q["needs_dtcc_sim"] for q in questions)
    assert not any("answer_key" in q for q in questions)  # reserved, not scored


@pytest.mark.parametrize("bad, message", [
    ([{"id": "a", "text": "t", "tags": [], "source": "s"}], "needs_dtcc_sim"),
    ([{"id": "a", "text": "t", "tags": [], "source": "s", "needs_dtcc_sim": False}] * 2, "duplicate"),
    ([{"id": "a", "text": "t", "tags": [], "source": "s", "needs_dtcc_sim": False, "score": 1}], "unknown"),
    ([], "no questions"),
])
def test_a_malformed_question_set_is_refused(tmp_path, bad, message):
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"questions": bad}))
    with pytest.raises(ValueError, match=message):
        measure.load_questions(path)


# -- The run -------------------------------------------------------------------

QS = [{"id": "q1", "text": "one", "needs_dtcc_sim": False},
      {"id": "q2", "text": "two", "needs_dtcc_sim": False},
      {"id": "sim", "text": "heat", "needs_dtcc_sim": True}]


def _asker(cost=0.1, answers=None):
    asked = []

    async def ask(text):
        asked.append(text)
        answer = (answers or {}).get(text, "fine")
        return {"status": "ok", "latency_s": 1.0, "error": None, "provenance": _prov(cost=cost),
                "images": 0, "files": 0, "answer": answer}

    return ask, asked


def test_runs_are_outermost_so_run_one_is_every_questions_cold_run():
    ask, asked = _asker()
    rows, notes = asyncio.run(measure.measure(QS[:2], 2, ask, max_cost=10, sim=False))
    assert asked == ["one", "two", "one", "two"]
    assert [(r["question_id"], r["cache"]) for r in rows] == [
        ("q1", "cold"), ("q2", "cold"), ("q1", "warm"), ("q2", "warm")]


def test_simulation_questions_are_skipped_when_the_probe_names_none():
    ask, asked = _asker(answers={measure.SIM_PROBE: "dtcc-sim hasn't answered yet, so no simulations."})
    rows, notes = asyncio.run(measure.measure(QS, 1, ask, max_cost=10))
    assert asked[0] == measure.SIM_PROBE and "heat" not in asked
    [skipped] = [r for r in rows if r["question_id"] == "sim"]
    assert skipped["status"] == "skipped" and skipped["error"] == "dtcc-sim not available"
    assert notes["sim_available"] is False


@pytest.mark.parametrize("answer", ["", "   "])
def test_an_empty_probe_answer_names_no_simulation(answer):
    assert measure.sim_available({"status": "ok", "answer": answer}) is False


def test_simulation_questions_run_when_the_probe_names_some():
    ask, asked = _asker(answers={measure.SIM_PROBE: "Available: urban_heat_simulation."})
    asyncio.run(measure.measure(QS, 1, ask, max_cost=10))
    assert "heat" in asked


def test_the_cost_cap_stops_the_run_cleanly():
    ask, asked = _asker(cost=0.75)
    rows, notes = asyncio.run(measure.measure(QS[:2], 3, ask, max_cost=2.0, sim=False))
    assert len(asked) == 3  # 0.75 + 0.75 + 0.75 crosses 2.00: no fourth question starts
    assert notes["stopped_at_cost_cap"] is True and notes["spent_usd"] == 2.25
    meta = {"started": "now", "git": "abc", "runs": 3, "max_cost": 2.0, "notes": notes}
    assert "stopped at cost cap" in measure.report(rows, ["q1", "q2"], meta)


# -- The report ----------------------------------------------------------------

def test_the_report_gives_median_and_max_over_successful_runs_only():
    rows = [_row("q1", 1, 30.0, cost=0.3), _row("q1", 2, 10.0, cost=0.1, out=10),
            _row("q1", 3, 14.0, cost=0.2, out=30), _row("q1", 4, None, status="timeout"),
            _row("q2", 1, None, status="error")]
    [q1, q2] = measure.summarise(rows, ["q1", "q2"])
    assert (q1["n_ok"], q1["n_runs"]) == (3, 4)
    assert q1["cold"] == (30.0, 30.0) and q1["warm"] == (12.0, 14.0)
    assert q1["cost"] == (0.2, 0.3) and q1["output"] == (20, 30)
    assert q1["input"] == (110, 110)  # fresh plus cache reads
    assert q1["problems"] == "1 timeout"
    assert (q2["n_ok"], q2["n_runs"], q2["cold"], q2["problems"]) == (0, 1, None, "1 error")

    meta = {"started": "now", "git": "abc", "runs": 3, "max_cost": 5.0,
            "notes": {"sim_available": False, "spent_usd": 0.6}}
    text = measure.report(rows, ["q1", "q2"], meta)
    assert "| q1 | 3 / 4 | 30.0 / 30.0 | 12.0 / 14.0 |" in text
    assert "| q2 | 0 / 1 | — | — |" in text
    assert "Prompt version: `5fdfaaf2e87b`" in text and "claude-sonnet-4-5" in text
    assert "Core commit: `unknown`" in text  # no --log-dir
    assert "Runtime: sdk · provider: bedrock · cost source: sdk total_cost_usd" in text
    assert "Catalogue in prompt: " in text and " · variant: " in text


def test_operations_and_the_catalogue_come_from_the_provenance_logs(tmp_path):
    (tmp_path / "answers.jsonl").write_text(json.dumps(
        {"at": "2026-10-03T10:00:05.000Z", "turn_id": "turn_00000001"}) + "\n")
    (tmp_path / "operations.jsonl").write_text("".join(json.dumps(l) + "\n" for l in [
        {"type": "catalogue", "at": "2026-10-03T10:00:00.000Z", "core_commit": "bb95f2f", "operations": 133},
        {"at": "2026-10-03T10:00:01.000Z", "turn_id": "turn_00000001", "tool": "geocode"},
        {"at": "2026-10-03T10:00:02.000Z", "turn_id": "turn_00000001", "tool": "run_operation"},
    ]))
    rows = [_row("q1", 1, 5.0)]
    measure.attach_operations(rows, tmp_path)
    assert [op["tool"] for op in rows[0]["operations"]] == ["geocode", "run_operation"]
    meta = {"started": "now", "git": "abc", "runs": 1, "max_cost": 5.0, "notes": {}}
    text = measure.report(rows, ["q1"], meta)
    assert "Core commit: `bb95f2f` · catalogue: 133 operations" in text
    assert "| 2 / 2 | " in text  # Ops column


# -- Against the real chat app, with a faked agent -----------------------------

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def chat_url(monkeypatch, tmp_path):
    """The chatbot app served over a real socket; its agent answers instantly."""
    # The app refuses to start without a Bedrock credential (#85); the agent is faked.
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test")
    import uvicorn
    from types import SimpleNamespace

    import chatbot.app as app_module
    from tests.test_chatbot_app import _StubMemory, _fake_runtime

    async def answer(ws, session, text, turn):
        turn.saw_model("claude-sonnet-4-5")
        turn.saw_tool("mcp__dtcc-agent__geocode")
        turn.saw_result(SimpleNamespace(num_turns=1, duration_ms=5, duration_api_ms=4,
                                        total_cost_usd=0.01, is_error=False, model_usage={},
                                        usage={"input_tokens": 1, "output_tokens": 2,
                                               "cache_read_input_tokens": 0,
                                               "cache_creation_input_tokens": 0}))
        await ws.send_json({"type": "text", "content": "There are 188 buildings."})
        return "There are 188 buildings."

    monkeypatch.setattr(app_module, "ACCESS_CODE", "a-sixteen-char-code")
    monkeypatch.setattr(app_module, "_log_dir", tmp_path / "logs")
    monkeypatch.setattr(app_module, "memory", _StubMemory())
    monkeypatch.setattr(app_module, "runtime", _fake_runtime(answer))

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app_module.app, host="127.0.0.1", port=port,
                                           log_level="warning", ws="websockets"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "chat app never started"
        time.sleep(0.05)
    yield f"ws://127.0.0.1:{port}/chat", tmp_path / "logs"
    server.should_exit = True
    thread.join(timeout=10)


def test_a_run_against_the_chat_app_writes_both_files(chat_url, tmp_path):
    url, log_dir = chat_url
    questions = [{"id": "q1", "text": "How many?", "tags": [], "source": "t", "needs_dtcc_sim": False},
                 {"id": "q2", "text": "Again?", "tags": [], "source": "t", "needs_dtcc_sim": False}]
    data, md, text = asyncio.run(measure.run(
        url=url, runs=2, questions=questions, out_dir=tmp_path / "runs", log_dir=str(log_dir),
        max_cost=5, sim=None, access_code="a-sixteen-char-code", timeout=30))

    rows = [json.loads(l) for l in data.read_text().splitlines()]
    assert [(r["question_id"], r["cache"], r["status"]) for r in rows] == [
        ("q1", "cold", "ok"), ("q2", "cold", "ok"), ("q1", "warm", "ok"), ("q2", "warm", "ok")]
    assert len({r["provenance"]["turn_id"] for r in rows}) == 4  # a fresh chat each time
    assert rows[0]["answer"] == "There are 188 buildings."
    assert md.read_text() == text and "| q1 | 2 / 2 |" in text


def test_a_refused_access_code_is_an_error_row_not_a_crash(chat_url, tmp_path):
    url, _ = chat_url
    row = asyncio.run(measure.ask(url, "How many?", access_code="wrong-wrong-wrong-wrong", timeout=10))
    assert row["status"] == "error" and row["error"] == "admission_required"


# -- Pooling runs (#85) --------------------------------------------------------

def test_pooled_runs_report_every_sample_together(tmp_path):
    first = [_row("q1", 1, 30.0), _row("q1", 2, 10.0), _row("q1", 3, 12.0)]
    second = [_row("q1", 1, 26.0), _row("q1", 2, 20.0), _row("q1", 3, 14.0)]
    paths = []
    for name, rows in (("20261004T220414Z-713b47a", first), ("20261004T222251Z-0e1c2fd", second)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        paths.append(path)

    md, text = measure.pool(paths, tmp_path / "out")

    [q1] = measure.summarise(json_rows := [r for p in paths for r in map(json.loads, p.read_text().splitlines())], ["q1"])
    assert (q1["n_ok"], q1["n_runs"]) == (6, 6) and len(json_rows) == 6
    assert q1["cold"] == (28.0, 30.0) and q1["warm"] == (13.0, 20.0)  # four warm samples
    assert "| q1 | 6 / 6 | 28.0 / 30.0 | 13.0 / 20.0 |" in text
    assert "Pooled from 2 runs" in text and "713b47a" in text and "0e1c2fd" in text
    assert md.read_text() == text


def test_pool_runs_from_the_command_line(tmp_path, capsys):
    path = tmp_path / "20261004T220414Z-713b47a.jsonl"
    path.write_text(json.dumps(_row("q01-building-count", 1, 5.0)) + "\n")
    measure.main(["--pool", str(path), str(path), "--out-dir", str(tmp_path / "out")])
    assert "Pooled from 2 runs" in capsys.readouterr().out


def test_an_unpriced_turn_stops_the_run(monkeypatch):
    # The cost cap adds turn costs; an unpriced turn would add nothing (#86).
    def ask(text):
        async def answer():
            return {"status": "ok", "error": None, "latency_s": 1.0,
                    "provenance": {**_prov(), "total_cost_usd": None, "cost_source": "unpriced"}}
        return answer()

    with pytest.raises(SystemExit, match="unpriced"):
        asyncio.run(measure.measure(QS[:1], 1, ask, max_cost=5.0, sim=False))
