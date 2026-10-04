"""The measurement layer (ADR-0008, T34): run a fixed question set through the
chat, repeatably, and report latency, tokens and cost per question.

    python -m eval.measure --url ws://localhost:8050/chat --runs 3 --log-dir data/agent/logs

Each run of each question opens a fresh chat, so no history leaks between
questions. The first run of a question is "cold" and later runs "warm": a
cached download changes latency several-fold, so the two are reported apart.
Figures come from the chat's provenance frame (T33); with --log-dir, each
turn's operations and catalogue revision come from the provenance logs.

Measurement only: nothing is scored right or wrong. The question file's
`answer_key` field is reserved for the correctness layer (Anders or Nuri).

Writes eval/runs/<UTC time>-<git sha>.jsonl (one line per question and run)
and a Markdown report beside it, which it also prints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
QUESTIONS = HERE / "questions.json"
RUNS_DIR = HERE / "runs"

_FIELDS = {"id": str, "text": str, "tags": list, "source": str, "needs_dtcc_sim": bool}
_OPTIONAL = {"answer_key"}  # reserved for the correctness layer

SIM_PROBE = "List the available simulations."
# A probe answer like these means dtcc-sim has not registered any simulation.
_NO_SIM = re.compile(r"no simulations|not available|unavailable|(hasn't|has not|haven't) "
                     r"(answered|responded|registered)|not (yet )?running|can(no|')t reach", re.I)

Ask = Callable[[str], Awaitable[dict[str, Any]]]


# -- Questions -----------------------------------------------------------------

def load_questions(path: str | os.PathLike = QUESTIONS) -> list[dict[str, Any]]:
    """The question set, checked: every field present and typed, ids unique."""
    questions = json.loads(Path(path).read_text())["questions"]
    if not questions:
        raise ValueError(f"{path}: no questions")
    seen = set()
    for q in questions:
        for field, kind in _FIELDS.items():
            if not isinstance(q.get(field), kind):
                raise ValueError(f"{path}: {q.get('id', q)!r} needs {field!r} ({kind.__name__})")
        if unknown := set(q) - set(_FIELDS) - _OPTIONAL:
            raise ValueError(f"{path}: {q['id']!r} has unknown fields {sorted(unknown)}")
        if q["id"] in seen:
            raise ValueError(f"{path}: duplicate id {q['id']!r}")
        seen.add(q["id"])
    return questions


# -- One question --------------------------------------------------------------

async def ask(url: str, text: str, *, access_code: str | None, timeout: float = 300) -> dict[str, Any]:
    """Ask `text` in a fresh chat; what came back and how long it took."""
    import websockets

    row: dict[str, Any] = {"status": "ok", "error": None, "latency_s": None, "provenance": None,
                           "images": 0, "files": 0, "answer": ""}
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout):
            async with websockets.connect(url, max_size=None) as ws:
                await ws.send(json.dumps({"session_id": None, "access_code": access_code}))
                first = json.loads(await ws.recv())
                if first.get("type") != "session":
                    return {**row, "status": "error", "error": first.get("code", "no session")}
                started = time.monotonic()  # time the answer, not the handshake
                await ws.send(json.dumps({"content": text}))
                parts = []
                while (frame := json.loads(await ws.recv()))["type"] != "done":
                    kind = frame["type"]
                    if kind == "text":
                        parts.append(frame["content"])
                    elif kind == "image":
                        row["images"] += 1
                    elif kind == "file":
                        row["files"] += 1
                    elif kind == "provenance":
                        row["provenance"] = {k: v for k, v in frame.items() if k != "type"}
                row["answer"] = "".join(parts)
    except TimeoutError:
        return {**row, "status": "timeout", "error": f"no answer in {timeout:.0f} s",
                "latency_s": round(time.monotonic() - started, 2)}
    except Exception as exc:  # a dropped socket or a refused connection: record, move on
        return {**row, "status": "error", "error": type(exc).__name__}
    row["latency_s"] = round(time.monotonic() - started, 2)
    if row["provenance"] is None:
        row.update(status="error", error="no provenance frame")
    elif row["provenance"].get("is_error"):
        row.update(status="error", error=row["provenance"].get("error") or "agent error")
    return row


def sim_available(probe: dict[str, Any]) -> bool:
    """Whether the probe's answer names any simulation. An empty answer names none."""
    answer = probe.get("answer") or ""
    return probe["status"] == "ok" and bool(answer.strip()) and not _NO_SIM.search(answer)


# -- The run -------------------------------------------------------------------

async def measure(questions: list[dict[str, Any]], runs: int, ask: Ask, *,
                  max_cost: float, sim: bool | None = None) -> tuple[list[dict], dict]:
    """Every question `runs` times, runs outermost, so run 1 is every
    question's cold run. Stops cleanly once `max_cost` is spent. `sim` None
    asks the chat whether dtcc-sim is up, if any question needs it."""
    rows: list[dict[str, Any]] = []
    spent = 0.0
    notes: dict[str, Any] = {"stopped_at_cost_cap": False}

    if sim is None and any(q["needs_dtcc_sim"] for q in questions):
        probe = await ask(SIM_PROBE)
        spent += _cost(probe)
        sim = sim_available(probe)
        notes["sim_probe"] = "names simulations" if sim else "names none"
    notes["sim_available"] = sim

    for run in range(1, runs + 1):
        for q in questions:
            base = {"question_id": q["id"], "run": run, "cache": "cold" if run == 1 else "warm"}
            if q["needs_dtcc_sim"] and not sim:
                rows.append({**base, "status": "skipped", "error": "dtcc-sim not available"})
                continue
            if spent >= max_cost:
                notes["stopped_at_cost_cap"] = True
                notes["spent_usd"] = round(spent, 4)
                return rows, notes
            row = await ask(q["text"])
            spent += _cost(row)
            rows.append({**base, **row})
    notes["spent_usd"] = round(spent, 4)
    return rows, notes


def _cost(row: dict[str, Any]) -> float:
    return ((row.get("provenance") or {}).get("total_cost_usd")) or 0.0


def attach_operations(rows: list[dict[str, Any]], log_dir: str | os.PathLike) -> None:
    """Each row's operations and catalogue, from the provenance logs (T33)."""
    from dtcc_agent.provenance import join_records

    joined = {r["turn_id"]: r for r in join_records(log_dir)}
    for row in rows:
        record = joined.get((row.get("provenance") or {}).get("turn_id"))
        if record:
            row["operations"] = record["operations"]
            row["catalogue"] = record["catalogue"]


# -- The report ----------------------------------------------------------------

_INPUT_KEYS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _input_tokens(p: dict[str, Any]) -> int | None:
    """Every input token the model read: fresh, cache read and cache written.
    Most of a turn's input is cached, so input_tokens alone is misleading."""
    usage = p.get("usage")
    return sum(usage[k] for k in _INPUT_KEYS) if usage else None


def _stats(values: list[float]) -> tuple[float, float] | None:
    return (statistics.median(values), max(values)) if values else None


def _fmt(stat: tuple[float, float] | None, digits: int = 1) -> str:
    if stat is None:
        return "—"
    median, top = stat
    return f"{median:,.{digits}f} / {top:,.{digits}f}"


def summarise(rows: list[dict[str, Any]], question_ids: list[str]) -> list[dict[str, Any]]:
    """Per question: medians and maxes over its successful runs only."""
    summary = []
    for qid in question_ids:
        mine = [r for r in rows if r["question_id"] == qid]
        ok = [r for r in mine if r["status"] == "ok"]
        prov = [r["provenance"] for r in ok]
        problems = [r["status"] for r in mine if r["status"] != "ok"]
        summary.append({
            "id": qid, "n_ok": len(ok), "n_runs": len(mine),
            "cold": _stats([r["latency_s"] for r in ok if r["cache"] == "cold"]),
            "warm": _stats([r["latency_s"] for r in ok if r["cache"] == "warm"]),
            "input": _stats([t for p in prov if (t := _input_tokens(p)) is not None]),
            "output": _stats([p["usage"]["output_tokens"] for p in prov if p.get("usage")]),
            "cost": _stats([p["total_cost_usd"] for p in prov if p.get("total_cost_usd") is not None]),
            "tools": _stats([len(p.get("tools_called") or []) for p in prov]),
            "ops": _stats([len(r["operations"]) for r in ok if "operations" in r]),
            "problems": ", ".join(f"{problems.count(s)} {s}" for s in dict.fromkeys(problems)),
        })
    return summary


def _one(values: set[Any]) -> str:
    values.discard(None)
    return ", ".join(sorted(map(str, values))) or "unknown"


def report(rows: list[dict[str, Any]], question_ids: list[str], meta: dict[str, Any]) -> str:
    ok = [r for r in rows if r["status"] == "ok"]
    prov = [r["provenance"] for r in ok]
    catalogues = {json.dumps(r["catalogue"], sort_keys=True) for r in ok if r.get("catalogue")}
    catalogue = json.loads(next(iter(catalogues))) if len(catalogues) == 1 else None
    lines = [
        f"# Measurement run {meta['started']}",
        "",
        f"- Commit: `{meta['git']}` · runs per question: {meta['runs']} · questions: {len(question_ids)}",
        f"- Model: {_one({p.get('model') for p in prov})} "
        f"(all models used: {_one({m for p in prov for m in p.get('models_used') or []})})",
        f"- Prompt version: `{_one({p.get('prompt_version') for p in prov})}` · "
        f"SDK: {_one({p.get('sdk') for p in prov})}",
        f"- Runtime: {_one({p.get('runtime') for p in prov})} · "
        f"provider: {_one({p.get('provider') for p in prov})} · "
        f"cost source: {_one({p.get('cost_source') for p in prov})}",
        f"- Core commit: `{catalogue['core_commit'] if catalogue else 'unknown'}` · "
        f"catalogue: {catalogue['operations'] if catalogue else 'unknown'} operations",
        f"- dtcc-sim: {'available' if meta['notes'].get('sim_available') else 'not available'}"
        + (f" (probe {meta['notes']['sim_probe']})" if "sim_probe" in meta["notes"] else ""),
        f"- Spent: ${meta['notes'].get('spent_usd', 0):.2f} of a ${meta['max_cost']:.2f} cap"
        + (" · **stopped at cost cap**" if meta["notes"].get("stopped_at_cost_cap") else ""),
        "",
        "Each cell is median / max over the question's successful runs; failed runs are left out",
        "and counted under OK and Problems. Run 1 is cold, later runs warm. Input tokens include",
        "cache reads and writes. Cost is computed as the cost source above says.",
        "",
        "| Question | OK | Cold latency (s) | Warm latency (s) | Input tok | Output tok "
        "| Cost ($) | Tools | Ops | Problems |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summarise(rows, question_ids):
        lines.append(
            f"| {s['id']} | {s['n_ok']} / {s['n_runs']} | {_fmt(s['cold'])} | {_fmt(s['warm'])} "
            f"| {_fmt(s['input'], 0)} | {_fmt(s['output'], 0)} | {_fmt(s['cost'], 3)} "
            f"| {_fmt(s['tools'], 0)} | {_fmt(s['ops'], 0)} | {s['problems'] or ''} |"
        )
    total_cost = sum(_cost(r) for r in rows)
    total_time = sum(r["latency_s"] or 0 for r in rows if r.get("latency_s"))
    lines += ["", f"**Totals:** {len(ok)} of {len(rows)} runs ok · ${total_cost:.2f} · "
                  f"{total_time:,.0f} s of answering", ""]
    return "\n".join(lines)


# -- Command line --------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def run(*, url: str, runs: int, questions: list[dict[str, Any]], out_dir: Path,
              log_dir: str | None, max_cost: float, sim: bool | None,
              access_code: str | None, timeout: float) -> tuple[Path, Path, str]:
    """Measure, then write the .jsonl and .md; returns their paths and the report."""
    started = datetime.now(timezone.utc)
    rows, notes = await measure(
        questions, runs, lambda text: ask(url, text, access_code=access_code, timeout=timeout),
        max_cost=max_cost, sim=sim)
    if log_dir:
        attach_operations(rows, log_dir)
    meta = {"started": started.strftime("%Y-%m-%d %H:%M UTC"), "git": _git_sha(), "runs": runs,
            "max_cost": max_cost, "notes": notes}
    stem = f"{started:%Y%m%dT%H%M%SZ}-{meta['git']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data, md = out_dir / f"{stem}.jsonl", out_dir / f"{stem}.md"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    text = report(rows, [q["id"] for q in questions], meta)
    md.write_text(text)
    return data, md, text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.measure", description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="ws://localhost:8050/chat")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--questions", default=str(QUESTIONS))
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--log-dir", help="the provenance log dir, for operations and catalogue")
    parser.add_argument("--max-cost", type=float, default=2.00, help="stop once this many USD are spent")
    parser.add_argument("--timeout", type=float, default=300, help="seconds per question")
    parser.add_argument("--assume-sim", action="store_true", help="skip the dtcc-sim probe")
    parser.add_argument("--out-dir", default=str(RUNS_DIR))
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)
    if args.only:
        wanted = args.only.split(",")
        unknown = set(wanted) - {q["id"] for q in questions}
        if unknown:
            parser.error(f"unknown question ids: {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q["id"] in wanted]
    data, md, text = asyncio.run(run(
        url=args.url, runs=args.runs, questions=questions, out_dir=Path(args.out_dir),
        log_dir=args.log_dir, max_cost=args.max_cost, sim=True if args.assume_sim else None,
        access_code=os.getenv("DTCC_AGENT_ACCESS_CODE"), timeout=args.timeout))
    print(text)
    print(f"Wrote {data} and {md}", file=sys.stderr)


if __name__ == "__main__":
    main()
