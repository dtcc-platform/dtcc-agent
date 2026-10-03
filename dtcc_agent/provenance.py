"""Provenance (T33, #73): what produced each answer, recorded first-hand.

Each service writes only what it knows itself, and the two records are joined
on the turn id the chatbot mints per user message (refs.TURN):

- The MCP server appends one line per tool call to operations.jsonl: the
  turn, the Session, the tool and operation, a hash of the arguments, how long
  it took, whether it worked, cache hit, the Object references it returned
  and the catalogue it ran against. It also writes one "catalogue" line when
  its catalogue is built.
- The chatbot appends one line per turn to answers.jsonl: model, prompt
  version, tokens, cost and latency (chatbot/provenance.py).

Neither file holds parameter values, error texts or file contents: arguments
are hashed, and an error is reduced to its category. A record never fails the
call it describes. The server writes only with DTCC_AGENT_LOG_DIR set.

`python -m dtcc_agent.provenance join <log_dir> [--turn <id>]` prints one
complete record per turn.

This module imports nothing heavy at import time, so the chatbot can share it
without loading dtcc-core.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OPERATIONS = "operations.jsonl"
ANSWERS = "answers.jsonl"

# An error's category is the label before its first ":", such as "Refused" or
# "Invalid bounds". Anything else could be a value, so it is not written.
_CATEGORY = re.compile(r"[A-Za-z][A-Za-z ]{0,39}")


def now() -> str:
    """UTC with milliseconds, as both files write it: sortable as text."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def append(log_dir: str | os.PathLike, name: str, line: dict[str, Any]) -> bool:
    """Append one JSON line to `log_dir/name`. Never raises: a record must
    never fail what it describes. False when it could not be written."""
    try:
        path = Path(log_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:  # one short line per append
            f.write(json.dumps(line, default=str) + "\n")
        return True
    except Exception:
        logger.warning("Could not append to %s in %s", name, log_dir, exc_info=True)
        return False


def params_hash(arguments: dict[str, Any]) -> str:
    """The first 16 hex characters of the sha256 of the canonical arguments."""
    payload = json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def error_category(message: str) -> str:
    """The label before the first ":" when it is one, else "error"."""
    head = message.split(":", 1)[0].strip() if ":" in message else ""
    return head if _CATEGORY.fullmatch(head) else "error"


@functools.cache
def _core_commit() -> str:
    from .disk_cache import _core_build

    return _core_build()


def catalogue_info() -> dict[str, Any] | None:
    """The catalogue this process serves, or None if it has not been built:
    stdio builds it only when a tool first needs it (#22)."""
    from . import registry

    built = registry._REGISTRY
    return {"core_commit": _core_commit(), "operations": len(built)} if built is not None else None


def record_catalogue(operations: int) -> None:
    """One line saying which catalogue this process now serves."""
    if log_dir := os.getenv("DTCC_AGENT_LOG_DIR"):
        append(log_dir, OPERATIONS, {"type": "catalogue", "at": now(),
                                     "core_commit": _core_commit(), "operations": operations})


def record_operation(*, tool: str, arguments: dict[str, Any], result: str | None,
                     error: BaseException | None, seconds: float, session_id: str,
                     subject: str, turn_id: str | None) -> None:
    """One line for one tool call, after it returned (`result`) or raised
    (`error`). Writes nothing without DTCC_AGENT_LOG_DIR."""
    log_dir = os.getenv("DTCC_AGENT_LOG_DIR")
    if not log_dir:
        return
    try:
        try:
            body = json.loads(result) if result is not None else None
        except (TypeError, json.JSONDecodeError):
            body = None
        body = body if isinstance(body, dict) else {}
        if error is not None:
            failure = type(error).__name__
        elif "error" in body:
            failure = error_category(str(body["error"]))
        else:
            failure = None
        if "object_ref" in body:
            object_refs = [body["object_ref"]]
        elif isinstance(body.get("object_refs"), list):
            object_refs = body["object_refs"]
        else:
            object_refs = []
        line = {
            "at": now(), "turn_id": turn_id, "session_id": session_id, "subject": subject,
            "tool": tool,
            "operation": arguments.get("name") if tool == "run_operation" else None,
            "params_hash": params_hash(arguments), "seconds": round(seconds, 3),
            "ok": failure is None, "error": failure,
            "cache_hit": body.get("cache_hit") is True, "object_refs": object_refs,
            "catalogue": catalogue_info(),
        }
    except Exception:
        logger.warning("Could not describe the %s call for %s", tool, OPERATIONS, exc_info=True)
        return
    append(log_dir, OPERATIONS, line)


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def join_records(log_dir: str | os.PathLike, turn: str | None = None) -> list[dict[str, Any]]:
    """Each answer with its operations, oldest first, and the catalogue
    revision last recorded before it: a turn that ran no operation still
    names one."""
    log_dir = Path(log_dir)
    operations: dict[str, list[dict[str, Any]]] = {}
    catalogues = []
    for line in _read(log_dir / OPERATIONS):
        if line.get("type") == "catalogue":
            catalogues.append(line)
        elif line.get("turn_id"):
            operations.setdefault(line["turn_id"], []).append(line)
    catalogues.sort(key=lambda c: c["at"])

    joined = []
    for answer in sorted(_read(log_dir / ANSWERS), key=lambda a: a["at"]):
        if turn and answer["turn_id"] != turn:
            continue
        before = [c for c in catalogues if c["at"] <= answer["at"]]
        joined.append({
            **answer,
            "catalogue": ({k: before[-1][k] for k in ("core_commit", "operations")}
                          if before else None),
            "operations": sorted(operations.get(answer["turn_id"], []), key=lambda o: o["at"]),
        })
    return joined


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m dtcc_agent.provenance")
    commands = parser.add_subparsers(dest="command", required=True)
    join = commands.add_parser("join", help="print one complete record per turn")
    join.add_argument("log_dir")
    join.add_argument("--turn", help="only this turn id")
    args = parser.parse_args(argv)
    for record in join_records(args.log_dir, turn=args.turn):
        print(json.dumps(record), file=sys.stdout)


if __name__ == "__main__":
    main()
