"""A record of builder calls, kept while builder results are not cached.

U2 (#11), decided 2026-09-30: builder results are not cached until their cache
keys are correct. The old key described an input object by metadata alone, so
different inputs could collide, and it dropped `bounds`, so different areas
did. Whether correct (provenance) keys are worth building depends on how often
the same builder call repeats and how long it takes, so each call is recorded
here: the operation, its duration, and the key a cache would have matched.

The key keeps `bounds` and describes an input Object by its type, source
operation, size and label, as the old cache did. Equal keys are therefore an
upper bound on the hits a correct cache would get. Only a hash is written,
never the parameters. Recording needs DTCC_AGENT_LOG_DIR (Docker sets
/data/logs); without it nothing is written.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .object_store import ObjectStore

logger = logging.getLogger(__name__)


def would_be_key(operation: str, params: dict[str, Any], store: ObjectStore) -> str:
    """The key a cache shared across Sessions would have looked this call up by."""
    listed = {e["object_ref"]: e for e in store.list(limit=len(store))}
    described = {}
    for name, value in params.items():
        entry = listed.get(value) if isinstance(value, str) else None
        described[name] = ({k: entry[k] for k in ("type", "source_op", "nbytes", "label")}
                           if entry else value)
    payload = json.dumps({"op": operation, "params": described}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def record(operation: str, params: dict[str, Any], store: ObjectStore,
           seconds: float, ok: bool) -> None:
    """Append one line for a builder call, if DTCC_AGENT_LOG_DIR is set."""
    log_dir = os.getenv("DTCC_AGENT_LOG_DIR")
    if not log_dir:
        return
    try:
        line = {"at": round(time.time(), 3), "operation": operation, "ok": ok,
                "seconds": round(seconds, 3), "key": would_be_key(operation, params, store)}
        path = Path(log_dir) / "builder_calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:  # one short line per append
            f.write(json.dumps(line) + "\n")
    except Exception:  # a record must never fail the call it describes
        logger.warning("Could not record builder call %s", operation, exc_info=True)
