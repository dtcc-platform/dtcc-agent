"""What a served process sets up once, before its first tool call (#12).

`start()` runs from the HTTP app's lifespan before it accepts requests.
FastMCP's own lifespan runs per session (or per request when stateless), so
nothing here belongs in it. stdio does not call it: the chatbot starts a stdio
server per message, and most never read the catalogue, so they build it on
first use (#22).
"""

from __future__ import annotations

import os
import sys

import anyio

from . import registry

# Tool bodies running at once, across every Session (#19). Each operation may
# deepcopy a heavy input, so anyio's default of 40 threads would trade a
# capacity limit for an OOM kill. Creating it is instant and cannot fail, so it
# is made at import rather than in start().
_workers_env = os.getenv("DTCC_MCP_WORKERS", "4")
if not (_workers_env.isascii() and _workers_env.isdigit() and int(_workers_env) >= 1):
    raise ValueError(f"DTCC_MCP_WORKERS must be a whole number >= 1, got {_workers_env!r}")
WORKERS = int(_workers_env)
workers = anyio.CapacityLimiter(WORKERS)

# At most this many of them from one Session, so one busy Session never
# leaves the others without a worker.
SESSION_WORKERS = max(1, WORKERS // 2)


def start() -> None:
    """Build the catalogue (over a second, cold), so no request pays for it.

    Raises registry.CatalogueError on a broken Core install: the process then
    fails to start instead of serving a partial catalogue.
    """
    catalogue = registry.build()  # and starts asking dtcc-sim, in the background
    print(f"dtcc-agent: catalogue built: {len(catalogue)} operations",
          file=sys.stderr, flush=True)
