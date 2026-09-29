---
type: architecture
title: Sessions and isolation
description: How dtcc-agent makes the Session the isolation unit (ADR-0004), covering the X-DTCC-Session header, per-Session object stores and runs, the eight-Session cap with LRU eviction, the local stdio Session, and the gaps that remain.
tags: [session, isolation, security, adr-0004, http]
sources:
  - id: openwiki-source-778a883bcdc0a6ed0b3401f7
    resource: repo://chatbot/config.py
  - id: openwiki-source-931ea4e3e14cfe3c996abf4a
    resource: repo://chatbot/memory.py
  - id: openwiki-source-2d18eac28ce6bc775094dd62
    resource: repo://docs/adr/0004-session-is-the-isolation-unit.md
  - id: openwiki-source-052f7c9f16ee5a8169a3fb7d
    resource: repo://dtcc_agent/disk_cache.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
  - id: openwiki-source-7da8cb11cdc15fb1e5a1f088
    resource: repo://tests/test_http_sessions.py
generated: { by: "claude-code", at: "2026-09-29T13:24:05.367Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-29T13:24:05.367Z
---

# Sessions and isolation

**The Session is the isolation unit** (ADR-0004, accepted 2026-09-19). Objects, Runs, conversation memory and budgets each belong to exactly one Session and are never visible from another. The ADR rejected the alternative, a shared store with session-scoped views, because that makes isolation a property of every query and so something a later feature can quietly break.

A Session is one continuous conversation, identified anonymously and scoped to a browser. Session lifetime is not person lifetime. When authentication arrives, a *subject* will own many Sessions; the subject identifier is a separate field and does not replace the Session.

## Server side (`dtcc_agent/server.py`)

`_Session` is a dataclass holding everything one Session owns:

- `objects`: its own `ObjectStore`
- `results`: its simulation runs, keyed by run id
- `in_flight`: the number of tool calls currently running
- `workers`: its share of the process worker pool (see [MCP server and tool execution](mcp-server-and-tool-execution.md))

### Resolving the caller

`_request_session()` reads the MCP `request_ctx`:

- **HTTP request present.** The `X-DTCC-Session` header is required. A missing or empty header raises `ToolError("Refused: ...")`. Otherwise `_session_for(id, acquire=True)` returns the live Session and marks a call in flight in the same locked step.
- **No request, while serving HTTP.** The call is refused rather than being handed the local Session, which would otherwise be shared between clients.
- **No request under stdio, or an in-process caller.** Returns `None`, and `_session()` falls back to the single process-wide `_local_session`.

Sessions are keyed by the header value rather than by MCP transport session, because the chatbot opens a new connection per message. Objects therefore survive into the next connection of the same Session.

### Cap and eviction

- `MAX_SESSIONS = 8` Sessions are kept in an `OrderedDict` in least-recently-used order.
- Each HTTP Session's store gets `OBJECT_BUDGET_BYTES // MAX_SESSIONS` (2 GiB / 8 = 256 MiB), so the Sessions together never exceed the old single-store budget. The local stdio Session gets the whole 2 GiB and the whole worker pool.
- `_evict_idle_excess()` drops the least recently used **idle** Sessions beyond the cap, with their objects and runs. A Session with a call in flight is never evicted. If every Session is busy the cap is exceeded temporarily, and the excess is trimmed as calls finish (`_release`).

## Chatbot side

- `chatbot/sessions.py` `SessionManager` mints a 12-hex-char id per browser session, keeps it in memory for up to an hour (cleaned on `create()`), and stores the Agent SDK session id used for resume.
- `chatbot/config.py` `get_mcp_server_config(session_id)`: with `DTCC_MCP_URL` set, it connects over HTTP and sends `X-DTCC-Session: <session_id>`. Otherwise it spawns the server over stdio, which gives one Session per child process.
- `chatbot/memory.py` `ConversationMemory.retrieve()` filters the ChromaDB query with `where={"session_id": ...}`. This closes the cross-user memory leak ADR-0004 was written to fix.

## The hybrid boundary for cached data

Taken literally, "never visible from another" would destroy the disk cache's containment reuse. ADR-0004's corrected split is:

- Public upstream downloads (`datasets.point_cloud`, `datasets.buildings`) stay shared. They are keyed on bounds and source alone. `get_buildings` has no entry of its own: it reads and writes the `datasets.buildings` download and summarises per request (#39).
- Builder results derived from user objects should be session-local, because `content_fingerprint` hashes only metadata (`type`, `source_op`, `nbytes`, `label`) and two Sessions can collide. Cross-session reuse of derived geometry is tracked as `TODOS.md` T-001.

## Known gaps (recorded, not hidden)

- `DiskCache` keys carry no Session identity yet, so the builder entries in `CACHE_ALLOWLIST` are still shared in code. See [Disk cache](../concepts/disk-cache.md).
- The Session id is client-supplied and unauthenticated until central auth lands (U11, #15). A client that invents ids gets a worker share and object budget per id.
- There is no Session expiry on the server, only the cap of 8. Budgets are an equal share rather than per-Session budgets (T11/U7, #23).

## Tests

`tests/test_http_sessions.py` starts a real HTTP server and checks:

- objects and runs are invisible to another Session and survive a reconnect of the same Session;
- a missing or empty header is refused;
- no per-connection state is kept;
- stdio works without a header;
- the cap evicts the LRU idle Session, never one in flight;
- the per-Session budgets add up to the process budget.

`tests/test_chatbot_sessions.py` covers `SessionManager`.
