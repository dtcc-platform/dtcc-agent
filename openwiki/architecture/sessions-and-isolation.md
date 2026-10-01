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
  - id: openwiki-source-052f7c9f16ee5a8169a3fb7d
    resource: repo://dtcc_agent/disk_cache.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
  - id: openwiki-source-7da8cb11cdc15fb1e5a1f088
    resource: repo://tests/test_http_sessions.py
generated: { by: "claude-code", at: "2026-10-01T20:29:16.810Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-10-01T20:29:16.810Z
---

# Sessions and isolation

**The Session is the isolation unit** (ADR-0004, accepted 2026-09-19). Objects, Runs, conversation memory and budgets each belong to exactly one Session and are never visible from another. The ADR rejected the alternative, a shared store with session-scoped views, because that makes isolation a property of every query and so something a later feature can quietly break.

A Session is one continuous conversation, identified anonymously and scoped to a browser. Session lifetime is not person lifetime. When authentication arrives, a *subject* will own many Sessions; the subject identifier is a separate field and does not replace the Session.

## Server side (`dtcc_agent/server.py`)

`_Session` is a dataclass holding everything one Session owns:

- `id`: the chatbot's session id (the `X-DTCC-Session` value over HTTP; `DTCC_AGENT_SESSION` or `"local"` over stdio). It names the Session's artifact folder, see [Artifacts and the file boundary](../concepts/artifacts-and-file-boundary.md)
- `objects`: its own `ObjectStore`, drawing on the process-wide memory budget
- `results`: its simulation Run records, keyed by Run reference (`run_…`), oldest first and capped at `MAX_RUNS = 100`. A record holds what was run and the `object_ref` of its result; the result itself lives in the Session's `objects` (U6, ADR-0010)
- `in_flight`: the number of tool calls currently running
- `workers`: its share of the process worker pool (see [MCP server and tool execution](mcp-server-and-tool-execution.md))

### Resolving the caller

`_request_session()` reads the MCP `request_ctx`:

- **HTTP request present.** The `X-DTCC-Session` header is required. A missing or empty header raises `ToolError("Refused: ...")`. Otherwise `_session_for(id, acquire=True)` returns the live Session and marks a call in flight in the same locked step.
- **No request, while serving HTTP.** The call is refused rather than being handed the local Session, which would otherwise be shared between clients.
- **No request under stdio, or an in-process caller.** Returns `None`, and `_session()` falls back to the single process-wide `_local_session`.

Sessions are keyed by the header value rather than by MCP transport session, because the chatbot opens a new connection per message. Objects therefore survive into the next connection of the same Session.

### Cap, budget and eviction

- `MAX_SESSIONS = 8` Sessions are kept in an `OrderedDict` in least-recently-used order.
- Memory is not split into equal shares any more (T11, #66). Every Session's store draws on one `MemoryBudget` of `OBJECT_BUDGET_BYTES` (2 GiB); when the stores together pass it, the least recently used Object in any Session is evicted. An HTTP Session may hold at most `SESSION_OBJECT_BYTES` (1 GiB); the local stdio Session may use the whole budget and the whole worker pool. Details in [Dispatch and the object store](../concepts/dispatch-and-object-store.md).
- `_evict_idle_excess()` drops the least recently used **idle** Sessions beyond the cap, with their runs, and clears their store so its bytes go back to the budget at once. A Session with a call in flight is never evicted. If every Session is busy the cap is exceeded temporarily, and the excess is trimmed as calls finish (`_release`).

## Chatbot side

- `chatbot/sessions.py` `SessionManager` mints a 12-hex-char id per browser session, keeps it in memory for up to an hour (cleaned on `create()`), and stores the Agent SDK session id used for resume. Removing or expiring a session also deletes its artifact folder.
- `chatbot/config.py` `get_mcp_server_config(session_id)`: with `DTCC_MCP_URL` set, it connects over HTTP and sends `X-DTCC-Session: <session_id>`. Otherwise it spawns the server over stdio, one child process per message, and passes the id as `DTCC_AGENT_SESSION` so the child writes into that Session's artifact folder.
- The chatbot's `/artifacts/<session>/<name>` route serves a file only while that session is live in `SessionManager`.
- `chatbot/memory.py` `ConversationMemory.retrieve()` filters the ChromaDB query with `where={"session_id": ...}`. This closes the cross-user memory leak ADR-0004 was written to fix.

## The hybrid boundary for cached data

Taken literally, "never visible from another" would destroy the disk cache's containment reuse. ADR-0004's corrected split is:

- Public upstream downloads (`datasets.point_cloud`, `datasets.buildings`) stay shared. They are keyed on bounds and source alone. `get_buildings` has no entry of its own: it reads and writes the `datasets.buildings` download and summarises per request (#39).
- Builder results are not cached at all since U2 (#11, #62): their old keys described inputs by metadata only and could collide across Sessions. `CACHE_ALLOWLIST` holds just the two downloads, so nothing derived from a user's objects is shared. Cross-session reuse of derived geometry would need provenance keys (`TODOS.md` T-001).

## Known gaps (recorded, not hidden)

- The Session id is client-supplied and unauthenticated until admission control lands (T14). In the two-service deployment the MCP port is reachable only over loopback from the chatbot (U11), and an artifact URL is a capability: the live session id plus a random token.
- There is no Session expiry on the server, only the cap of 8 (U7 is open). Memory held by idle Sessions is reclaimed by the shared budget's eviction.

## Tests

`tests/test_http_sessions.py` starts a real HTTP server and checks:

- objects and runs are invisible to another Session and survive a reconnect of the same Session;
- a missing or empty header is refused;
- no per-connection state is kept;
- stdio works without a header;
- the cap evicts the LRU idle Session, never one in flight;
- every Session draws on the one process budget, and a dropped Session hands its bytes back.

`tests/test_chatbot_sessions.py` covers `SessionManager`.
