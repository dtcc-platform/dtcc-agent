---
type: architecture
title: Sessions and isolation
description: How dtcc-agent makes the Session the isolation unit (ADR-0004), covering the X-DTCC-Session header, per-Session object stores and runs, the eight-Session cap with LRU eviction, the local stdio Session, and the gaps that remain.
tags: [session, isolation, security, adr-0004, http]
sources:
  - id: openwiki-source-d82fbc21a9f74516f7bfd0f8
    resource: repo://chatbot/app.py
  - id: openwiki-source-778a883bcdc0a6ed0b3401f7
    resource: repo://chatbot/config.py
  - id: openwiki-source-931ea4e3e14cfe3c996abf4a
    resource: repo://chatbot/memory.py
  - id: openwiki-source-f4e2e41b78ef4f0591d5eac2
    resource: repo://chatbot/sessions.py
  - id: openwiki-source-2d18eac28ce6bc775094dd62
    resource: repo://docs/adr/0004-session-is-the-isolation-unit.md
  - id: openwiki-source-052f7c9f16ee5a8169a3fb7d
    resource: repo://dtcc_agent/disk_cache.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
  - id: openwiki-source-9b5246b31b1e97d9e4442ac9
    resource: repo://tests/test_chatbot_app.py
  - id: openwiki-source-b861a39afd7c88e57af52d10
    resource: repo://tests/test_chatbot_sessions.py
  - id: openwiki-source-7da8cb11cdc15fb1e5a1f088
    resource: repo://tests/test_http_sessions.py
generated: { by: "claude-code", at: "2026-10-31T10:29:51.442Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-10-31T10:29:51.442Z
---

# Sessions and isolation

**The Session is the isolation unit** (ADR-0004, accepted 2026-09-19). Objects, Runs, conversation memory and budgets each belong to exactly one Session and are never visible from another. The ADR rejected the alternative, a shared store with session-scoped views, because that makes isolation a property of every query and so something a later feature can quietly break.

A Session is one continuous conversation, identified anonymously and scoped to a browser. Session lifetime is not person lifetime. When authentication arrives, a *subject* will own many Sessions; the subject identifier is a separate field and does not replace the Session.

## Server side (`dtcc_agent/server.py`)

`_Session` is a dataclass holding everything one Session owns:

- `id`: the chatbot's session id (the `X-DTCC-Session` value over HTTP; `DTCC_AGENT_SESSION` or `"local"` over stdio). It names the Session's artifact folder, see [Artifacts and the file boundary](../concepts/artifacts-and-file-boundary.md)
- `objects`: its own `ObjectStore`, drawing on the process-wide memory budget
- `results`: its simulation Run records, keyed by Run reference (`run_…`), oldest first and capped at `MAX_RUNS = 100`. A record holds what was run and the `object_ref` of its result; the result itself lives in the Session's `objects` (U6, ADR-0010)
- `subject`: who the Session acts for, from the `X-DTCC-Subject` header when it is created (`DTCC_AGENT_SUBJECT` over stdio). It is `"anonymous"` until central auth exists, and nothing is authorised by it yet (T14)
- `in_flight`: the number of tool calls currently running
- `last_used`: when its last call started or ended, for idle expiry
- `workers`: its share of the process worker pool (see [MCP server and tool execution](mcp-server-and-tool-execution.md))

### Resolving the caller

`_request_session()` reads the request headers that MCPServer hands each tool wrapper as `mcp_context`:

- **HTTP request present.** The `X-DTCC-Session` header is required. A missing or empty header raises `ToolError("Refused: ...")`. Otherwise `_session_for(id, acquire=True)` returns the live Session and marks a call in flight in the same locked step.
- **No request, while serving HTTP.** The call is refused rather than being handed the local Session, which would otherwise be shared between clients.
- **No request under stdio, or an in-process caller.** Returns `None`, and `_session()` falls back to the single process-wide `_local_session`.

Sessions are keyed by the header value rather than by MCP transport session, because the chatbot opens a new connection per message. Objects therefore survive into the next connection of the same Session.

### Cap, budget and eviction

- `MAX_SESSIONS = 8` Sessions are kept in an `OrderedDict` in least-recently-used order.
- A Session unused for `SESSION_IDLE_SECONDS` (3600 s) is dropped with its Objects (U7, #71). The chatbot expires its sessions after the same idle hour, and a test holds the two constants equal. An expired id that comes back gets a fresh, empty Session. The local stdio Session never expires.
- Memory is not split into equal shares any more (T11, #66). Every Session's store draws on one `MemoryBudget` of `OBJECT_BUDGET_BYTES` (2 GiB); when the stores together pass it, the least recently used Object in any Session is evicted. An HTTP Session may hold at most `SESSION_OBJECT_BYTES` (1 GiB); the local stdio Session may use the whole budget and the whole worker pool. Details in [Dispatch and the object store](../concepts/dispatch-and-object-store.md).
- `_evict_idle_excess()` first drops idle Sessions past the idle hour, then the least recently used **idle** Sessions beyond the cap, with their runs, and clears their store so its bytes go back to the budget at once. A Session with a call in flight is never evicted. If every Session is busy the cap is exceeded temporarily, and the excess is trimmed as calls finish (`_release`).

## Chatbot side

- `chatbot/sessions.py` `SessionManager` mints a 12-hex-char id per browser session and keeps it in memory. A session ends after an hour **without a message**, not an hour after it started (U7, #71), and never while a turn is running (`turns`). The next `get()` of it or the next `create()` removes it, and removing a session also deletes its artifact folder. A session holds the conversation's pydantic-ai message history and a lock held for a whole turn, so two tabs on one session take turns; "new chat" clears the history but keeps the session and its files. See [Model runtime](model-runtime.md).
- `chatbot/config.py` `get_mcp_server_config(session_id, subject, turn_id)`: with `DTCC_MCP_URL` set, it connects over HTTP and sends `X-DTCC-Session`, `X-DTCC-Subject`, `X-DTCC-Turn` and, when `DTCC_MCP_SECRET` is set, `Authorization: Bearer <secret>`. Otherwise it spawns the server over stdio, one child process per message, and passes the same identity as `DTCC_AGENT_SESSION`, `DTCC_AGENT_SUBJECT` and `DTCC_AGENT_TURN`, so the child writes into that Session's artifact folder.
- The chatbot's `/artifacts/<session>/<name>` route serves a file only while that session is live in `SessionManager`.
- `chatbot/memory.py` `ConversationMemory.retrieve()` filters the ChromaDB query with `where={"session_id": ...}`. This closes the cross-user memory leak ADR-0004 was written to fix.

## Admission (T14, #72)

Two checks keep a deployment from running open by accident; Compose refuses to start without either.

- **Resuming is not activity** (#115). The page reconnects by itself whenever its socket drops, so a reconnect on a live session id does not count toward the idle hour; only a message or the end of a turn does. A tab left open still expires.
- **Each chat counts its turns.** `Session.turns_started` counts every turn the chat has started, refused and failed ones included, against `DTCC_AGENT_TURNS_PER_CHAT` (#132). "New chat" in the page keeps the session, so it keeps the count.
- **Opening a chat needs an access code.** `DTCC_AGENT_ACCESS_CODE` (at least 16 characters, or the chatbot does not start) is compared with `hmac.compare_digest`. A missing or wrong code closes the websocket with 4401 and creates no session; a live session resumes on its id alone. Unset, admission is off and the chatbot logs a warning, repeated every 100 chats opened.
- **Every MCP call needs the secret.** With `DTCC_MCP_SECRET` set, the MCP server answers any HTTP request without `Authorization: Bearer <secret>` with a bare 401; without it, the server refuses to bind a non-loopback host. See [MCP server and tool execution](mcp-server-and-tool-execution.md).

## Deployment-wide ceilings are not Session budgets

ADR-0004 has a recorded exception (#132): the monthly model-spend limit and the cap on model requests at once belong to the deployment, not to a Session. They bound what the operator pays and what Bedrock is asked for, across everyone, so one chat can use up the month for all. They hold only amounts and counts, never another Session's objects, runs or memory. Per-Session budgets (the object-budget share, the turn limit) stay per-Session. See [Turn and spend limits](../operations/turn-and-spend-limits.md).

## The hybrid boundary for cached data

Taken literally, "never visible from another" would destroy the disk cache's containment reuse. ADR-0004's corrected split is:

- Public upstream downloads (`datasets.point_cloud`, `datasets.buildings`) stay shared. They are keyed on bounds and source alone. `get_buildings` has no entry of its own: it reads and writes the `datasets.buildings` download and summarises per request (#39).
- Builder results are not cached at all since U2 (#11, #62): their old keys described inputs by metadata only and could collide across Sessions. `CACHE_ALLOWLIST` holds just the two downloads, so nothing derived from a user's objects is shared. Cross-session reuse of derived geometry would need provenance keys (`TODOS.md` T-001).

## Known gaps (recorded, not hidden)

- The Session id is client-supplied. The access code admits a person to open chats; it does not authenticate who they are, and the subject stays `"anonymous"` until central auth (Napo's backend and the API gateway) exists. In the two-service deployment the MCP port is reachable only over loopback from the chatbot (U11), and an artifact URL is a capability: the live session id plus a random token.
- Sessions live in each process's memory, so several agent instances behind a load balancer need sticky routing per chat. A restart loses open conversations; the cache and artifacts on `/data` survive.

## Tests

`tests/test_http_sessions.py` starts a real HTTP server and checks:

- objects and runs are invisible to another Session and survive a reconnect of the same Session;
- a missing or empty header is refused;
- no per-connection state is kept;
- stdio works without a header;
- the cap evicts the LRU idle Session, never one in flight, and an idle hour drops a Session;
- a server with a secret refuses requests without it;
- every Session draws on the one process budget, and a dropped Session hands its bytes back.

`tests/test_chatbot_sessions.py` covers `SessionManager`, including idle expiry by last message and never during a turn. `tests/test_chatbot_app.py` covers the access code.
