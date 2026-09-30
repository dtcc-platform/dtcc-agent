---
type: integration
title: Lurkie chatbot
description: The chatbot package covers the FastAPI app and its WebSocket /chat protocol, the Claude Agent SDK loop with resume and retry, the system prompt, how it connects to the MCP server, ChromaDB conversation memory, and the render-serving gap.
tags: [chatbot, lurkie, websocket, claude-agent-sdk, memory, ui]
sources:
  - id: openwiki-source-d82fbc21a9f74516f7bfd0f8
    resource: repo://chatbot/app.py
  - id: openwiki-source-778a883bcdc0a6ed0b3401f7
    resource: repo://chatbot/config.py
  - id: openwiki-source-931ea4e3e14cfe3c996abf4a
    resource: repo://chatbot/memory.py
  - id: openwiki-source-b14e7ad1e4ed8425d47e20d4
    resource: repo://chatbot/static/index.html
  - id: openwiki-source-e00d8f9ae7bfdd9a24f35525
    resource: repo://dtcc_agent/renderer.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
  - id: openwiki-source-c0b62da1c8d12500b49cd428
    resource: repo://tests/test_catalogue_startup.py
generated: { by: "claude-code", at: "2026-09-27T19:28:40.080Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-30T14:41:08.402Z
---

# Lurkie chatbot

`chatbot/` is "DTCC Lurkie", the web front door. It is a FastAPI app (`chatbot/app.py`, served by `uvicorn chatbot.app:app` on port 8050 in Docker) that runs a Claude agent against the dtcc-agent MCP server and streams the answer to a single-page UI (`chatbot/static/index.html`).

## HTTP surface

| Route | Purpose |
|---|---|
| `GET /` | Serves `static/index.html`, or a placeholder if the file is missing |
| `GET /health` | `{"status": "ok", "service": "dtcc-agent", "component": "lurkie"}` for the Docker healthcheck |
| `/static/*` | UI assets |
| `/renders/*` | Files under `DTCC_AGENT_RENDERS_DIR` (default `/tmp/dtcc_screenshots`) |
| `WS /chat` | The conversation |

## The `/chat` protocol

1. **Session.** The client sends `{"session_id": <stored or null>}`. The server reuses a known id or mints a new one (`SessionManager.create()`), then replies `{"type": "session", "session_id": ...}`.
2. **Messages.** Each client message is `{"content": "..."}`. Messages over 10,000 characters are rejected with a text reply. `{"type": "new_chat"}` clears the stored SDK session so the next turn starts fresh.
3. **Replies.** For each turn the server sends:
   - `status: thinking`;
   - a `text` frame for every assistant `TextBlock`;
   - a `tool_call` frame (`name`, `status: running`) for every `ToolUseBlock`;
   - finally `done`.

The UI renders markdown through `marked` and `DOMPurify.sanitize`. It also handles an `image` frame (`data.url`), but **the server never sends one**. Rendered PNGs are written by `renderer.render_to_file` into a fresh `tempfile.mkdtemp(prefix="dtcc_screenshots_")` directory, not into the `/renders` mount. Images therefore reach the browser only if the model mentions a path it can serve. The rebuild plan leaves whether Lurkie renders geometry open (D1, D2).

## The agent loop

Each user message builds `ClaudeAgentOptions` in `_build_options`:

- the `SYSTEM_PROMPT` from `chatbot/config.py`, plus any retrieved memory;
- `mcp_servers=get_mcp_server_config(session_id)`;
- `permission_mode="bypassPermissions"` (commented as prototype-only; production should use an explicit allowlist);
- the hardcoded model `claude-sonnet-4-5`.

It then opens a `ClaudeSDKClient`, sends the query and streams `_stream_response`. That loop logs every block and returns the SDK session id from the `ResultMessage`.

- **Resume.** The SDK session id is stored per chat session and passed as `resume` on the next turn.
- **Retry.** If a resumed call raises (for example, a context-window limit), the stored SDK session is cleared and the turn is retried once fresh. If that also fails, or a fresh call fails, the user gets an apology text.
- **Nested Claude Code.** `CLAUDECODE` is popped from the environment at import so the chatbot can start from inside a Claude Code terminal.

ADR-0003 replaces this SDK with pydantic-ai for provider portability (M3). Until then there is no model configuration.

## The system prompt

`SYSTEM_PROMPT` casts the assistant as an urban digital twin chatbot for Sweden. It tells the model to:

- use a 250 m geocoding radius by default;
- prefer rendering meshes over rasters;
- parallelise tool calls.

It **embeds parameter schemas for seven common operations** (`datasets.point_cloud`, `datasets.buildings`, `builder.build_terrain_raster`, `builder.raster.slope_aspect`, `builder.build_terrain_surface_mesh`, `builder.build_city_surface_mesh`, `builder.pc_filter.classification_filter`) so the model can skip `describe_operation`. `CONTEXT.md` records that these are to move to versioned configuration as-is. ADR-0006 plans a stable, cacheable prompt prefix.

## Connecting to the MCP server

`get_mcp_server_config(session_id)`:

- **With `DTCC_MCP_URL` set,** connects to a running streamable-http server and sends `X-DTCC-Session: <session_id>`.
- **Otherwise,** launches `python -m dtcc_agent` over stdio with the current interpreter (or `DTCC_AGENT_PYTHON`).
  That server does not build the operation catalogue at startup, since most messages never read it: the first tool call that needs it pays the build (about a second) and, on a broken Core install, fails naming the section. The chatbot does not check the MCP server's status, so this is where such a failure shows.

See [Sessions and isolation](../architecture/sessions-and-isolation.md).

## Conversation memory

`chatbot/memory.py` `ConversationMemory` is a persistent ChromaDB collection (`conversations`, cosine space) under `DTCC_AGENT_MEMORY_DIR` (default `/tmp/dtcc_lurkie_memory`).

- **Store.** After every turn that produced text, the exchange is stored with `session_id` metadata.
- **Retrieve.** On a **fresh** (non-resumed) SDK session, `retrieve()` fetches up to 5 exchanges from **the same Session only**, keeps those with distance under 0.8, and appends them to the system prompt as "excerpts from earlier in this session". Resumed sessions skip retrieval because the history is already in context.

This is conversation memory, not a Corpus. Document retrieval is ADR-0005's separate `dtcc-docs` server.

## Logging

Logs go to both the console and `DTCC_AGENT_LOG_DIR/lurkie-<timestamp>.log` (default `/tmp/dtcc_lurkie_logs`). Tool inputs and results are truncated previews.

## Tests

- `tests/test_chatbot_app.py` uses mock Chroma and the SDK, and needs the `chatbot` extra; without it, collection fails.
- `tests/test_chatbot_config.py` checks that the session header is carried, the stdio command, and the interpreter override.
- `tests/test_chatbot_memory.py` checks session-filtered retrieval and the distance threshold.
- `tests/test_chatbot_sessions.py` covers `SessionManager`.
