---
type: integration
title: Lurkie chatbot
description: The chatbot package's FastAPI app, Lurkie. It serves the page, admits chats with an access code, runs the /chat WebSocket protocol, hands each turn to the pydantic-ai runtime, stores memory, writes one answer record per turn, and serves artifacts while their session is live.
tags: [chatbot, lurkie, websocket, claude-agent-sdk, memory, ui]
sources:
  - id: openwiki-source-d82fbc21a9f74516f7bfd0f8
    resource: repo://chatbot/app.py
  - id: openwiki-source-778a883bcdc0a6ed0b3401f7
    resource: repo://chatbot/config.py
  - id: openwiki-source-931ea4e3e14cfe3c996abf4a
    resource: repo://chatbot/memory.py
  - id: openwiki-source-5f2b4e0b591e0b8e093c4c16
    resource: repo://chatbot/runtime/__init__.py
  - id: openwiki-source-6ed7b95a502f0a6d7126df1d
    resource: repo://chatbot/runtime/pai.py
  - id: openwiki-source-b14e7ad1e4ed8425d47e20d4
    resource: repo://chatbot/static/index.html
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
  - id: openwiki-source-c0b62da1c8d12500b49cd428
    resource: repo://tests/test_catalogue_startup.py
  - id: openwiki-source-9b5246b31b1e97d9e4442ac9
    resource: repo://tests/test_chatbot_app.py
generated: { by: "claude-code", at: "2026-10-13T19:36:58.526Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-10-13T19:36:58.526Z
---

# Chatbot (Lurkie)

`chatbot/app.py` is the web app users talk to: a FastAPI server (`python -m chatbot`, port `DTCC_AGENT_PORT`, default 8050) with a single-page UI in `chatbot/static/index.html`. It owns the conversation plumbing: admission, sessions, the WebSocket protocol, memory writes and answer records. Each turn's agent loop runs in `chatbot/runtime/pai.py`; see [Model runtime](../architecture/model-runtime.md).

## Routes

| Route | What it does |
|---|---|
| `GET /` | The chat page. |
| `GET /admission` | `{"required": true}` when the page must ask for an access code before opening a chat. |
| `GET /health` | `{"status": "ok", ...}` for Compose health checks. |
| `GET /artifacts/{session_id}/{name}` | A file a tool produced, served only while that session is live, as `image/png` for images and as a download otherwise, with `Cache-Control: private, no-store` and `nosniff`. See [Artifacts and the file boundary](../concepts/artifacts-and-file-boundary.md). |
| `WS /chat` | The conversation (below). |

## Startup

The app's lifespan exits unless an AWS credential is set, logs the model and runtime, and starts the background Bedrock key probe. At import, `load_runtime()` exits if `DTCC_AGENT_RUNTIME` still asks for the removed Agent SDK, and `load_access_code()` exits on an access code shorter than 16 characters.

## The `/chat` protocol

1. **Open.** The page sends one JSON message: `{"session_id": ...}` to resume, plus `access_code` when opening a new chat. A live session resumes on its id alone. Otherwise the code is checked with `hmac.compare_digest`; a missing or wrong one gets `{"type": "error", "code": "admission_required"}` and a close with **4401**, and no session is created. The server then sends `{"type": "session", "session_id": ...}`.
2. **Each message.** If the session has been idle for the hour (U7), the socket closes with **4408** and the page starts a new session. `{"type": "new_chat"}` clears the conversation history but keeps the session and its files. An empty message is ignored; one over 10,000 characters gets a short text reply and `done`.
3. **A turn.** Inside `sessions.turn(...)`, so the session cannot expire mid-turn, the server sends `{"type": "status", "content": "thinking"}`, calls `runtime.answer(...)`, which streams `text`, `tool_call`, `image` and `file` frames, stores the exchange in conversation memory off the event loop (#94), and finishes with a `provenance` frame and `done`.

Exactly one `answers.jsonl` line is written per turn, whatever happens: a fresh retry adds to it, and a turn cut short by a closed socket or a cancelled task still writes its record. See [Provenance and evaluation](../operations/provenance-and-evaluation.md).

## Sessions and memory

`chatbot/sessions.py` keeps sessions in memory: a 12-hex-character id, the subject, the pydantic-ai message history, a lock for the running turn and the last activity time. A session ends after an hour without a message and never during a turn; ending it deletes its artifact folder. Conversation memory (`chatbot/memory.py`, ChromaDB) is written after each answered turn and read only at the start of a fresh conversation, from the same session's exchanges. Details in [Sessions and isolation](../architecture/sessions-and-isolation.md).

## Reaching the tools

The runtime connects each turn to the dtcc-agent MCP server: over streamable HTTP when `DTCC_MCP_URL` is set (the two-service deployment), with the session, subject and turn in headers and the bearer secret, or as a stdio child process otherwise. The stdio server the chatbot spawns does not build the operation catalogue at startup; the first tool call that reads it builds it, and a broken Core then fails that call rather than server startup.

## Tests

- `tests/test_chatbot_app.py`: the artifact route's 200/404 cases and headers, artifact frames, admission (4401 without the right code, open without one when admission is off), session expiry on the socket, one answer record per turn including cancelled ones.
- `tests/test_chatbot_sessions.py`: `SessionManager`, including idle expiry by last message and never during a turn.
- `tests/test_chatbot_runtime_pai.py` and `tests/test_bedrock_auth.py`: the turn itself (see [Model runtime](../architecture/model-runtime.md)).
