# DTCC Chatbot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a web chatbot that wraps dtcc-agent via the Claude Agent SDK, exposing urban digital twin capabilities through a browser-based chat interface.

**Architecture:** FastAPI backend with WebSocket endpoint streams responses from `ClaudeSDKClient` (which connects to dtcc-agent as an MCP server). A single HTML file serves as the chat frontend.

**Tech Stack:** Python 3.12+, FastAPI, uvicorn, claude-agent-sdk, WebSocket, vanilla HTML/JS

---

### Task 1: Project Scaffolding

**Files:**
- Create: `chatbot/__init__.py`
- Create: `chatbot/static/` (directory)
- Modify: `pyproject.toml`

**Step 1: Create the chatbot package directory**

```bash
mkdir -p chatbot/static
```

**Step 2: Create `chatbot/__init__.py`**

```python
"""DTCC Chatbot — web chat interface powered by Claude Agent SDK."""
```

**Step 3: Add chatbot dependencies to `pyproject.toml`**

Add a `chatbot` optional dependency group:

```toml
[project.optional-dependencies]
test = ["pytest"]
chatbot = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "claude-agent-sdk>=0.1.0",
    "websockets>=13.0",
]
```

Update the packages.find to include `chatbot*`:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["dtcc_agent*", "chatbot*"]
```

**Step 4: Install the new dependencies**

Run: `pip install -e ".[chatbot]"`

**Step 5: Commit**

```bash
git add chatbot/__init__.py pyproject.toml
git commit -m "feat: scaffold chatbot package with dependencies"
```

---

### Task 2: Config Module

**Files:**
- Create: `chatbot/config.py`
- Test: `tests/test_chatbot_config.py`

**Step 1: Write the failing test**

```python
# tests/test_chatbot_config.py
from chatbot.config import SYSTEM_PROMPT, get_mcp_server_config


def test_system_prompt_mentions_sweden():
    assert "Sweden" in SYSTEM_PROMPT


def test_mcp_server_config_has_command():
    config = get_mcp_server_config()
    assert "dtcc-agent" in config
    dtcc = config["dtcc-agent"]
    assert "command" in dtcc
    assert "args" in dtcc
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chatbot_config.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the config module**

```python
# chatbot/config.py
"""Chatbot configuration: system prompt, MCP server config, defaults."""

from __future__ import annotations

import shutil

SYSTEM_PROMPT = """\
You are DTCC Assistant, an urban digital twin chatbot for Sweden.
You help users explore buildings, terrain, run heat/air quality simulations,
and visualize 3D city models anywhere in Sweden. Use the dtcc-agent tools
available to you. When showing simulation results, always render a 3D
visualization. Keep responses concise and focus on the data.\
"""

# Default port for the chatbot web server
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"


def get_mcp_server_config() -> dict:
    """Return MCP server configuration for dtcc-agent.

    Detects whether conda is available and builds the appropriate
    command to launch dtcc-agent in the fenicsx-env environment.
    """
    conda = shutil.which("conda")
    if conda is None:
        # Fallback: assume dtcc_agent is importable in current env
        return {
            "dtcc-agent": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "dtcc_agent"],
            }
        }

    return {
        "dtcc-agent": {
            "type": "stdio",
            "command": conda,
            "args": [
                "run", "--no-capture-output", "-n", "fenicsx-env",
                "python", "-m", "dtcc_agent",
            ],
        }
    }
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chatbot_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add chatbot/config.py tests/test_chatbot_config.py
git commit -m "feat: add chatbot config module with system prompt and MCP config"
```

---

### Task 3: Session Manager

**Files:**
- Create: `chatbot/sessions.py`
- Test: `tests/test_chatbot_sessions.py`

**Step 1: Write the failing test**

```python
# tests/test_chatbot_sessions.py
from chatbot.sessions import SessionManager


def test_create_session_returns_id():
    mgr = SessionManager()
    sid = mgr.create()
    assert isinstance(sid, str)
    assert len(sid) > 0


def test_get_session_returns_none_for_unknown():
    mgr = SessionManager()
    assert mgr.get("nonexistent") is None


def test_store_and_retrieve_session_id():
    mgr = SessionManager()
    sid = mgr.create()
    mgr.set_sdk_session(sid, "sdk-session-abc")
    assert mgr.get_sdk_session(sid) == "sdk-session-abc"


def test_remove_session():
    mgr = SessionManager()
    sid = mgr.create()
    mgr.remove(sid)
    assert mgr.get(sid) is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_chatbot_sessions.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the session manager**

```python
# chatbot/sessions.py
"""In-memory session manager for chatbot conversations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Session:
    """A chatbot session."""

    id: str
    created_at: datetime = field(default_factory=datetime.now)
    sdk_session_id: str | None = None


class SessionManager:
    """Manages chatbot sessions in memory.

    For prototype: no persistence. Sessions are lost on restart.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self) -> str:
        """Create a new session, return its ID."""
        sid = uuid.uuid4().hex[:12]
        self._sessions[sid] = Session(id=sid)
        return sid

    def get(self, session_id: str) -> Session | None:
        """Get a session by ID, or None if not found."""
        return self._sessions.get(session_id)

    def get_sdk_session(self, session_id: str) -> str | None:
        """Get the Agent SDK session ID for resuming."""
        session = self._sessions.get(session_id)
        return session.sdk_session_id if session else None

    def set_sdk_session(self, session_id: str, sdk_session_id: str) -> None:
        """Store the Agent SDK session ID for a session."""
        session = self._sessions.get(session_id)
        if session:
            session.sdk_session_id = sdk_session_id

    def remove(self, session_id: str) -> None:
        """Remove a session."""
        self._sessions.pop(session_id, None)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chatbot_sessions.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add chatbot/sessions.py tests/test_chatbot_sessions.py
git commit -m "feat: add in-memory session manager for chatbot"
```

---

### Task 4: FastAPI App with WebSocket + Agent SDK

**Files:**
- Create: `chatbot/app.py`

This is the core module. It wires together FastAPI, WebSocket, the Agent SDK, and session management.

**Step 1: Write `chatbot/app.py`**

```python
# chatbot/app.py
"""FastAPI application with WebSocket chat endpoint powered by Claude Agent SDK."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from chatbot.config import SYSTEM_PROMPT, get_mcp_server_config, DEFAULT_HOST, DEFAULT_PORT
from chatbot.sessions import SessionManager

logger = logging.getLogger(__name__)

app = FastAPI(title="DTCC Chatbot")
sessions = SessionManager()

# Serve rendered images from dtcc-agent
_renders_dir = Path("/tmp/dtcc_screenshots")
_renders_dir.mkdir(exist_ok=True)
app.mount("/renders", StaticFiles(directory=str(_renders_dir)), name="renders")

# Serve static frontend files
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def index():
    """Serve the chat UI."""
    html_path = _static_dir / "index.html"
    return HTMLResponse(html_path.read_text())


def _build_options(sdk_session_id: str | None = None) -> ClaudeAgentOptions:
    """Build Agent SDK options, optionally resuming a session."""
    opts = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers=get_mcp_server_config(),
        permission_mode="bypassPermissions",
        model="claude-sonnet-4-5",
    )
    if sdk_session_id:
        opts.resume = sdk_session_id
    return opts


async def _stream_response(
    client: ClaudeSDKClient,
    ws: WebSocket,
) -> str | None:
    """Stream Agent SDK responses over WebSocket. Returns SDK session_id."""
    sdk_session_id = None

    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    await ws.send_json({"type": "text", "content": block.text})
                elif isinstance(block, ToolUseBlock):
                    await ws.send_json({
                        "type": "tool_call",
                        "name": block.name,
                        "status": "running",
                    })

        elif isinstance(msg, ResultMessage):
            sdk_session_id = msg.session_id
            break

    return sdk_session_id


@app.websocket("/chat")
async def chat(ws: WebSocket):
    """WebSocket endpoint for chat conversations."""
    await ws.accept()

    # Read initial message to get or create session
    try:
        init = await ws.receive_json()
    except (WebSocketDisconnect, json.JSONDecodeError):
        return

    session_id = init.get("session_id")
    if not session_id or not sessions.get(session_id):
        session_id = sessions.create()

    # Send session ID to client
    await ws.send_json({"type": "session", "session_id": session_id})

    try:
        while True:
            data = await ws.receive_json()
            user_text = data.get("content", "").strip()
            if not user_text:
                continue

            await ws.send_json({"type": "status", "content": "thinking"})

            sdk_session_id = sessions.get_sdk_session(session_id)
            options = _build_options(sdk_session_id)

            async with ClaudeSDKClient(options=options) as client:
                await client.query(user_text)
                new_sdk_session = await _stream_response(client, ws)

                if new_sdk_session:
                    sessions.set_sdk_session(session_id, new_sdk_session)

            await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("Client disconnected from session %s", session_id)


def main():
    """Run the chatbot server."""
    import uvicorn
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
```

**Step 2: Verify it imports cleanly**

Run: `python -c "from chatbot.app import app; print('OK')"`
Expected: `OK` (requires claude-agent-sdk installed)

**Step 3: Commit**

```bash
git add chatbot/app.py
git commit -m "feat: add FastAPI app with WebSocket chat endpoint and Agent SDK"
```

---

### Task 5: Frontend Chat UI

**Files:**
- Create: `chatbot/static/index.html`

**Step 1: Write the chat UI**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DTCC Assistant</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f5f5f5;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background: #1a1a2e;
            color: white;
            padding: 1rem;
            text-align: center;
        }

        header h1 { font-size: 1.2rem; font-weight: 600; }
        header p { font-size: 0.8rem; opacity: 0.7; margin-top: 0.25rem; }

        #messages {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .msg {
            max-width: 80%;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            line-height: 1.5;
            font-size: 0.95rem;
        }

        .msg.user {
            align-self: flex-end;
            background: #1a1a2e;
            color: white;
        }

        .msg.assistant {
            align-self: flex-start;
            background: white;
            border: 1px solid #e0e0e0;
        }

        .msg.assistant table {
            border-collapse: collapse;
            margin: 0.5rem 0;
            font-size: 0.85rem;
        }

        .msg.assistant th, .msg.assistant td {
            border: 1px solid #ddd;
            padding: 0.3rem 0.6rem;
            text-align: left;
        }

        .msg.assistant img {
            max-width: 100%;
            border-radius: 8px;
            margin: 0.5rem 0;
        }

        .msg.assistant code {
            background: #f0f0f0;
            padding: 0.15rem 0.3rem;
            border-radius: 3px;
            font-size: 0.85em;
        }

        .msg.assistant pre {
            background: #f0f0f0;
            padding: 0.5rem;
            border-radius: 6px;
            overflow-x: auto;
        }

        .tool-pill {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.8rem;
            background: #e8f4fd;
            color: #1976d2;
            margin: 0.15rem 0;
        }

        .tool-pill.done { background: #e8f5e9; color: #388e3c; }

        #status {
            text-align: center;
            padding: 0.25rem;
            font-size: 0.8rem;
            color: #888;
            min-height: 1.5rem;
        }

        #input-area {
            display: flex;
            gap: 0.5rem;
            padding: 1rem;
            background: white;
            border-top: 1px solid #e0e0e0;
        }

        #input-area textarea {
            flex: 1;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 0.95rem;
            font-family: inherit;
            resize: none;
            rows: 1;
        }

        #input-area button {
            padding: 0.75rem 1.5rem;
            background: #1a1a2e;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.95rem;
        }

        #input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
    </style>
</head>
<body>
    <header>
        <h1>DTCC Assistant</h1>
        <p>Urban digital twin chatbot for Sweden</p>
    </header>

    <div id="messages"></div>
    <div id="status"></div>

    <div id="input-area">
        <textarea id="input" placeholder="Ask about buildings, terrain, heat simulations..."
                  rows="1"></textarea>
        <button id="send" onclick="sendMessage()">Send</button>
    </div>

    <script>
        const messagesEl = document.getElementById("messages");
        const inputEl = document.getElementById("input");
        const sendBtn = document.getElementById("send");
        const statusEl = document.getElementById("status");

        let ws = null;
        let sessionId = null;
        let currentAssistantEl = null;
        let currentAssistantText = "";

        function connect() {
            const proto = location.protocol === "https:" ? "wss:" : "ws:";
            ws = new WebSocket(`${proto}//${location.host}/chat`);

            ws.onopen = () => {
                ws.send(JSON.stringify({ session_id: sessionId }));
                statusEl.textContent = "Connected";
                setTimeout(() => { statusEl.textContent = ""; }, 1500);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };

            ws.onclose = () => {
                statusEl.textContent = "Disconnected. Reconnecting...";
                setTimeout(connect, 2000);
            };
        }

        function handleMessage(data) {
            switch (data.type) {
                case "session":
                    sessionId = data.session_id;
                    break;

                case "status":
                    statusEl.textContent = data.content === "thinking"
                        ? "Thinking..." : data.content;
                    break;

                case "text":
                    if (!currentAssistantEl) {
                        currentAssistantEl = addMessage("", "assistant");
                        currentAssistantText = "";
                    }
                    currentAssistantText += data.content;
                    currentAssistantEl.innerHTML = marked.parse(currentAssistantText);
                    messagesEl.scrollTop = messagesEl.scrollHeight;
                    break;

                case "tool_call":
                    const pill = document.createElement("div");
                    pill.className = `tool-pill ${data.status === "done" ? "done" : ""}`;
                    pill.textContent = data.status === "done"
                        ? `${data.name} done` : `${data.name}...`;
                    messagesEl.appendChild(pill);
                    statusEl.textContent = data.status === "done"
                        ? "" : `Running ${data.name}...`;
                    messagesEl.scrollTop = messagesEl.scrollHeight;
                    break;

                case "image":
                    if (!currentAssistantEl) {
                        currentAssistantEl = addMessage("", "assistant");
                        currentAssistantText = "";
                    }
                    currentAssistantText += `\n\n![render](${data.url})\n\n`;
                    currentAssistantEl.innerHTML = marked.parse(currentAssistantText);
                    messagesEl.scrollTop = messagesEl.scrollHeight;
                    break;

                case "done":
                    currentAssistantEl = null;
                    currentAssistantText = "";
                    statusEl.textContent = "";
                    sendBtn.disabled = false;
                    inputEl.disabled = false;
                    break;
            }
        }

        function addMessage(text, role) {
            const el = document.createElement("div");
            el.className = `msg ${role}`;
            if (role === "user") {
                el.textContent = text;
            } else {
                el.innerHTML = marked.parse(text);
            }
            messagesEl.appendChild(el);
            messagesEl.scrollTop = messagesEl.scrollHeight;
            return el;
        }

        function sendMessage() {
            const text = inputEl.value.trim();
            if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

            addMessage(text, "user");
            ws.send(JSON.stringify({ content: text }));
            inputEl.value = "";
            sendBtn.disabled = true;
            inputEl.disabled = true;
        }

        inputEl.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        connect();
    </script>
</body>
</html>
```

**Step 2: Verify the file is well-formed**

Run: `python -c "from pathlib import Path; html = Path('chatbot/static/index.html').read_text(); assert '<html' in html; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add chatbot/static/index.html
git commit -m "feat: add chat UI as single HTML file"
```

---

### Task 6: Module Entry Point

**Files:**
- Create: `chatbot/__main__.py`

**Step 1: Write the entry point**

```python
# chatbot/__main__.py
"""Entry point for `python -m chatbot`."""

from chatbot.app import main

main()
```

**Step 2: Verify it launches**

Run: `python -m chatbot` — should start uvicorn on port 8000. Ctrl+C to stop.

**Step 3: Commit**

```bash
git add chatbot/__main__.py
git commit -m "feat: add chatbot entry point for python -m chatbot"
```

---

### Task 7: End-to-End Smoke Test

**Files:**
- Create: `tests/test_chatbot_app.py`

**Step 1: Write a WebSocket integration test**

This tests the FastAPI app directly (no Agent SDK needed — mocks it).

```python
# tests/test_chatbot_app.py
"""Smoke tests for the chatbot FastAPI app."""

import pytest
from fastapi.testclient import TestClient
from chatbot.app import app


def test_index_returns_html():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "DTCC Assistant" in resp.text


def test_websocket_session_handshake():
    client = TestClient(app)
    with client.websocket_connect("/chat") as ws:
        # Send init message
        ws.send_json({"session_id": None})
        # Should receive session message
        msg = ws.receive_json()
        assert msg["type"] == "session"
        assert "session_id" in msg
```

**Step 2: Run tests**

Run: `pytest tests/test_chatbot_app.py -v`
Expected: Both tests pass (WebSocket test connects but doesn't trigger Agent SDK)

**Step 3: Commit**

```bash
git add tests/test_chatbot_app.py
git commit -m "test: add smoke tests for chatbot HTTP and WebSocket endpoints"
```

---

### Task 8: Documentation Update

**Files:**
- Modify: `README.md`

**Step 1: Add chatbot section to README**

Append a section to the existing README:

```markdown
## Web Chatbot

The chatbot provides a browser-based chat interface powered by Claude Agent SDK.

### Install

```bash
pip install -e ".[chatbot]"
```

### Run

```bash
python -m chatbot
```

Then open http://localhost:8000 in your browser.

### How it works

The chatbot uses the Claude Agent SDK to connect to dtcc-agent as an MCP server.
User messages are sent via WebSocket, and Claude's responses (including tool calls
and 3D renders) stream back in real-time.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add chatbot section to README"
```
