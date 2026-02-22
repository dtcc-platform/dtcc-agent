# DTCC Chatbot Design

**Date:** 2026-02-18
**Status:** Approved

## Goal

Turn dtcc-agent into a web-based chatbot for non-technical end users (urban planners, citizens, stakeholders) covering all of Sweden. Users type natural language questions ("How hot does Lindholmen get in a heatwave?" or "Show me the tallest buildings in Malmö") and the system orchestrates simulations, data queries, and 3D visualizations autonomously. The underlying data (Lantmäteriet, EPSG:3006, Nominatim) is Sweden-wide; the chatbot should reflect this.

## Architecture

```
Browser (single-page chat UI)
    ↕ WebSocket (ws://host/chat)
FastAPI backend
    ↕ Claude Agent SDK (query())
        ↕ MCP stdio
    dtcc-agent (existing, unchanged)
        ↕ in-process
    dtcc-core / dtcc-sim / dtcc-viewer
```

**Engine:** Claude Agent SDK (Python) — Claude Code as a library. Provides the agentic tool-use loop, streaming, session management, and MCP integration.

**Transport:** WebSocket for real-time streaming of tokens, tool-call status, and images.

## Backend Design

### FastAPI app (`chatbot/app.py`)

Three responsibilities:

1. **WebSocket endpoint `/chat`**
   - Client connects, optionally sends `session_id` to resume
   - Each message triggers `claude_agent_sdk.query()` with user text
   - Streams three event types:
     - `{"type": "text", "content": "..."}` — response tokens
     - `{"type": "tool_call", "name": "geocode", "status": "running"}` — tool activity
     - `{"type": "image", "url": "/renders/abc.png"}` — 3D screenshots

2. **Session management (`chatbot/sessions.py`)**
   - In-memory dict: `session_id` → Agent SDK session
   - Enables multi-turn conversations
   - No persistence for prototype (sessions lost on restart)

3. **Static file serving**
   - Mounts `/tmp/dtcc_screenshots_*/` as `/renders/`
   - Serves `chatbot/static/index.html` as the frontend

### Agent SDK Configuration

```python
options = ClaudeAgentOptions(
    mcp_servers={
        "dtcc-agent": {
            "command": "conda",
            "args": ["run", "--no-capture-output", "-n", "fenicsx-env",
                     "python", "-m", "dtcc_agent"]
        }
    },
    allowed_tools=[...],  # All dtcc-agent MCP tools
    permission_mode="bypassPermissions",
)
```

### System Prompt

```
You are DTCC Assistant, an urban digital twin chatbot for Sweden.
You help users explore buildings, terrain, run heat/air quality simulations,
and visualize 3D city models anywhere in Sweden. Use the dtcc-agent tools
available to you. When showing simulation results, always render a 3D
visualization. Keep responses concise and focus on the data.
```

## Frontend Design

### Single HTML file (`chatbot/static/index.html`)

Minimal chat interface with:
- **Message list** — markdown rendering via `marked.js`, inline images
- **Input box** — text area + send button
- **Status bar** — shows tool-call activity ("Running simulation...")

Dependencies (loaded via CDN):
- `marked.js` for markdown
- Pico CSS or similar classless framework

No build step required.

### Message Types

| Event | Display |
|-------|---------|
| `text` | Rendered as markdown in chat bubble |
| `tool_call` (running) | Subtle status pill: "geocoding..." |
| `tool_call` (done) | Status pill turns green: "geocoded" |
| `image` | Inline `<img>` tag |

## Project Structure

```
dtcc-agent/
├── dtcc_agent/              # Existing MCP server (unchanged)
├── chatbot/                 # New chatbot package
│   ├── __init__.py
│   ├── app.py               # FastAPI app, WebSocket endpoint
│   ├── sessions.py          # Session manager
│   ├── config.py            # System prompt, MCP config, defaults
│   └── static/
│       └── index.html       # Chat UI
├── tests/
│   ├── test_chatbot.py      # WebSocket endpoint tests
├── pyproject.toml           # Updated with chatbot deps
└── CLAUDE.md                # System instructions
```

## New Dependencies

- `fastapi` — web framework
- `uvicorn` — ASGI server
- `claude-agent-sdk` — Claude Code as library
- `websockets` — WebSocket support

## Entry Point

```bash
# Run the chatbot
uvicorn chatbot.app:app --host 0.0.0.0 --port 8000

# Or as module
python -m chatbot.app
```

## Future: RAG Extensibility

Adding document search (RAG) requires no changes to the chatbot or dtcc-agent. Add a separate `dtcc-docs` MCP server:

```python
mcp_servers={
    "dtcc-agent": { ... },   # simulations, geometry, data (existing)
    "dtcc-docs":  { ... },   # document search via RAG (new)
}
```

The RAG server exposes tools like `search_documents(query)` and `get_document(id)`. Claude sees tools from both servers in one flat list and decides when to search documents vs run simulations. From the user's perspective it's one chatbot.

**Why a separate MCP server (not a tool in dtcc-agent):**
- Keeps dtcc-agent focused on simulations and geometry
- RAG dependencies (vector DB, embeddings) stay out of the FEniCSx conda env
- Develop, test, and deploy independently
- Swap RAG backends (Chroma, pgvector, cloud) without touching dtcc-agent

**Candidate document sources:**
- Municipal plans (detaljplaner, översiktsplaner)
- Building regulations (BBR, Boverket)
- DTCC research papers
- dtcc-core documentation
- Historical sensor/measurement data

## Constraints & Decisions

- **Prototype first:** No auth, no persistence, no rate limiting
- **dtcc-agent unchanged:** The MCP server is consumed as-is via stdio
- **Streaming essential:** Simulations take 1-5 minutes; users need real-time feedback
- **Single process:** Agent SDK spawns dtcc-agent as subprocess; one MCP server per session
- **No build tooling:** Frontend is a single HTML file loaded from disk
