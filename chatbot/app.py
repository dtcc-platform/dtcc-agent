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
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def index():
    """Serve the chat UI."""
    html_path = _static_dir / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h1>DTCC Chatbot</h1><p>Frontend not built yet. "
            "See chatbot/static/index.html.</p>",
            status_code=200,
        )
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
