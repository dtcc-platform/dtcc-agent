# chatbot/app.py
"""FastAPI application with WebSocket chat endpoint powered by Claude Agent SDK."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

# Prevent "cannot launch inside another Claude Code session" error
# when the chatbot is started from within a Claude Code terminal.
os.environ.pop("CLAUDECODE", None)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    UserMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
)

from chatbot.config import SYSTEM_PROMPT, get_mcp_server_config, DEFAULT_HOST, DEFAULT_PORT
from chatbot.memory import ConversationMemory
from chatbot.sessions import SessionManager

# --- Logging setup: file + console ---
_log_dir = Path("/tmp/dtcc_lurkie_logs")
_log_dir.mkdir(exist_ok=True)
_log_file = _log_dir / f"lurkie-{datetime.now():%Y%m%d-%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("lurkie")
logger.setLevel(logging.DEBUG)  # debug for our code only
logger.info("Log file: %s", _log_file)

app = FastAPI(title="DTCC Lurkie")
sessions = SessionManager()
memory = ConversationMemory()

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
            "<h1>DTCC Lurkie</h1><p>Frontend not built yet. "
            "See chatbot/static/index.html.</p>",
            status_code=200,
        )
    return HTMLResponse(html_path.read_text())


def _build_options(
    sdk_session_id: str | None = None,
    memory_context: str = "",
) -> ClaudeAgentOptions:
    """Build Agent SDK options, optionally resuming a session."""
    prompt = SYSTEM_PROMPT
    if memory_context:
        prompt += f"\n\n{memory_context}"
    opts = ClaudeAgentOptions(
        system_prompt=prompt,
        mcp_servers=get_mcp_server_config(),
        # SECURITY: bypassPermissions is used for the prototype since the
        # agent only has access to dtcc-agent MCP tools (no shell/filesystem).
        # For production, switch to an explicit allowlist.
        permission_mode="bypassPermissions",
        model="claude-sonnet-4-5",
    )
    if sdk_session_id:
        opts.resume = sdk_session_id
    return opts


async def _stream_response(
    client: ClaudeSDKClient,
    ws: WebSocket,
    session_id: str,
) -> tuple[str | None, str]:
    """Stream Agent SDK responses over WebSocket.

    Returns (sdk_session_id, collected_assistant_text).
    """
    sdk_session_id = None
    assistant_text_parts: list[str] = []

    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            logger.info("[%s] AssistantMessage (model=%s, stop=%s)",
                        session_id, getattr(msg, 'model', '?'),
                        getattr(msg, 'stop_reason', '?'))
            for block in msg.content:
                if isinstance(block, TextBlock):
                    preview = block.text[:120].replace('\n', ' ')
                    logger.info("[%s]   TextBlock: %s%s",
                                session_id, preview,
                                "..." if len(block.text) > 120 else "")
                    assistant_text_parts.append(block.text)
                    await ws.send_json({"type": "text", "content": block.text})

                elif isinstance(block, ToolUseBlock):
                    logger.info("[%s]   ToolUseBlock: %s (id=%s) input=%s",
                                session_id, block.name, block.id,
                                json.dumps(block.input, default=str)[:200])
                    await ws.send_json({
                        "type": "tool_call",
                        "name": block.name,
                        "status": "running",
                    })

                elif isinstance(block, ThinkingBlock):
                    preview = block.thinking[:100].replace('\n', ' ')
                    logger.debug("[%s]   ThinkingBlock: %s...", session_id, preview)

                else:
                    logger.debug("[%s]   Block type: %s", session_id, type(block).__name__)

        elif isinstance(msg, UserMessage):
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    content_str = str(block.content)[:300] if block.content else "(empty)"
                    logger.info("[%s]   ToolResult [%s]: %s%s",
                                session_id, block.tool_use_id,
                                content_str,
                                "..." if len(str(block.content)) > 300 else "")
                else:
                    logger.debug("[%s]   UserBlock: %s", session_id, type(block).__name__)

        elif isinstance(msg, SystemMessage):
            logger.info("[%s] SystemMessage [%s]: %s",
                        session_id, getattr(msg, 'subtype', '?'),
                        str(getattr(msg, 'data', ''))[:200])

        elif isinstance(msg, ResultMessage):
            sdk_session_id = msg.session_id
            logger.info("[%s] ResultMessage: turns=%s, cost=$%s, duration=%sms, session=%s",
                        session_id,
                        getattr(msg, 'num_turns', '?'),
                        getattr(msg, 'total_cost_usd', '?'),
                        getattr(msg, 'duration_ms', '?'),
                        sdk_session_id)
            break

        else:
            logger.debug("[%s] Unknown message type: %s", session_id, type(msg).__name__)

    return sdk_session_id, "".join(assistant_text_parts)


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

    logger.info("[%s] New WebSocket connection", session_id)

    # Send session ID to client
    await ws.send_json({"type": "session", "session_id": session_id})

    try:
        while True:
            data = await ws.receive_json()

            # Handle "new chat" reset from client
            if data.get("type") == "new_chat":
                logger.info("[%s] Client requested new chat, clearing SDK session", session_id)
                sessions.set_sdk_session(session_id, None)
                continue

            user_text = data.get("content", "").strip()
            if not user_text:
                continue
            if len(user_text) > 10_000:
                await ws.send_json({
                    "type": "text",
                    "content": f"Message too long ({len(user_text)} chars). Please keep it under 10,000.",
                })
                await ws.send_json({"type": "done"})
                continue

            logger.info("[%s] User: %s", session_id, user_text[:200])
            await ws.send_json({"type": "status", "content": "thinking"})

            sdk_session_id = sessions.get_sdk_session(session_id)
            if sdk_session_id:
                logger.info("[%s] Resuming SDK session %s", session_id, sdk_session_id)

            # Only inject RAG context on fresh sessions — resumed sessions
            # already have conversation history in their context window.
            memory_context = "" if sdk_session_id else memory.retrieve(user_text)
            options = _build_options(sdk_session_id, memory_context)

            assistant_text = ""
            try:
                async with ClaudeSDKClient(options=options) as client:
                    await client.query(user_text)
                    logger.info("[%s] Query sent, streaming response...", session_id)
                    new_sdk_session, assistant_text = await _stream_response(
                        client, ws, session_id,
                    )

                    if new_sdk_session:
                        sessions.set_sdk_session(session_id, new_sdk_session)

            except Exception:
                logger.exception("[%s] Error during Agent SDK call", session_id)
                # If we were resuming a session, try again fresh
                if sdk_session_id:
                    logger.info("[%s] Retrying with fresh session (previous may have hit context limit)", session_id)
                    sessions.set_sdk_session(session_id, None)
                    try:
                        fresh_options = _build_options(None, memory_context)
                        async with ClaudeSDKClient(options=fresh_options) as client:
                            await client.query(user_text)
                            new_sdk_session, assistant_text = await _stream_response(
                                client, ws, session_id,
                            )
                            if new_sdk_session:
                                sessions.set_sdk_session(session_id, new_sdk_session)
                    except Exception:
                        logger.exception("[%s] Fresh session also failed", session_id)
                        await ws.send_json({
                            "type": "text",
                            "content": "Sorry, an error occurred. Please try starting a new chat.",
                        })
                else:
                    await ws.send_json({
                        "type": "text",
                        "content": "Sorry, an error occurred. Check the server logs for details.",
                    })

            # Store the exchange in long-term memory
            if assistant_text:
                memory.store(session_id, user_text, assistant_text)

            await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("[%s] Client disconnected", session_id)


def main():
    """Run the chatbot server."""
    import uvicorn

    logger.info("Starting DTCC Lurkie on %s:%s", DEFAULT_HOST, DEFAULT_PORT)
    logger.info("Log file: %s", _log_file)
    logger.info("Tail logs with: tail -f %s", _log_file)
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
