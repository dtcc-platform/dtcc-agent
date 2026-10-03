# chatbot/app.py
"""FastAPI application with WebSocket chat endpoint powered by Claude Agent SDK."""

from __future__ import annotations

import hmac
import itertools
import json
import logging
import os
from datetime import datetime
from pathlib import Path

# Prevent "cannot launch inside another Claude Code session" error
# when the chatbot is started from within a Claude Code terminal.
os.environ.pop("CLAUDECODE", None)

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
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

from chatbot.config import (
    SYSTEM_PROMPT, get_mcp_server_config, load_access_code, DEFAULT_HOST, DEFAULT_PORT,
)
from chatbot.memory import ConversationMemory
from chatbot.provenance import TurnRecord
from chatbot.sessions import SessionManager
from dtcc_agent import artifacts, provenance, refs

# --- Logging setup: file + console ---
_log_dir = Path(os.getenv("DTCC_AGENT_LOG_DIR", "/tmp/dtcc_lurkie_logs"))
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
# Opening a chat needs this code; None when admission is off (T14, #72).
ACCESS_CODE = load_access_code()
_chats_opened = itertools.count(1)

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


@app.get("/artifacts/{session_id}/{name}")
async def artifact(session_id: str, name: str):
    """A file a tool produced, served only while its Session is live.

    The URL is the credential until T14 brings users: the Session id plus the
    artifact's random token."""
    path = artifacts.find(session_id, name) if sessions.get(session_id) else None
    if path is None:
        raise HTTPException(status_code=404)
    headers = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"}
    if path.suffix in artifacts.IMAGE_SUFFIXES:
        return FileResponse(path, media_type="image/png", headers=headers)
    return FileResponse(path, filename=artifacts.download_name(name),
                        media_type="application/octet-stream", headers=headers)


def _artifact_frame(session_id: str, content: object) -> dict | None:
    """The frame that shows a tool result's artifact on the page, if it has one."""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    try:
        result = json.loads(content) if isinstance(content, str) else None
        # FastMCP's structured output wraps a tool's returned string as
        # {"result": "<that string>"}, and the CLI passes that form on.
        if isinstance(result, dict) and isinstance(result.get("result"), str):
            result = json.loads(result["result"])
    except json.JSONDecodeError:
        return None
    found = result.get("artifact") if isinstance(result, dict) else None
    if not isinstance(found, dict) or not isinstance(found.get("name"), str):
        return None
    if artifacts.find(session_id, found["name"]) is None:
        return None
    url = f"/artifacts/{session_id}/{found['name']}"
    if found.get("kind") == "image":
        return {"type": "image", "url": url}
    return {"type": "file", "url": url, "name": artifacts.download_name(found["name"])}


@app.get("/admission")
async def admission():
    """Whether the page must ask for an access code before opening a chat."""
    return {"required": ACCESS_CODE is not None}


def _admitted(code: object) -> bool:
    if ACCESS_CODE is None:
        return True
    return isinstance(code, str) and hmac.compare_digest(code.encode(), ACCESS_CODE.encode())


@app.get("/health")
async def health():
    """Health endpoint for Docker Compose and local checks."""
    return {"status": "ok", "service": "dtcc-agent", "component": "lurkie"}


def _subject(session_id: str) -> str:
    session = sessions.get(session_id)
    return session.subject if session else "anonymous"


def _build_options(
    session_id: str,
    sdk_session_id: str | None = None,
    memory_context: str = "",
    turn_id: str | None = None,
) -> ClaudeAgentOptions:
    """Build Agent SDK options, optionally resuming a session."""
    prompt = SYSTEM_PROMPT
    if memory_context:
        prompt += f"\n\n{memory_context}"
    opts = ClaudeAgentOptions(
        system_prompt=prompt,
        mcp_servers=get_mcp_server_config(session_id, _subject(session_id), turn_id),
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
    turn: TurnRecord,
) -> tuple[str | None, str]:
    """Stream Agent SDK responses over WebSocket, noting on `turn` what the
    agent reports for provenance.

    Returns (sdk_session_id, collected_assistant_text).
    """
    sdk_session_id = None
    assistant_text_parts: list[str] = []

    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            logger.info("[%s] AssistantMessage (model=%s, stop=%s)",
                        session_id, getattr(msg, 'model', '?'),
                        getattr(msg, 'stop_reason', '?'))
            turn.saw_model(getattr(msg, "model", None))
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
                    turn.saw_tool(block.name)
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
                    if frame := _artifact_frame(session_id, block.content):
                        await ws.send_json(frame)
                else:
                    logger.debug("[%s]   UserBlock: %s", session_id, type(block).__name__)

        elif isinstance(msg, SystemMessage):
            logger.info("[%s] SystemMessage [%s]: %s",
                        session_id, getattr(msg, 'subtype', '?'),
                        str(getattr(msg, 'data', ''))[:200])

        elif isinstance(msg, ResultMessage):
            sdk_session_id = msg.session_id
            turn.saw_result(msg)
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


async def _answer(ws: WebSocket, session_id: str, user_text: str) -> None:
    """Run one turn: ask the agent and stream its answer, then send the
    turn's provenance record and done. Exactly one answers.jsonl line per
    turn, whatever happens: a fresh retry adds to it, a crash still writes it."""
    turn = TurnRecord(refs.new(refs.TURN), session_id, _subject(session_id), memory_context=False)
    try:
        logger.info("[%s] %s User: %s", session_id, turn.turn_id, user_text[:200])
        await ws.send_json({"type": "status", "content": "thinking"})

        sdk_session_id = sessions.get_sdk_session(session_id)
        if sdk_session_id:
            logger.info("[%s] Resuming SDK session %s", session_id, sdk_session_id)

        # Only inject RAG context on fresh sessions — resumed sessions
        # already have conversation history in their context window.
        memory_context = "" if sdk_session_id else memory.retrieve(user_text, session_id)
        turn.memory_context = bool(memory_context)
        options = _build_options(session_id, sdk_session_id, memory_context, turn_id=turn.turn_id)

        assistant_text = ""
        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(user_text)
                logger.info("[%s] Query sent, streaming response...", session_id)
                new_sdk_session, assistant_text = await _stream_response(
                    client, ws, session_id, turn,
                )

                if new_sdk_session:
                    sessions.set_sdk_session(session_id, new_sdk_session)

        except Exception as exc:
            logger.exception("[%s] Error during Agent SDK call", session_id)
            # If we were resuming a session, try again fresh, as the same turn
            if sdk_session_id:
                logger.info("[%s] Retrying with fresh session (previous may have hit context limit)", session_id)
                sessions.set_sdk_session(session_id, None)
                turn.retry()
                try:
                    fresh_options = _build_options(session_id, None, memory_context,
                                                   turn_id=turn.turn_id)
                    async with ClaudeSDKClient(options=fresh_options) as client:
                        await client.query(user_text)
                        new_sdk_session, assistant_text = await _stream_response(
                            client, ws, session_id, turn,
                        )
                        if new_sdk_session:
                            sessions.set_sdk_session(session_id, new_sdk_session)
                except Exception as fresh_exc:
                    logger.exception("[%s] Fresh session also failed", session_id)
                    turn.failed(fresh_exc)
                    await ws.send_json({
                        "type": "text",
                        "content": "Sorry, an error occurred. Please try starting a new chat.",
                    })
            else:
                turn.failed(exc)
                await ws.send_json({
                    "type": "text",
                    "content": "Sorry, an error occurred. Check the server logs for details.",
                })

        # Store the exchange in long-term memory
        if assistant_text:
            memory.store(session_id, user_text, assistant_text)
    except BaseException as exc:  # the socket closed or the task was cancelled mid-turn
        if turn.error is None:
            turn.failed(exc)
        _record_answer(turn)
        raise

    record = _record_answer(turn)
    await ws.send_json({"type": "provenance", **record})
    await ws.send_json({"type": "done"})


def _record_answer(turn: TurnRecord) -> dict:
    """Append the turn's answers.jsonl line; a failed write only warns."""
    record = turn.record()
    provenance.append(_log_dir, provenance.ANSWERS, record)
    return record


@app.websocket("/chat")
async def chat(ws: WebSocket):
    """WebSocket endpoint for chat conversations."""
    await ws.accept()

    # Read initial message to get or create session
    try:
        init = await ws.receive_json()
    except (WebSocketDisconnect, json.JSONDecodeError):
        return

    # A live session resumes on its id alone; opening a new one needs the code.
    session_id = init.get("session_id")
    if not session_id or not sessions.touch(session_id):
        if not _admitted(init.get("access_code")):
            logger.warning("Refused a new chat from %s: access code missing or wrong",
                           ws.client.host if ws.client else "?")
            await ws.send_json({"type": "error", "code": "admission_required"})
            await ws.close(code=4401, reason="access code required")
            return
        session_id = sessions.create()
        if ACCESS_CODE is None and next(_chats_opened) % 100 == 0:
            logger.warning("Admission is off (DTCC_AGENT_ACCESS_CODE unset): 100 more chats opened.")

    logger.info("[%s] New WebSocket connection", session_id)

    # Send session ID to client
    await ws.send_json({"type": "session", "session_id": session_id})

    try:
        while True:
            data = await ws.receive_json()
            if not sessions.touch(session_id):
                # Idle too long (U7): the page starts a new session.
                logger.info("[%s] Session expired, closing", session_id)
                await ws.close(code=4408, reason="session expired")
                return

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

            with sessions.turn(session_id):
                await _answer(ws, session_id, user_text)

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
