# chatbot/app.py
"""FastAPI application with the WebSocket chat endpoint. A turn is answered by
the runtime DTCC_AGENT_RUNTIME names (chatbot/runtime, #86)."""

from __future__ import annotations

import asyncio
import hmac
import itertools
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from chatbot.config import (
    load_access_code, DEFAULT_HOST, DEFAULT_PORT, load_model, require_bedrock_credentials,
)
from chatbot.memory import ConversationMemory
from chatbot.provenance import TurnRecord
from chatbot.runtime import artifact_frame, load_runtime
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

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Checked at server start, not import: every chat turn needs Bedrock (#85).
    require_bedrock_credentials()
    logger.info("Model: %s on Bedrock, runtime %s", load_model(), runtime.NAME)
    yield


app = FastAPI(title="DTCC Lurkie", lifespan=_lifespan)
# Exits here on an unknown runtime, or `sdk` without its extra.
runtime = load_runtime()
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


async def _answer(ws: WebSocket, session_id: str, user_text: str) -> None:
    """Run one turn: the runtime asks the agent and streams its answer, then
    the turn's provenance record and done go out. Exactly one answers.jsonl
    line per turn, whatever happens: a fresh retry adds to it, a crash still
    writes it."""
    session = sessions.get(session_id)
    turn = TurnRecord(refs.new(refs.TURN), session_id, session.subject, memory_context=False)
    try:
        logger.info("[%s] %s User: %s", session_id, turn.turn_id, user_text[:200])
        await ws.send_json({"type": "status", "content": "thinking"})
        assistant_text = await runtime.answer(ws, session, user_text, turn, memory)
        # Store the exchange in long-term memory, off the event loop (#94)
        if assistant_text:
            await asyncio.to_thread(memory.store, session_id, user_text, assistant_text)
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
                logger.info("[%s] Client requested new chat, clearing the conversation", session_id)
                sessions.get(session_id).reset_conversation()
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
