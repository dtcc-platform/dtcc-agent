"""The agent runtime that answers a chat turn (#86).

`pydantic-ai` (`pai.py`) runs the agent loop in this process on Bedrock. It is
the only runtime: the Claude Agent SDK, M3's rollback, was removed in M4.

The runtime module exposes NAME, PACKAGE and

    async def answer(ws, session, user_text, turn, memory) -> str

which streams the turn's frames to `ws`, notes provenance on `turn`, keeps the
session's conversation state, and returns the assistant's text.
"""

from __future__ import annotations

import json
import os
from types import ModuleType

from dtcc_agent import artifacts


def load_runtime() -> ModuleType:
    """The pydantic-ai runtime. DTCC_AGENT_RUNTIME, which chose between it and
    the Agent SDK, may still be set to pydantic-ai or left empty; anything else
    exits, so a deployment still asking for the SDK fails at start."""
    name = os.getenv("DTCC_AGENT_RUNTIME") or "pydantic-ai"
    if name != "pydantic-ai":
        raise SystemExit(
            f"DTCC_AGENT_RUNTIME={name!r}: the only runtime is pydantic-ai; the Agent SDK "
            "rollback was removed in M4. Unset DTCC_AGENT_RUNTIME."
        )
    from . import pai

    return pai


def artifact_frame(session_id: str, content: object) -> dict | None:
    """The frame that shows a tool result's artifact on the page, if it has one.

    `content` is the tool's result as a runtime hands it over: the JSON text
    the tool returned, a list of text parts (the SDK), or already parsed
    (pydantic-ai). An MCP server's structured output wraps a tool's returned
    string as {"result": "<that string>"}."""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    try:
        result = json.loads(content) if isinstance(content, str) else content
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
