"""The agent runtime that answers a chat turn (#86).

`pydantic-ai` (the default, `pai.py`) runs the agent loop in this process on
Bedrock. `sdk` (`sdk.py`) is the Claude Agent SDK, kept for one milestone as
M3's rollback; it needs the `sdk` extra, which bundles the claude CLI.

Each runtime module exposes NAME, PACKAGE and

    async def answer(ws, session, user_text, turn, memory) -> str

which streams the turn's frames to `ws`, notes provenance on `turn`, keeps the
session's conversation state, and returns the assistant's text.
"""

from __future__ import annotations

import json
import os
from types import ModuleType

from dtcc_agent import artifacts

RUNTIMES = ("pydantic-ai", "sdk")


def load_runtime() -> ModuleType:
    """The runtime DTCC_AGENT_RUNTIME names; unset or empty means pydantic-ai.
    Exits on anything else, and on `sdk` without its extra installed."""
    name = os.getenv("DTCC_AGENT_RUNTIME") or "pydantic-ai"
    if name == "pydantic-ai":
        from . import pai

        return pai
    if name == "sdk":
        try:
            from . import sdk
        except ImportError as exc:
            raise SystemExit(
                "DTCC_AGENT_RUNTIME=sdk needs the 'sdk' extra (the Claude Agent SDK): "
                "pip install 'dtcc-agent[chatbot,sdk]'."
            ) from exc
        return sdk
    raise SystemExit(f"DTCC_AGENT_RUNTIME must be one of {', '.join(RUNTIMES)}, got {name!r}.")


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
