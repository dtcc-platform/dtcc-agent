"""The Claude Agent SDK runtime: M2's agent loop, kept for one milestone as
M3's rollback (DTCC_AGENT_RUNTIME=sdk, #86). Needs the `sdk` extra.

Each turn spawns the bundled claude CLI, which resumes the conversation by
its SDK session id and reaches the dtcc-agent MCP server itself.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from chatbot.config import SYSTEM_PROMPT, bedrock_env, get_mcp_server_config, load_model
from chatbot.provenance import TurnRecord
from chatbot.sessions import Session

from . import artifact_frame

NAME = "sdk"
PACKAGE = f"claude-agent-sdk {importlib.metadata.version('claude-agent-sdk')}"

# Prevent "cannot launch inside another Claude Code session" when the
# chatbot is started from within a Claude Code terminal.
os.environ.pop("CLAUDECODE", None)

logger = logging.getLogger("lurkie")


def build_options(session: Session, sdk_session_id: str | None = None,
                  memory_context: str = "", turn_id: str | None = None) -> ClaudeAgentOptions:
    """Build Agent SDK options, optionally resuming a session."""
    prompt = SYSTEM_PROMPT
    if memory_context:
        prompt += f"\n\n{memory_context}"
    opts = ClaudeAgentOptions(
        system_prompt=prompt,
        mcp_servers=get_mcp_server_config(session.id, session.subject, turn_id),
        # SECURITY: the agent's tools are the dtcc-agent MCP server's, and no
        # built-in CLI tool but ToolSearch, which it loads them through.
        # Without `tools` the CLI enables all of its own (Bash, Read, Edit,
        # Write, Task...), which run in this container beside its credentials,
        # and bypassPermissions approves every call. strict_mcp_config keeps
        # out any MCP server configured elsewhere in the container.
        tools=["ToolSearch"],
        strict_mcp_config=True,
        permission_mode="bypassPermissions",
        model=load_model(),
        env=bedrock_env(),
    )
    if sdk_session_id:
        opts.resume = sdk_session_id
    return opts


async def stream_response(client: ClaudeSDKClient, ws, session_id: str,
                          turn: TurnRecord) -> tuple[str | None, str]:
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
                    if frame := artifact_frame(session_id, block.content):
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


async def answer(ws, session: Session, user_text: str, turn: TurnRecord, memory) -> str:
    """One turn on the Agent SDK. A resumed conversation that fails is tried
    once more fresh, as the same turn."""
    turn.runtime, turn.package = NAME, PACKAGE
    session_id = session.id
    sdk_session_id = session.sdk_session_id
    if sdk_session_id:
        logger.info("[%s] Resuming SDK session %s", session_id, sdk_session_id)

    # Only inject RAG context on fresh sessions — resumed sessions
    # already have conversation history in their context window.
    memory_context = "" if sdk_session_id else memory.retrieve(user_text, session_id)
    turn.memory_context = bool(memory_context)
    options = build_options(session, sdk_session_id, memory_context, turn_id=turn.turn_id)

    assistant_text = ""
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(user_text)
            logger.info("[%s] Query sent, streaming response...", session_id)
            new_sdk_session, assistant_text = await stream_response(client, ws, session_id, turn)
            if new_sdk_session:
                session.sdk_session_id = new_sdk_session

    except Exception as exc:
        logger.exception("[%s] Error during Agent SDK call", session_id)
        # If we were resuming a session, try again fresh, as the same turn
        if sdk_session_id:
            logger.info("[%s] Retrying with fresh session (previous may have hit context limit)", session_id)
            session.sdk_session_id = None
            turn.retry()
            try:
                fresh_options = build_options(session, None, memory_context, turn_id=turn.turn_id)
                async with ClaudeSDKClient(options=fresh_options) as client:
                    await client.query(user_text)
                    new_sdk_session, assistant_text = await stream_response(
                        client, ws, session_id, turn,
                    )
                    if new_sdk_session:
                        session.sdk_session_id = new_sdk_session
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
    return assistant_text
