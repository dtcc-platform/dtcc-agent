"""The pydantic-ai runtime (ADR-0003, #86): the agent loop runs in this
process and talks to Bedrock and the dtcc-agent MCP server directly. No CLI,
no process per message.

The prompt goes out tools, then system, then messages, with a cache point
after each (ADR-0006), so a follow-up re-reads the stable prefix from cache.
Memory context is a dynamic instruction, after the system cache point, so a
conversation's memory never changes the cached prefix (#87).
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import os
import time
from functools import cache
from typing import Any

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartEndEvent,
    TextPart,
)
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import InstructionPart, ModelResponse, ToolReturnPart
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.providers.bedrock import BedrockProvider
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from chatbot.config import DEFAULT_REGION, SYSTEM_PROMPT, get_mcp_server_config, load_model
from chatbot.prices import price
from chatbot.provenance import TurnRecord
from chatbot.sessions import Session

from . import artifact_frame

NAME = "pydantic-ai"
# Said when the model's provider ends a reply with its content filter (#96).
REFUSAL = "I can't help with that request."
PACKAGE = f"pydantic-ai-slim {importlib.metadata.version('pydantic-ai-slim')}"

logger = logging.getLogger("lurkie")


@cache
def model() -> BedrockConverseModel:
    """The configured model on Bedrock. Built on first use: creating the
    Bedrock client resolves AWS credentials. Thinking is left at the model's
    default, as the SDK left it, so the M3 gate compares runtimes and not
    thinking budgets."""
    return BedrockConverseModel(
        load_model(),
        provider=BedrockProvider(region_name=os.getenv("AWS_REGION") or DEFAULT_REGION),
    )


@cache
def agent() -> Agent:
    """The process's one agent. It owns the prompt and the cache settings;
    each run is given the model."""
    return Agent(
        instructions=SYSTEM_PROMPT,
        model_settings=BedrockModelSettings(
            bedrock_cache_tool_definitions=True,
            bedrock_cache_instructions=True,
            bedrock_cache_messages=True,
        ),
    )


def toolset(session: Session, turn_id: str) -> AbstractToolset:
    """The dtcc-agent MCP server, for one turn: its headers carry the turn.
    The same configuration the SDK runtime hands its CLI. Never pydantic-ai's
    native MCPServerTool: Bedrock does not support it (ADR-0003)."""
    server = get_mcp_server_config(session.id, session.subject, turn_id)["dtcc-agent"]
    if server["type"] == "http":
        return MCPToolset(server["url"], headers=server["headers"])
    from fastmcp.client.transports import StdioTransport

    return MCPToolset(StdioTransport(command=server["command"], args=server["args"],
                                     env={**os.environ, **server["env"]}))


async def answer(ws, session: Session, user_text: str, turn: TurnRecord, memory) -> str:
    """One turn. A failure partway into a conversation (its context too long,
    say) is retried once from a fresh conversation, as the same turn; the
    usage of every attempt is counted."""
    turn.runtime, turn.package = NAME, PACKAGE
    usage = RunUsage()
    started = time.monotonic()
    models: list[str] = []
    async with session.lock:
        try:
            try:
                return await _run(ws, session, user_text, turn, memory, usage, models)
            except ContentFilterError:
                raise  # a refusal, which a fresh retry would only repeat
            except Exception as exc:
                if not session.history:
                    raise
                logger.warning("[%s] Turn failed with history (%s); retrying fresh",
                               session.id, type(exc).__name__)
                session.history = []
                turn.retry()
                return await _run(ws, session, user_text, turn, memory, usage, models)
        except ContentFilterError:
            # Bedrock ended the reply with its content filter: the model
            # declined. Say so plainly; it isn't a server fault (#96).
            logger.info("[%s] The model declined the request (content filter)", session.id)
            turn.refuse()
            await ws.send_json({"type": "text", "content": REFUSAL})
            return ""
        except Exception as exc:
            logger.exception("[%s] The agent failed", session.id)
            turn.failed(exc)
            await ws.send_json({"type": "text",
                                "content": "Sorry, an error occurred. Check the server logs for details."})
            return ""
        finally:
            # A turn that failed before any model request used nothing: its
            # record keeps usage null, as the SDK runtime's does.
            if usage.requests:
                cost, source = price(usage, load_model())
                turn.saw_run(usage, elapsed_ms=round((time.monotonic() - started) * 1000),
                             cost=cost, cost_source=source, models=models)


async def _run(ws, session: Session, user_text: str, turn: TurnRecord, memory,
               usage: RunUsage, models: list[str]) -> str:
    # Memory only starts a conversation; a follow-up has the history instead.
    # Off the event loop: retrieval embeds the question (#94).
    memory_context = "" if session.history else await asyncio.to_thread(
        memory.retrieve, user_text, session.id)
    turn.memory_context = bool(memory_context)
    texts: list[str] = []
    async with agent().run_stream_events(
        user_text,
        model=model(),
        message_history=session.history or None,
        toolsets=[toolset(session, turn.turn_id)],
        # A plain string would count as static and sit inside the cached
        # prefix; marked dynamic, memory goes after the cache point.
        instructions=InstructionPart(memory_context, dynamic=True) if memory_context else None,
        usage=usage,
    ) as events:
        async for event in events:
            await _relay(ws, session, turn, event, texts, models)
    return "".join(texts)


async def _relay(ws, session: Session, turn: TurnRecord, event: Any,
                 texts: list[str], models: list[str]) -> None:
    """Pass one agent event on to the page and the turn's record."""
    if isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
        texts.append(event.part.content)
        await ws.send_json({"type": "text", "content": event.part.content})
    elif isinstance(event, FunctionToolCallEvent):
        logger.info("[%s]   Tool call: %s", session.id, event.part.tool_name)
        turn.saw_tool(event.part.tool_name)
        await ws.send_json({"type": "tool_call", "name": event.part.tool_name, "status": "running"})
    elif isinstance(event, FunctionToolResultEvent) and isinstance(event.part, ToolReturnPart):
        if frame := artifact_frame(session.id, event.part.content):
            await ws.send_json(frame)
    elif isinstance(event, AgentRunResultEvent):
        new = event.result.new_messages()
        session.history = event.result.all_messages()
        for message in new:
            if isinstance(message, ModelResponse):
                # Each request's Bedrock usage, in Bedrock's terms: pydantic-ai
                # counts cache tokens inside input_tokens (#86, #87).
                u = message.usage
                logger.debug("[%s]   Usage: inputTokens=%d cacheReadInputTokens=%d "
                             "cacheWriteInputTokens=%d outputTokens=%d", session.id,
                             u.input_tokens - u.cache_read_tokens - u.cache_write_tokens,
                             u.cache_read_tokens, u.cache_write_tokens, u.output_tokens)
            if isinstance(message, ModelResponse) and message.model_name:
                turn.saw_model(message.model_name)
                if message.model_name not in models:
                    models.append(message.model_name)
