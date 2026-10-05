"""The pydantic-ai runtime (ADR-0003, #86): the agent loop runs in this
process and talks to Bedrock and the dtcc-agent MCP server directly. No CLI,
no process per message.

The prompt goes out tools, then system, then messages, with a cache point
after each (ADR-0006), so a follow-up re-reads the stable prefix from cache.
The system part is the base prompt and the operation catalogue (#87), both
static and before the instructions cache point; memory context is dynamic and
comes after it.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
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
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import InstructionPart, ModelResponse, ToolReturnPart
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.providers.bedrock import BedrockProvider
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from chatbot.config import DEFAULT_REGION, SYSTEM_PROMPT, get_mcp_server_config, load_model
from chatbot.prices import price
from chatbot.provenance import TurnRecord, prompt_version
from chatbot.sessions import Session

from . import artifact_frame

NAME = "pydantic-ai"
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
    """The process's one agent. It owns the cache settings; each run is given
    the model and the instructions."""
    return Agent(
        model_settings=BedrockModelSettings(
            bedrock_cache_tool_definitions=True,
            bedrock_cache_instructions=True,
            bedrock_cache_messages=True,
        ),
    )


# The catalogue variants #87 measures; one is kept and the other deleted.
CATALOGUE_VARIANTS = ("summary", "full")
CATALOGUE_LINES = {
    "summary": "The catalogue below lists every operation. Call `describe_operation` for an "
               "operation's parameters before running it. Call `list_operations` only to search.",
    "full": "The catalogue below gives every operation's full schema. Do not call "
            "`list_operations` or `describe_operation`.",
}
# Sent instead of a catalogue when the fetch failed.
DISCOVERY_LINE = ("Use `list_operations` and `describe_operation` to find an operation and "
                  "its parameters before `run_operation`.")


def catalogue_variant() -> str:
    """DTCC_AGENT_CATALOGUE; unset or empty means summary."""
    name = os.getenv("DTCC_AGENT_CATALOGUE") or "summary"
    if name not in CATALOGUE_VARIANTS:
        raise SystemExit(f"DTCC_AGENT_CATALOGUE must be one of {', '.join(CATALOGUE_VARIANTS)}, "
                         f"got {name!r}.")
    return name


class _Memo:
    """The catalogue text, fetched once per process. It only changes with a
    dtcc-core upgrade, which recreates the containers (#29)."""

    def __init__(self) -> None:
        self.text: str | None = None
        self.lock = asyncio.Lock()


_catalogue = _Memo()


async def _call(tools: AbstractToolset, name: str, args: dict[str, Any]) -> str:
    """One MCP tool's JSON text; its error result raises."""
    text = await tools.direct_call_tool(name, args)
    result = json.loads(text)
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"{name} failed: {result['error']}")
    return text


async def fetch_catalogue(tools: AbstractToolset, variant: str) -> str:
    """The catalogue text through the turn's MCP toolset: `list_operations`,
    or every operation's `describe_operation`."""
    async with tools:
        summary = await _call(tools, "list_operations", {})
        if variant == "summary":
            return summary
        return "\n\n".join([await _call(tools, "describe_operation", {"name": op["name"]})
                             for op in json.loads(summary)])


async def catalogue(tools: AbstractToolset, variant: str) -> str | None:
    """The memoised catalogue, fetched on first use. Concurrent first turns
    fetch once; a failed fetch logs, returns None, and the next turn tries again."""
    if _catalogue.text is None:
        async with _catalogue.lock:
            if _catalogue.text is None:
                try:
                    _catalogue.text = await fetch_catalogue(tools, variant)
                except Exception as exc:
                    logger.warning("Catalogue fetch failed (%s: %s); this turn runs without it",
                                   type(exc).__name__, exc)
    return _catalogue.text


def static_instructions(variant: str, text: str | None) -> list[str]:
    """What goes before the instructions cache point: the base prompt, then
    the catalogue with its variant's line, or the discovery line without one."""
    return [SYSTEM_PROMPT, f"{CATALOGUE_LINES[variant]}\n\n{text}" if text else DISCOVERY_LINE]


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
            except Exception as exc:
                if not session.history:
                    raise
                logger.warning("[%s] Turn failed with history (%s); retrying fresh",
                               session.id, type(exc).__name__)
                session.history = []
                turn.retry()
                return await _run(ws, session, user_text, turn, memory, usage, models)
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
    memory_context = "" if session.history else memory.retrieve(user_text, session.id)
    turn.memory_context = bool(memory_context)
    tools = toolset(session, turn.turn_id)
    variant = catalogue_variant()
    text = await catalogue(tools, variant)
    static = static_instructions(variant, text)
    turn.prompt_version = prompt_version("\n\n".join(static))
    turn.catalogue_in_prompt, turn.catalogue_variant = text is not None, variant
    # A plain string is a static instruction; memory is marked dynamic so
    # it stays after the cache point.
    texts: list[str] = []
    async with agent().run_stream_events(
        user_text,
        model=model(),
        message_history=session.history or None,
        toolsets=[tools],
        instructions=[*static, *([InstructionPart(memory_context, dynamic=True)]
                                 if memory_context else [])],
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
            if isinstance(message, ModelResponse) and message.model_name:
                turn.saw_model(message.model_name)
                if message.model_name not in models:
                    models.append(message.model_name)
