---
type: architecture
title: "Model runtime: pydantic-ai on Bedrock"
description: How one chat turn runs. The pydantic-ai agent loop inside the chatbot process talks to Amazon Bedrock and the dtcc-agent MCP server, caches its prompt prefix, adds per-Session memory, retries a failed follow-up once, and turns refusals and rejected Bedrock keys into plain messages.
tags: [runtime, pydantic-ai, bedrock, prompt-caching, memory, chatbot]
sources:
  - id: openwiki-source-d82fbc21a9f74516f7bfd0f8
    resource: repo://chatbot/app.py
  - id: openwiki-source-778a883bcdc0a6ed0b3401f7
    resource: repo://chatbot/config.py
  - id: openwiki-source-931ea4e3e14cfe3c996abf4a
    resource: repo://chatbot/memory.py
  - id: openwiki-source-5f2b4e0b591e0b8e093c4c16
    resource: repo://chatbot/runtime/__init__.py
  - id: openwiki-source-6ed7b95a502f0a6d7126df1d
    resource: repo://chatbot/runtime/pai.py
  - id: openwiki-source-456d11852f0964a8eeba901c
    resource: repo://tests/test_bedrock_auth.py
  - id: openwiki-source-43e58ed47a1074dbb384b73d
    resource: repo://tests/test_chatbot_config.py
  - id: openwiki-source-21aa3c6e702bf53bea1fe0ae
    resource: repo://tests/test_chatbot_runtime_pai.py
  - id: openwiki-source-34f19409acce27bb03b9d119
    resource: repo://tests/test_limits.py
generated: { by: "claude-code", at: "2026-10-31T10:29:51.442Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-10-31T10:29:51.442Z
---

# Model runtime: pydantic-ai on Bedrock

A chat turn is answered by `chatbot/runtime/pai.py`, the only runtime since M4. The agent loop runs **inside the chatbot process**: pydantic-ai calls Bedrock for the model and the dtcc-agent MCP server for tools directly, with no CLI and no process per message (ADR-0003, #86). The Claude Agent SDK that M3 kept as a rollback was removed in #105.

`chatbot/runtime/__init__.py` exposes the contract the chat app calls: `load_runtime()` returns the module, and the module has `NAME`, `PACKAGE` and `async answer(ws, session, user_text, turn, memory) -> str`. `DTCC_AGENT_RUNTIME` may still be unset or `pydantic-ai`; any other value (a deployment still asking for `sdk`) stops the chatbot at import with a message saying the SDK was removed.

## Model and provider

- **Bedrock only** (#29). `model()` builds one `BedrockConverseModel` per process, on first use, because creating the Bedrock client resolves AWS credentials. The region is `AWS_REGION`, default `eu-north-1`.
- **Behind the limits gate** (#132). `model()` returns the Bedrock model wrapped in `LimitedModel` (`chatbot/limits.py`), so every model request, a turn's, its fresh retry's and the startup probe's, passes the deployment's spend and concurrency limits. With a spend limit set it also builds the Bedrock client with `AWS_MAX_ATTEMPTS=1` and refuses to start if botocore's own retries are still on. See [Turn and spend limits](../operations/turn-and-spend-limits.md). The lifespan builds the model at startup, so a bad limit setting or an unreadable spend ledger stops the server there.
- **The model ID is configuration.** `load_model()` returns `DTCC_AGENT_MODEL`, default `eu.anthropic.claude-sonnet-5-5`. Claude has no on-demand IDs in `eu-north-1`, only `eu.*` cross-region inference profiles. Changing models is this one variable.
- **Credentials:** `require_bedrock_credentials()` runs in the app lifespan. A set `AWS_BEARER_TOKEN_BEDROCK` is accepted; otherwise it asks botocore's credential chain, the one the Bedrock client uses: environment keys, a profile, or a container or instance role, which sets no variable at all (#130). It refuses to start when the chain finds nothing, or when the AWS configuration is unusable (a missing profile, a key id without its secret). It resolves credentials but sends nothing to Bedrock, so the lifespan also starts `probe()` in the background (#106): one 1-token request whose failure is logged loudly (`BEDROCK REJECTED THE SERVER'S KEY`) on 401/403 and as a warning otherwise. The probe never raises and never delays startup; `DTCC_AGENT_BEDROCK_PROBE=0` turns it off, and `tests/conftest.py` sets that so tests never call Bedrock.

## One agent, cached prompt

`agent()` is the process's single `Agent`. It owns the system prompt (`SYSTEM_PROMPT` in `chatbot/config.py`) and turns on Bedrock prompt caching for tool definitions, instructions and messages, so a request goes out as tools, then system, then messages, with a cache point after each (ADR-0006). A follow-up re-reads the stable prefix from cache.

The system prompt names the persona, gives the common `run_operation` schemas directly (so the model need not call `describe_operation` for them), asks for small geocoding radii and parallel tool calls, and, since #104, says simulation results are summary statistics: the model must not say where values are high or low unless a tool returned that. The prompt was cut to about a third in #83, and #95 measured putting the full operation catalogue in the cached prefix and did not adopt it.

## Tools

`toolset(session, turn_id)` connects to the dtcc-agent MCP server for one turn, using `get_mcp_server_config`:

- with `DTCC_MCP_URL` set, an `MCPToolset` over streamable HTTP whose headers carry `X-DTCC-Session`, `X-DTCC-Subject`, `X-DTCC-Turn` and, when `DTCC_MCP_SECRET` is set, `Authorization: Bearer <secret>`;
- otherwise a stdio server launched with the current interpreter (`DTCC_AGENT_PYTHON` overrides it) and the same identity in environment variables.

It never uses pydantic-ai's native `MCPServerTool`, which Bedrock does not support. The model sees exactly the dtcc-agent tools. How those tools run is in [MCP server and tool execution](mcp-server-and-tool-execution.md).

## A turn

`answer()` holds the Session's lock, so two turns of one Session run one after the other. Before running, it refuses the turn if the Session has already started `DTCC_AGENT_TURNS_PER_CHAT` turns (`Session.turns_started`, which counts refused and failed turns too), and it runs the turn inside `asyncio.timeout(DTCC_AGENT_TURN_SECONDS)`. Both limits are off when unset. `_run()`:

1. Retrieves **memory** only for a fresh conversation: a follow-up has its message history instead. Retrieval runs in a worker thread because it embeds the question (#94). The result is passed as a *dynamic* instruction, which places it after the system cache point, so memory never changes the cached prefix (#87).
2. Streams the agent's events with `run_stream_events`, passing the Session's message history. `_relay()` forwards each finished text part as a `text` frame, each tool call as a `tool_call` frame (and notes it on the turn's record), and each tool result that names an artifact as an `image` or `file` frame (`artifact_frame`). When the run ends it stores the full message history on the Session and notes each model name and its Bedrock usage.

`chatbot/memory.py` is a ChromaDB collection persisted under `DTCC_AGENT_MEMORY_DIR`. `store()` saves each exchange with its Session id; `retrieve()` queries only that Session's exchanges (ADR-0004) and keeps results within cosine distance 0.8.

## Failure handling

| What happened | What the user sees | Retried? |
|---|---|---|
| A follow-up fails (its context too long, say) | The answer from a fresh conversation, counted as the same turn | Once, with the history cleared; the usage of both attempts is summed |
| Bedrock's content filter ends the reply (`ContentFilterError`) | "I can't help with that request." The record marks it refused, not an error (#96) | No: a retry would only repeat it |
| Bedrock rejects the server's key (HTTP 401 or 403) | "I can't reach the model: the server's Bedrock key is invalid or has expired. Please tell the administrator." (#106) | No |
| A limit refuses it (`LimitReached`: turns per chat, or the monthly spend) | The limit's message, for example "This chat has reached its limit of 30 questions." Recorded with `error: "LimitReached"` | No |
| The turn passes its deadline | "This answer took longer than 180 seconds and was stopped." Its partial history is dropped. Recorded with `error: "TimeoutError"`. A `TimeoutError` from anything else in the turn is an ordinary error | No |
| Anything else | "Sorry, an error occurred. Check the server logs for details." | Only the follow-up retry above |

`http_status()` (`chatbot/provenance.py`) finds the status code inside exception groups, causes and implicit contexts; the failure log line and the turn's record carry it. A turn that fails before any model request records no usage. Usage is priced with genai-prices (`chatbot/prices.py`) on Bedrock's prices; see [Provenance and evaluation](../operations/provenance-and-evaluation.md).

A tool that *raises* is a failure of the turn: pydantic-ai retries the tool once and then aborts, which the user sees as "Sorry, an error occurred". That is why tools return errors as JSON instead.

## Tests

- `tests/test_chatbot_runtime_pai.py`: frames streamed per turn, history on follow-ups and cleared by a new chat, memory read only for a fresh conversation, the fresh retry and summed usage, refusals not retried, no usage before any request, turns of one Session serialised, Bedrock usage mapping, the record naming the runtime, the model seeing only dtcc-agent's tools.
- `tests/test_bedrock_auth.py`: a rejected key's message, no fresh retry on 401/403, other failures still "Sorry", `http_status` through groups, causes and contexts, the startup probe's log lines, and that tests never run the real probe.
- `tests/test_chatbot_config.py`: model and access-code configuration; the credential check with botocore isolated from the machine (a key, an access-key pair, a profile and an instance role start; nothing, a missing profile or a key id without its secret refuse); the MCP server configuration with its headers; and the statistics-only rule in the prompt.
- `tests/test_limits.py`: the turn limit, the deadline, the spend gate and the ledger; see [Turn and spend limits](../operations/turn-and-spend-limits.md).

See also [Chatbot (Lurkie)](../integrations/chatbot-lurkie.md) for the web app around this, and the [ADR map](../decisions/adr-map.md) for ADR-0003 and ADR-0006.
