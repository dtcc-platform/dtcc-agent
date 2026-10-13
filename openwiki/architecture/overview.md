---
type: architecture
title: Architecture overview
description: End-to-end picture of dtcc-agent, from the Lurkie web chatbot through the MCP server and generic dispatch to dtcc-core and dtcc-sim, with a module ownership map and the two deployment modes.
tags: [architecture, overview, mcp, chatbot, dtcc-core]
sources:
  - id: openwiki-source-6ed7b95a502f0a6d7126df1d
    resource: repo://chatbot/runtime/pai.py
  - id: openwiki-source-b79fbbd921df689b4bbdc82f
    resource: repo://docker-compose.yml
  - id: openwiki-source-322ab22151ae73c933ac2f97
    resource: repo://docs/adr/0001-agent-is-a-conversational-front-door.md
  - id: openwiki-source-7cb0fe42631b753a02cd6ba2
    resource: repo://dtcc_agent/__init__.py
  - id: openwiki-source-32c33c58f635fb0708a0e8c6
    resource: repo://dtcc_agent/runner.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
generated: { by: "claude-code", at: "2026-10-13T19:36:58.526Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-10-13T19:36:58.526Z
---

# Architecture overview

dtcc-agent is the conversational front door to the DTCC platform (ADR-0001, accepted 2026-09-15). A person asks a question in their own words. An LLM turns it into a sequence of platform Operations, and the answer comes back as text, rendered images and downloadable files.

The repo ships two Python packages:

- **`dtcc_agent`**: an MCP server exposing dtcc-core and the simulations as tools. It can be used on its own from Claude Desktop, Claude Code or any MCP client.
- **`chatbot`** ("Lurkie"): a FastAPI web app that runs the agent loop with pydantic-ai on Amazon Bedrock against that MCP server, and streams the conversation over a WebSocket.

```
browser (chatbot/static/index.html)
   │  WebSocket /chat (access code to open a chat)
   ▼
chatbot/app.py ── runtime/pai.py: pydantic-ai agent ── Amazon Bedrock (model from DTCC_AGENT_MODEL)
   │  MCP: streamable-http + X-DTCC-Session/Subject/Turn + bearer secret (or stdio child)
   │                         ▲ GET /artifacts/<session>/<name> (files tools wrote)
   ▼
dtcc_agent/server.py   @tool functions, worker pool, per-Session state, provenance
   ├── runtime.py       process-wide setup: Engine config check, catalogue build, worker limits
   ├── registry.py      catalogue from the pinned Core, plus dtcc-sim datasets
   ├── dispatcher.py    run_operation: resolve refs → call → store → summarise
   ├── object_store.py  per-Session stores under one shared memory budget
   ├── artifacts.py     per-Session file folders; the only files tools read/write
   ├── disk_cache.py    persistent cache for the two dataset downloads (crop.py: exact sub-areas)
   ├── serializers.py   type-specific summaries (never raw arrays)
   ├── runner.py        simulations: the DTCC Engine, the dtcc-sim mini-service, or in process
   ├── engine.py        DTCC Engine client: submit, poll, download .dtccpkg
   ├── analysis.py      field statistics for simulation results
   ├── geocode.py       place name → EPSG:3006 bounds
   └── renderer.py      PNG with matplotlib (Agg: no GL, no display)
   ▼
dtcc-core (pinned commit, in process) · DTCC Engine or dtcc-sim (simulations, over HTTP)
```

## Module ownership

| Concern | Owner | Page |
|---|---|---|
| The agent loop, model, prompt, memory, failures | `chatbot/runtime/pai.py`, `chatbot/config.py`, `chatbot/memory.py` | [Model runtime](model-runtime.md) |
| Tool registration, concurrency, transports, startup | `server.py`, `runtime.py` | [MCP server and tool execution](mcp-server-and-tool-execution.md) |
| Session isolation and admission | `server.py` (`_Session`), `chatbot/sessions.py`, `chatbot/app.py` | [Sessions and isolation](sessions-and-isolation.md) |
| What operations exist | `registry.py` | [Operation catalogue](../concepts/operation-catalogue.md) |
| Running an operation and holding its result | `dispatcher.py`, `object_store.py`, `serializers.py` | [Dispatch and the object store](../concepts/dispatch-and-object-store.md) |
| Persistent caching | `disk_cache.py`, `crop.py` | [Disk cache](../concepts/disk-cache.md) |
| Files in and out: renders, exports, path refusal | `artifacts.py`, `renderer.py`, `chatbot/app.py` | [Artifacts and the file boundary](../concepts/artifacts-and-file-boundary.md) |
| Simulations, geocoding, runs | `runner.py`, `analysis.py`, `geocode.py` | [Simulations and task tools](../workflows/simulations.md) |
| Simulations on the DTCC Engine | `engine.py`, `runner.py` | [DTCC Engine integration](../integrations/dtcc-engine.md) |
| Web chat app | `chatbot/app.py`, `chatbot/sessions.py` | [Chatbot (Lurkie)](../integrations/chatbot-lurkie.md) |
| Answer records and measurement | `chatbot/provenance.py`, `dtcc_agent/provenance.py`, `eval/measure.py` | [Provenance and evaluation](../operations/provenance-and-evaluation.md) |

## Two tool surfaces

The server exposes two kinds of tools side by side:

- **Hardcoded, task-shaped tools** such as `geocode`, `get_buildings`, `run_simulation` and `compare_scenarios`. They compose several steps and return summaries.
- **Generic dispatch tools** (`list_operations`, `describe_operation`, `run_operation`, and the object tools `list_objects`, `inspect_object`, `export_object`, `spatial_query` and others). These expose the entire dtcc-core catalogue (133 operations). Results are kept server-side and referred to by typed references, so multi-step pipelines never pass geometry through the LLM.

ADR-0007 keeps this generic dispatch in the agent. Since 2026-10-09 the platform direction is that all queuing goes through the DTCC Engine, so ADR-0007 is due to be superseded once M4 measures whether Core downloads should move there (#110, #111).

## dtcc-core is a hard dependency

`dtcc_agent/__init__.py` imports `dtcc_core` at package import and raises an `ImportError` with reinstall instructions if it is missing. Historically, a missing Core degraded silently into a server with an empty catalogue that "looks exactly like success". Core is pinned to a commit in `pyproject.toml` and moved only after the contract workflow passes (see [Deployment and CI](../operations/deployment-and-ci.md)).

## Deployment shape and simulation paths

`docker-compose.yml` runs two services from one image (T13): the MCP server over streamable HTTP, bound to `127.0.0.1:8051` and never published (U11), and the chatbot, which shares its network namespace and its `/data` volume. The chat's port 8050 is published on the MCP service, since the two share one namespace. Compose refuses to start without the MCP secret, the access code and a Bedrock key.

Simulations run on exactly one of three paths, chosen by configuration with no fallback between them (`runner.py`):

1. **The DTCC Engine** (#113), with `DTCC_ENGINE_URL` and `DTCC_ENGINE_TOKEN` set: each simulation is an Engine job, and its result comes back over HTTP as a package. No folder is shared.
2. **The dtcc-sim mini-service**, with `DTCC_REMOTE_SERVICES` or `DTCC_SIM_SERVICE_URL` set: Core's remote dataset protocol submits over HTTP but reads the result file from a folder both services mount (`SHARED_RESULTS_DIR`).
3. **In process**, with neither: dtcc-sim is imported directly, which needs the full scientific stack (FEniCSx/dolfinx and the TetGen wrapper).

Everything else (buildings, terrain, meshing, rendering) runs in the agent in all three cases.

## History and direction

The rebuild (ADR-0009, `docs/plans/2026-09-19-rebuild-plan.md`) replaced the Claude Agent SDK with pydantic-ai on Bedrock (ADR-0003, M3; the SDK rollback was removed in #105), introduced typed references (ADR-0010) and plans retrieval as a separate MCP server (ADR-0005). M4 (#30) puts the agent on the DTCC Engine and makes it ready for a deployment that the platform team designs. See the [ADR map](../decisions/adr-map.md) for how far each decision has landed.
