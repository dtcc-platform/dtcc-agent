---
type: architecture
title: Architecture overview
description: End-to-end picture of dtcc-agent, from the Lurkie web chatbot through the MCP server and generic dispatch to dtcc-core and dtcc-sim, with a module ownership map and the two deployment modes.
tags: [architecture, overview, mcp, chatbot, dtcc-core]
sources:
  - id: openwiki-source-d82fbc21a9f74516f7bfd0f8
    resource: repo://chatbot/app.py
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
generated: { by: "claude-code", at: "2026-09-27T19:28:40.080Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-30T14:41:08.402Z
---

# Architecture overview

dtcc-agent is the conversational front door to the DTCC platform (ADR-0001, accepted 2026-09-15). A person asks a question in their own words. An LLM turns it into a sequence of platform Operations, and the answer comes back as text and rendered 3D images.

The repo ships two Python packages:

- **`dtcc_agent`**: an MCP server exposing dtcc-core and dtcc-sim as tools. It can be used on its own from Claude Desktop, Claude Code or any MCP client.
- **`chatbot`** ("Lurkie"): a FastAPI web app that runs a Claude agent session against that MCP server and streams the conversation over a WebSocket.

```
browser (chatbot/static/index.html)
   │  WebSocket /chat
   ▼
chatbot/app.py ── Claude Agent SDK ── system prompt (chatbot/config.py)
   │  MCP (stdio child process, or streamable-http + X-DTCC-Session)
   ▼
dtcc_agent/server.py   @tool functions, worker pool, per-Session state
   ├── runtime.py       process-wide setup: catalogue build, worker limits
   ├── registry.py      catalogue from the pinned Core, plus dtcc-sim datasets
   ├── dispatcher.py    run_operation: resolve refs → call → store → summarise
   ├── object_store.py  per-Session LRU of live objects
   ├── disk_cache.py    persistent cache for downloads and builders
   ├── serializers.py   type-specific summaries (never raw arrays)
   ├── runner.py        simulations: remote dtcc-sim service or in-process
   ├── geocode.py       place name → EPSG:3006 bounds
   └── renderer.py      offscreen PNG via dtcc-viewer
   ▼
dtcc-core (pinned commit) / dtcc-sim (optional, local or remote)
```

## Module ownership

| Concern | Owner | Page |
|---|---|---|
| Tool registration, concurrency, transports, startup | `server.py`, `runtime.py` | [MCP server and tool execution](mcp-server-and-tool-execution.md) |
| Session isolation | `server.py` (`_Session`), `chatbot/sessions.py` | [Sessions and isolation](sessions-and-isolation.md) |
| What operations exist | `registry.py` | [Operation catalogue](../concepts/operation-catalogue.md) |
| Running an operation and holding its result | `dispatcher.py`, `object_store.py`, `serializers.py` | [Dispatch, object references and serialization](../concepts/dispatch-and-object-store.md) |
| Persistent caching | `disk_cache.py`, `crop.py` | [Disk cache](../concepts/disk-cache.md) |
| Simulations, geocoding, runs | `runner.py`, `geocode.py`, `analysis.py` | [Simulations, runs and geocoding](../workflows/simulations.md) |
| Web chat, memory, prompt | `chatbot/` | [Lurkie chatbot](../integrations/chatbot-lurkie.md) |

## Two tool surfaces

The server exposes two kinds of tools side by side:

- **Hardcoded, task-shaped tools** such as `geocode`, `get_buildings`, `run_simulation` and `compare_scenarios`. They compose several steps and return summaries.
- **Generic dispatch tools** (`list_operations`, `describe_operation`, `run_operation`, and the object tools `list_objects`, `inspect_object`, `export_object`, `spatial_query` and others). These expose the entire dtcc-core catalogue. Results are kept server-side and referred to by short IDs, so multi-step pipelines never pass geometry through the LLM.

ADR-0007 keeps this generic dispatch in the agent, rather than waiting for a Twin-owned catalogue.

## dtcc-core is a hard dependency

`dtcc_agent/__init__.py` imports `dtcc_core` at package import and raises an `ImportError` with reinstall instructions if it is missing. Historically, a missing Core degraded silently into a server with an empty catalogue that "looks exactly like success". Core is pinned to a commit in `pyproject.toml` and moved only after the contract workflow passes (see [Deployment, configuration and CI](../operations/deployment-and-ci.md)).

## Deployment modes

- **Mini-service mode.** The chatbot and MCP server run in a light Python container (`Dockerfile`, `docker-compose.yml`). Simulations are delegated to a running dtcc-sim service over dtcc-core's remote dataset protocol, configured with `DTCC_REMOTE_SERVICES` or `DTCC_SIM_SERVICE_URL`.
- **Direct mode.** The server imports `dtcc_core` and `dtcc_sim` in-process. This needs the full scientific stack (FEniCSx/dolfinx and the TetGen wrapper).

`runner.py` picks the path at call time: remote when a service URL is configured, local dtcc-sim otherwise.

## Where this is heading

The rebuild (ADR-0009, `docs/plans/2026-09-19-rebuild-plan.md`) replaces the Claude Agent SDK with pydantic-ai (ADR-0003), introduces typed references (ADR-0010) and moves retrieval into a separate MCP server (ADR-0005). The platform's "DTCC Engine" API is also expected to sit between the agent and Core/Sim. See [Architecture decisions map](../decisions/adr-map.md) for how far each decision has landed.
