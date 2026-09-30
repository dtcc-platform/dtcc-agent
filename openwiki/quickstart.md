---
type: quickstart
title: Quickstart
description: A starting point for dtcc-agent. It says what the repo is, gives the commands to set up, run and test it, and points to the wiki page for each common task.
tags: [quickstart, setup, routing]
sources:
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-a0c3189dfa34667761d05df4
    resource: repo://chatbot/__main__.py
  - id: openwiki-source-d82fbc21a9f74516f7bfd0f8
    resource: repo://chatbot/app.py
  - id: openwiki-source-778a883bcdc0a6ed0b3401f7
    resource: repo://chatbot/config.py
  - id: openwiki-source-e23a39a8942c83ff4b942657
    resource: repo://dtcc_agent/runtime.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: { by: "claude-code", at: "2026-09-30T14:41:08.402Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-30T14:41:08.402Z
---

# Quickstart

dtcc-agent is the conversational front door to the DTCC urban digital twin platform. It has two parts:

- **An MCP server** (`dtcc_agent/`). It exposes dtcc-core and dtcc-sim as tools: 22 of them, 133 operations behind the generic ones at the current Core pin.
- **A web chatbot, "Lurkie"** (`chatbot/`). It drives a Claude agent against that server.

Read `CONTEXT.md` for the vocabulary (Operation, Object, Run, Session, Field) and `docs/adr/` for the decisions. ADRs take precedence over `docs/plans/`.

## Set up and run

```sh
uv venv --python 3.12 && uv sync --locked --extra test --extra chatbot   # 3.11 does not resolve
uv run pytest -q                                                         # full suite
python -m dtcc_agent                                                     # MCP over stdio
DTCC_MCP_TRANSPORT=http DTCC_MCP_PORT=8051 python -m dtcc_agent          # MCP over HTTP (needs X-DTCC-Session)
DTCC_MCP_URL=http://127.0.0.1:8051/mcp python -m chatbot                 # chatbot on :8050 against it
```

Without `DTCC_MCP_URL`, the chatbot spawns the server over stdio. The HTTP server prints `dtcc-agent: catalogue built: N operations` at startup; if it exits instead, naming a catalogue section, the dtcc-core install is broken. A stdio server builds the catalogue on the first call that needs it, and a broken install shows up as that call's error.

dtcc-core is pinned to a commit in `pyproject.toml`. Do not move the pin by hand: run the `dtcc-core contract` workflow first.

## Where to go for a task

| I want to… | Read |
|---|---|
| Understand the whole system | [Architecture overview](architecture/overview.md) |
| Add or change an MCP tool, or debug concurrency, hangs or "event loop" errors | [MCP server and tool execution](architecture/mcp-server-and-tool-execution.md) |
| Reason about multi-user safety or what one user can see | [Sessions and isolation](architecture/sessions-and-isolation.md) |
| Find out why an operation is missing, or pick up a new Core | [Operation catalogue](concepts/operation-catalogue.md) |
| Understand `run_operation`, typed references (`obj_…`, `run_…`) and result summaries | [Dispatch, object references and serialization](concepts/dispatch-and-object-store.md) |
| Debug stale, wrong or slow cached results | [Disk cache](concepts/disk-cache.md) |
| Work on heat or air-quality simulations, geocoding or scenario comparison | [Simulations, runs and geocoding](workflows/simulations.md) |
| Change the chatbot, prompt, memory or UI | [Lurkie chatbot](integrations/chatbot-lurkie.md) |
| Deploy, configure env vars, or change CI | [Deployment, configuration and CI](operations/deployment-and-ci.md) |
| Know what is decided and what is still planned | [Architecture decisions map](decisions/adr-map.md) |
| Run or extend tests | [Test suite](testing/test-suite.md) |

## Working conventions

- Issues live on GitHub, `dtcc-platform/dtcc-agent`. Read `docs/agents/issue-tracker.md` before filing, and `docs/agents/triage-labels.md` for the five triage roles.
- The rebuild happens on feature branches off `develop`, with one PR per milestone task (ADR-0009). `CHANGELOG.md` records what shipped and how it was verified.
