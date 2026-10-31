---
type: operations
title: Deployment, configuration and CI
description: Covers the Docker image and two-service compose layout, Bedrock credentials, admission, memory limits and restarts, every environment variable the server, chatbot and cache read (including the DTCC Engine settings), and the GitHub workflows (build-tests, dtcc-core contract, PR-Agent).
tags: [deployment, docker, configuration, ci, github-actions]
sources:
  - id: openwiki-source-715dace563ef484b6e8bd1e2
    resource: repo://.dockerignore
  - id: openwiki-source-b69b0be58e9ad19ca644d766
    resource: repo://.github/workflows/ci-build-tests.yml
  - id: openwiki-source-b3e643290de65ed93425d581
    resource: repo://.github/workflows/dtcc-core-contract.yml
  - id: openwiki-source-8ee5c2e6aef948e8ffd0e18a
    resource: repo://.github/workflows/pr-agent.yml
  - id: openwiki-source-0b9b769bb175577b7f47f6cd
    resource: repo://.pr_agent.toml
  - id: openwiki-source-d82fbc21a9f74516f7bfd0f8
    resource: repo://chatbot/app.py
  - id: openwiki-source-778a883bcdc0a6ed0b3401f7
    resource: repo://chatbot/config.py
  - id: openwiki-source-931ea4e3e14cfe3c996abf4a
    resource: repo://chatbot/memory.py
  - id: openwiki-source-b79fbbd921df689b4bbdc82f
    resource: repo://docker-compose.yml
  - id: openwiki-source-bb1ebe868e35e9e500714501
    resource: repo://Dockerfile
  - id: openwiki-source-140a7f24af942881ef83af15
    resource: repo://docs/deployment.md
  - id: openwiki-source-58b44e3a6e999eaaad2363b4
    resource: repo://dtcc_agent/artifacts.py
  - id: openwiki-source-052f7c9f16ee5a8169a3fb7d
    resource: repo://dtcc_agent/disk_cache.py
  - id: openwiki-source-4163f0ea9e6726ccca521458
    resource: repo://dtcc_agent/registry.py
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-4a767438bf3ca3ff31fd9e91
    resource: repo://tests/test_deployment_doc.py
  - id: openwiki-source-7f232b69435f45b7dd45b6bd
    resource: repo://tests/test_dockerignore.py
generated: { by: "claude-code", at: "2026-10-31T10:29:51.442Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-10-31T10:29:51.442Z
---

# Deployment, configuration and CI

## Docker mini-service

- **`Dockerfile`**
  - Built from `python:3.12-slim` with `build-essential`, `curl` and `git`.
  - Runs `pip install -e ".[${EXTRAS}]"` (`EXTRAS` defaults to `chatbot`; the `sdk` extra was removed in #105), which installs dtcc-core from the commit pinned in `pyproject.toml`, then **asserts** that the installed Core's `direct_url.json` commit equals the pin, failing the build otherwise (U10, #61). The old `DTCC_CORE_REF` build argument, which defaulted to Core's moving `develop` and won over the pin, is gone.
  - Sets `DTCC_AGENT_ARTIFACTS_DIR=/data/artifacts` and creates `/data/artifacts`, `/data/cache`, `/data/logs`, `/data/memory` and `/shared/results` owned by the app user.
  - Runs as a non-root `dtcc-agent` user (`APP_UID` and `APP_GID`, default 1000). Its default command starts the chatbot with uvicorn on port 8050.
  - Copies the whole build context (`COPY . /app`), so `.dockerignore` decides what stays out. It excludes `.env` and `.env.*` (#133), so a local secrets file is never baked into the image; `tests/test_dockerignore.py` keeps those lines in place.
- **`docker-compose.yml`** runs **two services from one image** (T13, #67):
  - **`dtcc-agent-mcp`** runs `python -m dtcc_agent` with `DTCC_MCP_TRANSPORT=http` on `127.0.0.1:8051`. That port is **never published** (U11): until admission control (T14) nothing outside the pair can reach it. Because it owns the network namespace, it publishes the chatbot's port `8050`. It mounts `/data` and dtcc-sim's shared results at `/shared/results` (`SHARED_RESULTS_DIR`). Its healthcheck connects to 8051, which happens only after the catalogue is built (`start_period` 120 s).
  - **`dtcc-agent`** (the chatbot) uses `network_mode: service:dtcc-agent-mcp`, so it reaches the server over loopback at `DTCC_MCP_URL=http://127.0.0.1:8051/mcp`, which also satisfies FastMCP's default DNS-rebinding allowlist. It starts once the MCP service is healthy, mounts the same `/data`, and carries the Bedrock settings, the probe switch and the four limit settings. Healthcheck: `GET /health`.
  - Both services share `/data`, so the MCP server writes each Session's artifacts to `/data/artifacts` and the chatbot serves them (see [Artifacts and the file boundary](../concepts/artifacts-and-file-boundary.md)).
  - The cache at `/data/cache` must pass the disk cache's trust check: owned by the container user (uid `APP_UID`, default 1000), not group- or world-writable, no symlinks inside. On a fresh data dir the agent creates it `0700` itself. If the host pre-creates it group-writable (for example `chmod 777` to paper over a uid mismatch), the agent refuses to start and names the fix. See [Disk cache](../concepts/disk-cache.md).
- **`build_docker.sh`** exports the image, tag, platform and UID/GID defaults, then runs `docker compose build dtcc-agent-mcp`, the service that carries the `build:` block.

Verified with `docker compose up --build`: port 8051 answered neither the host nor another container on the compose network; a session's render loaded through the chatbot (200) and was refused to another session (404); the container ran the pinned Core.

To start: run dtcc-sim (`docker compose up -d` in `../dtcc-sim`) and `docker compose up --build` here, with `DTCC_REMOTE_SERVICES` pointing at it. The order does not matter: dtcc-sim's datasets join the catalogue once it answers, asked from a background thread every 30 s.

### Bedrock credentials, admission and restarts

- **Bedrock.** The chatbot calls the model on Amazon Bedrock only (#29). `AWS_BEARER_TOKEN_BEDROCK` (a Bedrock API key) is optional (#130): Compose passes it through only when it is set, and without it the chatbot uses botocore's credential chain, such as the host's instance role in AWS (in containers that also needs an instance-metadata hop limit of 2). Compose also passes `AWS_REGION` (default `eu-north-1`), `DTCC_AGENT_MODEL` (default `eu.anthropic.claude-sonnet-5-5`) and `DTCC_AGENT_BEDROCK_PROBE` (default `1`; `0` skips the paid startup check, for liveness checks). Without a credential of any kind the chatbot does not start; with a wrong one it starts, logs a loud error from its background key probe, and tells users to ask the administrator (#106). See [Model runtime](../architecture/model-runtime.md).
- **Admission.** Compose refuses to start without `DTCC_MCP_SECRET` and `DTCC_AGENT_ACCESS_CODE`, naming the missing one, so a deployment cannot run open by accident (T14, #76). See [Sessions and isolation](../architecture/sessions-and-isolation.md).
- **Memory and restarts** (#79, #82). The MCP service has `mem_limit` (`DTCC_MCP_MEM_LIMIT`, default `5g`), so the kernel kills this server and not a neighbour if one operation outgrows memory, and `restart: unless-stopped`. If the MCP service restarts, the chatbot is left in the old network namespace, so the chatbot's health check (its own `/health` plus a connection to `127.0.0.1:8051`) kills its PID 1 on failure, and its restart policy brings it back inside the new namespace. A local load test (2026-10-09) ran 10 chats at once with no restarts and a peak of 4,15 of 5 GiB in the MCP service.
- **Limits** (#132). `DTCC_AGENT_MONTHLY_SPEND_USD`, `DTCC_AGENT_TURNS_PER_CHAT`, `DTCC_AGENT_TURN_SECONDS` and `DTCC_AGENT_MODEL_CONCURRENCY` are passed to the chatbot and are off when empty. The spend ledger is `/data/logs/spend.json`. Suggested for the invited beta: 90, 30, 180 and 4. See [Turn and spend limits](turn-and-spend-limits.md).
- **Behind a gateway** (#134). The page uses relative URLs, so it can be served under a prefix the gateway strips, such as `https://api.dtcc.chalmers.se/agent/`; the gateway must route bare `/agent` as well as `/agent/*`, and allow the websocket. See [Chatbot (Lurkie)](../integrations/chatbot-lurkie.md).
- **What a deployment needs.** `docs/deployment.md` lists the shape, secrets, settings, storage, memory, state and outbound network for the deployment design (#122); `tests/test_deployment_doc.py` fails when it and `docker-compose.yml` name different variables.
- **Simulations.** With `DTCC_ENGINE_URL` and `DTCC_ENGINE_TOKEN` set, simulations run on the DTCC Engine over HTTP and the `/shared/results` mount is not needed; otherwise the MCP service needs the folder it shares with dtcc-sim. See [DTCC Engine integration](../integrations/dtcc-engine.md).

## Environment variables

| Variable | Default | Read by | Effect |
|---|---|---|---|
| `DTCC_MCP_TRANSPORT` | `stdio` | `server.main` | `stdio` or `http` |
| `DTCC_MCP_HOST`, `DTCC_MCP_PORT` | `127.0.0.1`, `8051` | `server.main` | HTTP bind address |
| `DTCC_MCP_WORKERS` | `4` | `runtime` | Process-wide tool worker pool; must be an integer ≥ 1 |
| `DTCC_MCP_SECRET` | none | `server.main`, `chatbot.config` | Bearer secret every MCP HTTP request must carry; required to bind a non-loopback host |
| `DTCC_AGENT_ACCESS_CODE` | none | `chatbot.app` | Code to open a chat, at least 16 characters; unset turns admission off with a warning |
| `AWS_BEARER_TOKEN_BEDROCK` | none | `chatbot.config`, boto | Bedrock API key. Optional: without it, botocore's credential chain (keys, a profile, a container or instance role) |
| `AWS_REGION` | `eu-north-1` | `chatbot.runtime.pai` | Bedrock region |
| `DTCC_AGENT_MODEL` | `eu.anthropic.claude-sonnet-5-5` | `chatbot.config` | Bedrock model ID; changing models is this variable alone |
| `DTCC_AGENT_BEDROCK_PROBE` | `1` | `chatbot.app` | `0` turns off the paid startup credential check |
| `DTCC_AGENT_MONTHLY_SPEND_USD` | none (off) | `chatbot.limits` | Model spend per calendar month (UTC), never exceeded; needs `DTCC_AGENT_LOG_DIR` |
| `DTCC_AGENT_TURNS_PER_CHAT` | none (off) | `chatbot.limits` | Turns one chat may start |
| `DTCC_AGENT_TURN_SECONDS` | none (off) | `chatbot.limits` | Deadline per turn |
| `DTCC_AGENT_MODEL_CONCURRENCY` | none (off) | `chatbot.limits` | Model requests at once, across chats |
| `DTCC_AGENT_RUNTIME` | `pydantic-ai` | `chatbot.runtime` | Only `pydantic-ai` or unset; anything else stops the chatbot |
| `DTCC_ENGINE_URL`, `DTCC_ENGINE_TOKEN` | none | `engine`, `runner` | Run simulations as DTCC Engine jobs; the token is required with the URL |
| `DTCC_ENGINE_TIMEOUT` | `900` | `engine` | Seconds per simulation before a timeout |
| `DTCC_ENGINE_CORE` | none | `engine`, `dispatcher`, `server` | Experimental (#110): `1` downloads buildings and point clouds as Engine jobs too, without the disk cache; leave unset in a deployment |
| `DTCC_MCP_OBJECT_BUDGET_GIB` | `2` | `server` | Memory all chats may use for kept objects; one chat gets at most half (#122) |
| `DTCC_AGENT_MAX_AREA_KM2` | `10` | `dispatcher` | Largest area one heavy download may cover |
| `DTCC_SIM_SERVICE_URL` | none | `runner`, `registry` | Single dtcc-sim service; takes precedence over `DTCC_REMOTE_SERVICES`, and both are ignored while the Engine is set. Its datasets join the catalogue from a background retrier, never at startup |
| `DTCC_REMOTE_SERVICES` | none | `runner` | Comma-separated dtcc-sim services |
| `DTCC_AGENT_CACHE_DIR` | `$XDG_CACHE_HOME/dtcc_agent` (usually `~/.cache/dtcc_agent`) | `disk_cache` | Persistent cache location; must be private to the agent's user |
| `DTCC_MCP_URL` | none | `chatbot.config` | Connect the chatbot to an HTTP MCP server |
| `DTCC_AGENT_PYTHON` | current interpreter | `chatbot.config` | Interpreter for the stdio MCP child process |
| `DTCC_AGENT_HOST`, `DTCC_AGENT_PORT` | `0.0.0.0`, `8050` | `chatbot.config` | Chatbot bind address (for `python -m chatbot`) |
| `DTCC_AGENT_LOG_DIR` | `/tmp/dtcc_lurkie_logs` | `chatbot.config`, `builder_calls` | Log files, answer records, the spend ledger, and `builder_calls.jsonl` (not written when unset on the server) |
| `DTCC_AGENT_MEMORY_DIR` | `/tmp/dtcc_lurkie_memory` | `chatbot.memory` | ChromaDB persistence |
| `DTCC_AGENT_ARTIFACTS_DIR` | `<system temp>/dtcc_agent_artifacts` (Docker: `/data/artifacts`) | `artifacts` (server and chatbot) | Root of the per-Session artifact folders |
| `DTCC_AGENT_SESSION`, `DTCC_AGENT_SUBJECT`, `DTCC_AGENT_TURN` | `local`, `anonymous`, none | `server` | The stdio server's Session, subject and turn; the chatbot sets them per message |
| `SHARED_RESULTS_DIR` | none | `artifacts`, Core's remote protocol | The folder shared with dtcc-sim: `load_geojson` reads it, and the mini-service path reads every simulation result from it (Docker: `/shared/results`; not needed with the Engine) |

## CI workflows

**`build-tests`** (`ci-build-tests.yml`) runs on pushes and PRs to `develop` and `main`:

1. `uv sync --locked --extra test --extra chatbot` on Python 3.12. The pin is needed because fiona has no 3.14 wheels.
2. `pytest -q` (658 tests at `11f20bb`). The job that ran without the Agent SDK extra was removed with the SDK (#105).
3. Print `catalogue: N operations`. This is not an assertion; it makes silent drift visible.
4. `uv build`.

**`dtcc-core contract`** (`dtcc-core-contract.yml`) is triggered by `workflow_dispatch` with `core_sha`, or by a `repository_dispatch` of type `dtcc-core-contract`:

1. Validate a full 40-character SHA.
2. Sync the locked environment without Core, then install the candidate Core.
3. Assert the installed `direct_url.json` commit equals the SHA.
4. Run the whole suite as the contract.
5. Print the catalogue size.

A green run is the signal to move the `pyproject.toml` pin by hand. The upstream `bump-core-pin` job is omitted because this repo lacks the `DTCC_CORE_BUMP_TOKEN` secret. See [Operation catalogue](../concepts/operation-catalogue.md).

**`PR-Agent`** (`pr-agent.yml` plus `.pr_agent.toml`) is an automated first-pass reviewer, not a gate.

- **Triggers.** It runs on PR open, reopen and ready-for-review, and on `/review`, `/improve` and similar comments from owners, members or collaborators only. This protects the model key on a public repo.
- **Secret safety.** It uses `pull_request`, not `pull_request_target`, so fork PRs never see secrets.
- **Install.** It pip-installs `pr-agent==0.46.0`, because the org's Actions policy disallows the upstream action.
- **Model and scope.** It uses Gemini (`GEMINI_API_KEY`) with auto-review and auto-improve on and auto-describe off.
- **Repo guidance.** `.pr_agent.toml` directs reviews toward Session isolation, `@tool` concurrency ordering, and silent failures. It is trialling `focus_only_on_problems = false` for code suggestions.

This wiki has no workflow: it is regenerated locally with `/openwiki` (update mode) every few pull requests.

## Local development

```sh
uv venv --python 3.12 && uv sync --locked --extra test --extra chatbot
uv run pytest -q
python -m dtcc_agent                     # MCP over stdio
DTCC_MCP_TRANSPORT=http python -m dtcc_agent
python -m chatbot                        # web chatbot on :8050
```

Python ≥ 3.12 is required; 3.11 fails to resolve.
