---
type: operations
title: Deployment, configuration and CI
description: Covers the Docker image and compose layout, every environment variable the server, chatbot and cache read, Claude auth for containers, and the GitHub workflows (build-tests, dtcc-core contract, PR-Agent).
tags: [deployment, docker, configuration, ci, github-actions]
sources:
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
  - id: openwiki-source-b79fbbd921df689b4bbdc82f
    resource: repo://docker-compose.yml
  - id: openwiki-source-bb1ebe868e35e9e500714501
    resource: repo://Dockerfile
  - id: openwiki-source-58b44e3a6e999eaaad2363b4
    resource: repo://dtcc_agent/artifacts.py
  - id: openwiki-source-052f7c9f16ee5a8169a3fb7d
    resource: repo://dtcc_agent/disk_cache.py
  - id: openwiki-source-4163f0ea9e6726ccca521458
    resource: repo://dtcc_agent/registry.py
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: { by: "claude-code", at: "2026-10-01T20:29:16.810Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-10-01T20:29:16.810Z
---

# Deployment, configuration and CI

## Docker mini-service

- **`Dockerfile`**
  - Built from `python:3.12-slim` with `build-essential`, `curl` and `git`.
  - Runs `pip install -e ".[chatbot]"`, which installs dtcc-core from the commit pinned in `pyproject.toml`, then **asserts** that the installed Core's `direct_url.json` commit equals the pin, failing the build otherwise (U10, #61). The old `DTCC_CORE_REF` build argument, which defaulted to Core's moving `develop` and won over the pin, is gone.
  - Sets `DTCC_AGENT_ARTIFACTS_DIR=/data/artifacts` and creates `/data/artifacts`, `/data/cache`, `/data/logs`, `/data/memory` and `/shared/results` owned by the app user.
  - Runs as a non-root `dtcc-agent` user (`APP_UID` and `APP_GID`, default 1000). Its default command starts the chatbot with uvicorn on port 8050.
- **`docker-compose.yml`** runs **two services from one image** (T13, #67):
  - **`dtcc-agent-mcp`** runs `python -m dtcc_agent` with `DTCC_MCP_TRANSPORT=http` on `127.0.0.1:8051`. That port is **never published** (U11): until admission control (T14) nothing outside the pair can reach it. Because it owns the network namespace, it publishes the chatbot's port `8050`. It mounts `/data` and dtcc-sim's shared results at `/shared/results` (`SHARED_RESULTS_DIR`). Its healthcheck connects to 8051, which happens only after the catalogue is built (`start_period` 120 s).
  - **`dtcc-agent`** (the chatbot) uses `network_mode: service:dtcc-agent-mcp`, so it reaches the server over loopback at `DTCC_MCP_URL=http://127.0.0.1:8051/mcp`, which also satisfies FastMCP's default DNS-rebinding allowlist. It starts once the MCP service is healthy, mounts the same `/data`, and carries the Claude credentials. Healthcheck: `GET /health`.
  - Both services share `/data`, so the MCP server writes each Session's artifacts to `/data/artifacts` and the chatbot serves them (see [Artifacts and the file boundary](../concepts/artifacts-and-file-boundary.md)).
  - The cache at `/data/cache` must pass the disk cache's trust check: owned by the container user (uid `APP_UID`, default 1000), not group- or world-writable, no symlinks inside. On a fresh data dir the agent creates it `0700` itself. If the host pre-creates it group-writable (for example `chmod 777` to paper over a uid mismatch), the agent refuses to start and names the fix. See [Disk cache](../concepts/disk-cache.md).
- **`build_docker.sh`** exports the image, tag, platform and UID/GID defaults, then runs `docker compose build dtcc-agent-mcp`, the service that carries the `build:` block.

Verified with `docker compose up --build`: port 8051 answered neither the host nor another container on the compose network; a session's render loaded through the chatbot (200) and was refused to another session (404); the container ran the pinned Core.

To start: run dtcc-sim (`docker compose up -d` in `../dtcc-sim`) and `docker compose up --build` here, with `DTCC_REMOTE_SERVICES` pointing at it. The order does not matter: dtcc-sim's datasets join the catalogue once it answers, asked from a background thread every 30 s.

### Claude auth in containers

Set either `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY`. On macOS the Claude Code credential lives in the Keychain, so mounting `~/.claude` is not enough. Run `claude setup-token` interactively on the host and export the result. Do not wrap it in command substitution: it can capture the login screen into the variable. To check the token without printing it, the README gives a POSIX `case` test, run on the host and again through `docker compose exec` in the container. It reports an empty value or one containing whitespace (the captured-login-screen failure), a value without the `sk-ant-oat01-` prefix, or a correct-looking one. It checks shape only: one chat message through the service is the real test. The `verify_auth.py` script the README used to run was never in the repository (#40, fixed by #54).

## Environment variables

| Variable | Default | Read by | Effect |
|---|---|---|---|
| `DTCC_MCP_TRANSPORT` | `stdio` | `server.main` | `stdio` or `http` |
| `DTCC_MCP_HOST`, `DTCC_MCP_PORT` | `127.0.0.1`, `8051` | `server.main` | HTTP bind address |
| `DTCC_MCP_WORKERS` | `4` | `runtime` | Process-wide tool worker pool; must be an integer ≥ 1 |
| `DTCC_SIM_SERVICE_URL` | none | `runner`, `registry` | Single dtcc-sim service; takes precedence. Its datasets join the catalogue from a background retrier, never at startup |
| `DTCC_REMOTE_SERVICES` | none | `runner` | Comma-separated dtcc-sim services |
| `DTCC_AGENT_CACHE_DIR` | `$XDG_CACHE_HOME/dtcc_agent` (usually `~/.cache/dtcc_agent`) | `disk_cache` | Persistent cache location; must be private to the agent's user |
| `DTCC_MCP_URL` | none | `chatbot.config` | Connect the chatbot to an HTTP MCP server |
| `DTCC_AGENT_PYTHON` | current interpreter | `chatbot.config` | Interpreter for the stdio MCP child process |
| `DTCC_AGENT_HOST`, `DTCC_AGENT_PORT` | `0.0.0.0`, `8050` | `chatbot.config` | Chatbot bind address (for `python -m chatbot`) |
| `DTCC_AGENT_LOG_DIR` | `/tmp/dtcc_lurkie_logs` | `chatbot.app`, `builder_calls` | Log files, and `builder_calls.jsonl` (not written when unset on the server) |
| `DTCC_AGENT_MEMORY_DIR` | `/tmp/dtcc_lurkie_memory` | `chatbot.memory` | ChromaDB persistence |
| `DTCC_AGENT_ARTIFACTS_DIR` | `<system temp>/dtcc_agent_artifacts` (Docker: `/data/artifacts`) | `artifacts` (server and chatbot) | Root of the per-Session artifact folders |
| `DTCC_AGENT_SESSION` | `local` | `server` | The stdio server's Session id; the chatbot sets it per message |
| `SHARED_RESULTS_DIR` | none | `artifacts` | The only folder `load_geojson` reads (Docker: `/shared/results`) |

## CI workflows

**`build-tests`** (`ci-build-tests.yml`) runs on pushes and PRs to `develop` and `main`:

1. `uv sync --locked --extra test --extra chatbot` on Python 3.12. The pin is needed because fiona has no 3.14 wheels.
2. `pytest -q`.
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
