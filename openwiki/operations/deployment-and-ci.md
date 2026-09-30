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
  - id: openwiki-source-931ea4e3e14cfe3c996abf4a
    resource: repo://chatbot/memory.py
  - id: openwiki-source-b79fbbd921df689b4bbdc82f
    resource: repo://docker-compose.yml
  - id: openwiki-source-bb1ebe868e35e9e500714501
    resource: repo://Dockerfile
  - id: openwiki-source-052f7c9f16ee5a8169a3fb7d
    resource: repo://dtcc_agent/disk_cache.py
  - id: openwiki-source-4163f0ea9e6726ccca521458
    resource: repo://dtcc_agent/registry.py
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: { by: "claude-code", at: "2026-09-30T14:41:08.402Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-30T14:41:08.402Z
---

# Deployment, configuration and CI

## Docker mini-service

- **`Dockerfile`**
  - Built from `python:3.12-slim` with `build-essential`, `curl` and `git`.
  - Installs `dtcc-core @ git+...@${DTCC_CORE_REF}` (build arg, default `develop`), then `pip install -e ".[chatbot]"`.
  - Runs as a non-root `dtcc-agent` user (`APP_UID` and `APP_GID`, default 1000).
  - Its command is `uvicorn chatbot.app:app --host 0.0.0.0 --port 8050`: the container runs **the chatbot**. The chatbot spawns the MCP server over stdio unless `DTCC_MCP_URL` is set.
- **`docker-compose.yml`**
  - One `dtcc-agent` service on port `8050`, platform `linux/amd64` by default.
  - Mounts `${DTCC_AGENT_DATA:-./data/agent}:/data` (logs, memory, cache) and dtcc-sim's shared results at `/shared/results`.
  - The cache at `/data/cache` must pass the disk cache's trust check: owned by the container user (uid `APP_UID`, default 1000), not group- or world-writable, no symlinks inside. On a fresh data dir the agent creates it `0700` itself. If the host pre-creates it group-writable (for example `chmod 777` to paper over a uid mismatch), the agent refuses to start and names the fix. See [Disk cache](../concepts/disk-cache.md).
  - Healthcheck: `GET /health`.
- **`build_docker.sh`** exports the image, tag, platform, Core ref and UID/GID defaults, then runs `docker compose build dtcc-agent`.

The `DTCC_CORE_REF=develop` build default differs from the commit pinned in `pyproject.toml`. Which Core ends up in the image therefore depends on pip resolving the editable install against the pin. Pass `DTCC_CORE_REF=<pinned sha>` to be certain.

To start: run dtcc-sim (`docker compose up -d` in `../dtcc-sim`) and `docker compose up --build` here, with `DTCC_REMOTE_SERVICES` pointing at it. The order no longer matters: dtcc-sim's datasets join the catalogue once it answers, asked from a background thread every 30 s.

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
| `DTCC_AGENT_LOG_DIR` | `/tmp/dtcc_lurkie_logs` | `chatbot.app` | Log files |
| `DTCC_AGENT_MEMORY_DIR` | `/tmp/dtcc_lurkie_memory` | `chatbot.memory` | ChromaDB persistence |
| `DTCC_AGENT_RENDERS_DIR` | `/tmp/dtcc_screenshots` | `chatbot.app` | Directory served at `/renders` |

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

**`openwiki-update`** (`openwiki-update.yml`) is the scheduled workflow that regenerates this wiki.

## Local development

```sh
uv venv --python 3.12 && uv sync --locked --extra test --extra chatbot
uv run pytest -q
python -m dtcc_agent                     # MCP over stdio
DTCC_MCP_TRANSPORT=http python -m dtcc_agent
python -m chatbot                        # web chatbot on :8050
```

Python ≥ 3.12 is required; 3.11 fails to resolve.
