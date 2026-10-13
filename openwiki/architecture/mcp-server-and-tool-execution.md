---
type: architecture
title: MCP server and tool execution
description: How dtcc-agent registers its MCP tools on mcp 2.x's MCPServer, runs each call in a bounded worker pool bound to a Session, records every call, deduplicates concurrent downloads, guards HTTP with a bearer secret, and starts over stdio or stateless streamable-http.
tags: [mcp, server, concurrency, transport, startup]
verified:
  - by: openwiki/0.5.2
    at: 2026-10-13T19:36:58.526Z
sources:
  - id: openwiki-source-00348018efcad3e081121b7b
    resource: repo://dtcc_agent/provenance.py
  - id: openwiki-source-d8839a242913c8f59a48c041
    resource: repo://dtcc_agent/refs.py
  - id: openwiki-source-4163f0ea9e6726ccca521458
    resource: repo://dtcc_agent/registry.py
  - id: openwiki-source-e23a39a8942c83ff4b942657
    resource: repo://dtcc_agent/runtime.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
  - id: openwiki-source-c0b62da1c8d12500b49cd428
    resource: repo://tests/test_catalogue_startup.py
  - id: openwiki-source-0e41257659137c7a2c27de1c
    resource: repo://tests/test_engine.py
  - id: openwiki-source-7da8cb11cdc15fb1e5a1f088
    resource: repo://tests/test_http_sessions.py
  - id: openwiki-source-fa7af1493a897412d61af4b0
    resource: repo://tests/test_worker_pool.py
generated: { by: "claude-code", at: "2026-10-13T19:36:58.526Z" }
---

# MCP server and tool execution

`dtcc_agent/server.py` is the MCP server. It owns one `MCPServer("dtcc-agent")` instance (mcp 2.x; the 1.x `FastMCP` path was removed upstream), every tool definition, and the execution wrapper that decides *where* and *how many* tool bodies run at once. `dtcc_agent/runtime.py` owns the process-scoped pieces that must exist before the first call: the worker pool, the Engine configuration check and the catalogue build.

Entry point: `python -m dtcc_agent` → `dtcc_agent/__main__.py` → `server.main()`.

## The `@tool` decorator

Every tool is a plain synchronous function decorated with `@tool` (or `@tool(flight=...)`). The decorator does not hand the function itself to MCPServer; it registers an async wrapper, `run_bound`, and returns the original function unchanged so in-process callers (tests) can still call it synchronously. MCPServer finds the request context through an extra keyword-only `mcp_context: Context` parameter that the wrapper's signature adds; the tool's own schema never shows it.

For each call `run_bound`:

1. Reads the HTTP request's headers from `mcp_context` (None under stdio and for in-process `mcp.call_tool`), resolves the calling Session from them (see [Sessions and isolation](sessions-and-isolation.md)) and binds it into a `ContextVar`, so helpers like `_session()` find the right object store without it being passed around. Over HTTP a call without `X-DTCC-Session` is refused with a `ToolError`.
2. Computes an optional *flight key*, then enters the Session's worker share, then the flight lock, and finally runs the body on a worker thread with `anyio.to_thread.run_sync(..., limiter=runtime.workers)`.
3. Always writes one provenance line for the call (`provenance.record_operation`: tool, arguments, result or error, seconds, Session, subject, and the chat turn from `X-DTCC-Turn`), resets the ContextVar and releases the Session's in-flight count, including when the body raises. See [Provenance and evaluation](../operations/provenance-and-evaluation.md).

Why a thread at all: MCPServer would call a sync tool directly on the event loop, where dtcc-core's internal `asyncio.run()` (LiDAR and GeoPackage downloads) raises, and one slow tool would stall every other Session.

## The error contract: return, don't raise

A tool reports a problem the user should hear about as a JSON payload with an `error` key, not as an exception. The chat runtime (pydantic-ai, see [Model runtime](model-runtime.md)) treats a raised tool error as a failed call, retries it once, and then fails the whole chat turn ("Sorry, an error occurred"). #113 found this live: with a wrong Engine token, `get_simulation_schema` raised the Engine's 401 and the turn failed; it and `list_simulations` now return `EngineError` messages like `run_simulation` and `compare_scenarios` do. Reference errors follow the same rule (next section).

## Typed references in tool parameters

Tools that act on stored values take typed references (ADR-0010): `object_ref` (`obj_…`) for an Object in the Session's store and `run_ref` (`run_…`) for a simulation Run. Object tools resolve their argument through one helper, `_object`, and the run tools through `_run_record`. A reference of the other kind is refused with a message naming the tool to use instead, and an unknown one is reported as not found; both come back as a JSON error payload, never as an exception. See [Dispatch and the object store](../concepts/dispatch-and-object-store.md) and [Simulations](../workflows/simulations.md).

## Two capacity limits

| Limiter | Size | Scope |
|---|---|---|
| `runtime.workers` | `DTCC_MCP_WORKERS`, default 4 | all tool bodies in the process |
| `_Session.workers` | `max(1, WORKERS // 2)` for an HTTP Session; the full `WORKERS` for the single local (stdio) Session | one Session |

The per-Session share is acquired **before** the flight lock. A Session waiting for its own share therefore never holds a key another Session needs. `DTCC_MCP_WORKERS` is validated at import: anything that is not a whole number ≥ 1 raises `ValueError` naming the variable, so a misconfigured process never starts. The default is deliberately small: operations may deep-copy heavy inputs, and anyio's default of 40 threads would trade a capacity limit for an OOM kill.

Every tool, `render_object` included, runs on a worker thread under both limiters. Rendering draws with matplotlib's Agg backend, which needs no display or main thread. See [Artifacts and the file boundary](../concepts/artifacts-and-file-boundary.md).

## Single-flight downloads

`_flight(key)` lets one call per key run at a time; later callers wait on an `anyio.Lock` on the event loop, *before* taking a worker, and then usually hit the disk cache the first call filled. A waiting call holds no worker and can be cancelled cleanly; the `_flights` entry is removed when its holder count reaches zero.

Only downloads keyed by their parameters share a flight:

- `run_operation` gets a key only for a `datasets.*` operation that is in `CACHE_ALLOWLIST` and has `bounds`; the key is `(dataset, source or "LM", normalised bounds)`.
- `get_buildings` uses the same `("datasets.buildings", source, bounds)` key, so the direct tool and the generic operation never download one area at once.

Builders get no flight: their results are not cached at all (U2, #62), so a waiting caller would have nothing to read afterwards. Bounds are normalised to floats so `319700` and `319700.0` are the same tile.

## Transports, the secret and startup

`main()` reads `DTCC_MCP_TRANSPORT`:

- **`stdio`** (default): calls `mcp.run()` directly. It does not build the catalogue first, so the first call that needs the catalogue builds it (#22).
- **`http`**: refuses to start when `DTCC_MCP_HOST` is not a loopback address and `DTCC_MCP_SECRET` is unset. It sets `_serving_http = True`, builds `mcp.streamable_http_app(stateless_http=True)`, wraps its lifespan with `_starting_runtime` so `runtime.start()` runs once per process before the app accepts a request (the wrapper passes on whatever state the inner lifespan yields, which Starlette copies into each request), and serves it with uvicorn on `DTCC_MCP_HOST`/`DTCC_MCP_PORT` (default `127.0.0.1:8051`). With a secret, `_require_secret` wraps the app so every HTTP request, `initialize` included, must carry `Authorization: Bearer <secret>` (compared with `hmac.compare_digest`) or gets a bare 401 (T14, #72).
- Anything else exits with a message.

The HTTP transport is stateless because the chatbot opens a new connection per message; a stateful transport would keep a server task per connection that nothing frees. Session identity travels in the `X-DTCC-Session` header instead.

`runtime.start()` first calls `engine.check_config()`, which raises when `DTCC_ENGINE_URL` is set without `DTCC_ENGINE_TOKEN` (#113, see [DTCC Engine integration](../integrations/dtcc-engine.md)). It then calls `registry.build()`, which builds the operation catalogue from the pinned Core (over a second, cold) and starts the background thread that asks dtcc-sim, and prints `dtcc-agent: catalogue built: N operations` to stderr. It never waits on dtcc-sim. A `CatalogueError` from a broken Core install propagates, so the HTTP server exits at startup rather than serving a partial catalogue. Over stdio the same error fails the first call that reads the catalogue. See [Operation catalogue](../concepts/operation-catalogue.md).

## Tests that pin this behaviour

- `tests/test_worker_pool.py`: the pool bound across sessions, the per-Session share, the local Session using the whole pool, bad env values, workers returned after exceptions, and the flight rules (one download per tile, waiting holds no worker, cancellation leaves nothing behind, both ways of fetching buildings share one flight).
- `tests/test_tool_execution.py`: every tool registered async, `asyncio.run()` inside a tool body under uvloop, tools still directly callable.
- `tests/test_catalogue_startup.py`: HTTP builds the catalogue before the app starts and keeps the inner lifespan's state; a broken catalogue stops the HTTP app; stdio serves without building it.
- `tests/test_http_sessions.py`: the session header is required over HTTP; the secret middleware checks every request.
- `tests/test_engine.py`: an Engine URL without a token stops `runtime.start()`; Engine errors come back from all four simulation tools as error payloads.
- `tests/test_server_import.py`: import smoke tests; guards the `mcp>=2.3,<3` range, since the next mcp major could remove `MCPServer` the way 2.x removed `mcp.server.fastmcp`.
