---
type: architecture
title: MCP server and tool execution
description: How dtcc-agent registers its MCP tools, runs each call in a bounded worker pool bound to a Session, deduplicates concurrent downloads, and starts over stdio or stateless streamable-http.
tags: [mcp, server, concurrency, transport, startup]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-30T14:41:08.402Z
sources:
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
  - id: openwiki-source-a2a4605dd979138870606faf
    resource: repo://tests/test_tool_execution.py
  - id: openwiki-source-fa7af1493a897412d61af4b0
    resource: repo://tests/test_worker_pool.py
generated: { by: "claude-code", at: "2026-09-30T14:41:08.402Z" }
---

# MCP server and tool execution

`dtcc_agent/server.py` is the MCP server. It owns one `FastMCP("dtcc-agent", stateless_http=True)` instance, every tool definition, and the execution wrapper that decides *where* and *how many* tool bodies run at once. `dtcc_agent/runtime.py` owns the process-scoped pieces that must exist before the first call: the worker pool and the catalogue build.

Entry point: `python -m dtcc_agent` → `dtcc_agent/__main__.py` → `server.main()`.

## The `@tool` decorator

Every tool is a plain synchronous function decorated with `@tool` (or `@tool(...)`). The decorator does not hand the function itself to FastMCP; it registers an async wrapper, `run_bound`, and returns the original function unchanged so in-process callers (tests, the chatbot in direct mode) can still call it synchronously.

For each call `run_bound`:

1. Resolves the calling Session from the HTTP request (see [Sessions and isolation](sessions-and-isolation.md)) and binds it into a `ContextVar`, so helpers like `_session()` find the right object store without it being passed around.
2. If the tool is `main_thread=True`, calls it directly on the event loop.
3. Otherwise computes an optional *flight key*, then enters the Session's worker share, then the flight lock, and finally runs the body on a worker thread with `anyio.to_thread.run_sync(..., limiter=runtime.workers)`.
4. Always resets the ContextVar and releases the Session's in-flight count, including when the body raises.

Why a thread at all: FastMCP would call a sync tool directly on the event loop, where dtcc-core's internal `asyncio.run()` (LiDAR and GeoPackage downloads) raises, and one slow tool would stall every other Session.

## Typed references in tool parameters

Tools that act on stored values take typed references (ADR-0010): `object_ref` (`obj_…`) for an Object in the Session's store and `run_ref` (`run_…`) for a simulation Run. Object tools resolve their argument through one helper, `_object`, and the run tools through `_run_record`. A reference of the other kind is refused with a message naming the tool to use instead, and an unknown one is reported as not found; both come back as a JSON error payload, never as an exception. See [Dispatch and the object store](../concepts/dispatch-and-object-store.md) and [Simulations](../workflows/simulations.md).

## Two capacity limits

| Limiter | Size | Scope |
|---|---|---|
| `runtime.workers` | `DTCC_MCP_WORKERS`, default 4 | all tool bodies in the process |
| `_Session.workers` | `max(1, WORKERS // 2)` for an HTTP Session; the full `WORKERS` for the single local (stdio) Session | one Session |

The per-Session share is acquired **before** the flight lock. A Session waiting for its own share therefore never holds a key another Session needs. `DTCC_MCP_WORKERS` is validated at import: anything that is not a whole number ≥ 1 raises `ValueError` naming the variable, so a misconfigured process never starts. The default is deliberately small: operations may deep-copy heavy inputs, and anyio's default of 40 threads would trade a capacity limit for an OOM kill.

`render_object` is `main_thread=True`: GLFW must create its window on the main thread (on macOS anywhere else aborts the process), so it runs on the event loop and does not count against either limiter.

## Single-flight downloads

`_flight(key)` lets one call per key run at a time; later callers wait on an `anyio.Lock` on the event loop, *before* taking a worker, and then usually hit the disk cache the first call filled. A waiting call holds no worker and can be cancelled cleanly; the `_flights` entry is removed when its holder count reaches zero.

Only downloads keyed by their parameters share a flight:

- `run_operation` gets a key only for a `datasets.*` operation that is in `CACHE_ALLOWLIST` and has `bounds`; the key is `(dataset, source or "LM", normalised bounds)`.
- `get_buildings` uses the same `("datasets.buildings", source, bounds)` key, so the direct tool and the generic operation never download one area at once.

Builders are excluded on purpose: their cache key fingerprints input objects by metadata, which two different objects can share, so a shared flight could hand one caller another's result. Bounds are normalised to floats so `319700` and `319700.0` are the same tile.

## Transports and startup

`main()` reads `DTCC_MCP_TRANSPORT`:

- **`stdio`** (default): calls `mcp.run()` directly. It does not build the catalogue first: the chatbot starts a stdio server per message and most messages never read it, so the first call that needs the catalogue builds it (#22).
- **`http`**: sets `_serving_http = True`, builds `mcp.streamable_http_app()`, wraps its lifespan with `_starting_runtime` so `runtime.start()` runs once per process before the app accepts a request (the wrapper passes on whatever state the inner lifespan yields, which Starlette copies into each request), and serves it with uvicorn on `DTCC_MCP_HOST`/`DTCC_MCP_PORT` (default `127.0.0.1:8051`).
- Anything else exits with a message.

The HTTP transport is stateless because the chatbot opens a new connection per message; a stateful transport would keep a server task per connection that nothing frees. Session identity travels in the `X-DTCC-Session` header instead.

`runtime.start()` calls `registry.build()`, which builds the operation catalogue from the pinned Core (over a second, cold) and starts the background thread that asks dtcc-sim, then prints `dtcc-agent: catalogue built: N operations` to stderr. It never waits on dtcc-sim. A `CatalogueError` from a broken Core install propagates, so the HTTP server exits at startup (uvicorn exit code 3) rather than serving a partial catalogue. Over stdio the same error fails the first call that reads the catalogue. See [Operation catalogue](../concepts/operation-catalogue.md).

## Tests that pin this behaviour

- `tests/test_worker_pool.py`: the pool bound across sessions, the per-Session share, the local Session using the whole pool, bad env values, main-thread tools running while the pool is full, workers returned after exceptions, and the flight rules (one download per tile, waiting holds no worker, cancellation leaves nothing behind).
- `tests/test_tool_execution.py`: every tool registered async, `render_object` on the main thread, `asyncio.run()` inside a tool body under uvloop, tools still directly callable.
- `tests/test_catalogue_startup.py`: HTTP builds the catalogue before the app starts and keeps the inner lifespan's state; a broken catalogue stops the HTTP app (also checked with a real subprocess); stdio serves without building it.
- `tests/test_server_import.py`: import smoke tests; guards the `mcp<2` upper bound, since mcp 2.x removed `mcp.server.fastmcp`.
