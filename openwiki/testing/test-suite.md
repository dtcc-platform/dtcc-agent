---
type: testing
title: Test suite
description: How dtcc-agent's pytest suite is organised by subsystem, what the environment must provide, which tests pin the critical behaviour (startup, concurrency, isolation), and how to run a focused subset.
tags: [testing, pytest, ci, characterisation]
sources:
  - id: openwiki-source-b69b0be58e9ad19ca644d766
    resource: repo://.github/workflows/ci-build-tests.yml
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-1ce45006eecf563d5acf22dc
    resource: repo://tests/test_buildings_summary.py
  - id: openwiki-source-c0b62da1c8d12500b49cd428
    resource: repo://tests/test_catalogue_startup.py
  - id: openwiki-source-45618922e75f513256096f36
    resource: repo://tests/test_crop.py
  - id: openwiki-source-a4f55aeddd9f309fcd59982a
    resource: repo://tests/test_geocode.py
  - id: openwiki-source-f7a9ed42f310602857a88be4
    resource: repo://tests/test_path_refusal.py
  - id: openwiki-source-2474212d3cebf96cd7d1f586
    resource: repo://tests/test_server.py
  - id: openwiki-source-fa7af1493a897412d61af4b0
    resource: repo://tests/test_worker_pool.py
generated: { by: "claude-code", at: "2026-10-01T20:29:16.810Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-10-01T20:29:16.810Z
---

# Test suite

All tests live in `tests/` as plain pytest modules. There is no `conftest.py`; fixtures are local to each module and monkeypatching is the usual technique. `pyproject.toml` sets `pythonpath = ["."]` and declares one marker, `external`, for tests that need the network (the Nominatim cases in `tests/test_geocode.py`).

## Requirements

- **Python 3.12,** with `uv sync --locked --extra test --extra chatbot`.
- **The `chatbot` extra is mandatory.** Without it, `tests/test_chatbot_app.py` fails at collection (no `fastapi`) and pytest aborts the whole run.
- **The pinned dtcc-core must be installed.** The suite is the contract the `dtcc-core contract` workflow runs against candidate Core revisions, so Core-backed tests are real, not mocked. See [Deployment, configuration and CI](../operations/deployment-and-ci.md).

## Map by subsystem

| Area | Module(s) | What they pin |
|---|---|---|
| Tool surface | `test_server.py`, `test_server_import.py` | Exact set of 22 tools, descriptions, required params, error payloads instead of exceptions; import smoke test guarding `mcp<2` |
| Typed references (ADR-0010) | `test_server.py` | `obj_`/`run_` prefixes; a reference of the wrong kind refused by object tools, `get_run_summary` and `run_operation`; a Run records its result's `object_ref`; an evicted result is reported. These replaced the M0 characterisation tests that pinned the old, untyped behaviour (#57) |
| Tool execution | `test_tool_execution.py` | Every tool registered async; `asyncio.run()` inside a tool under uvloop; tools still directly callable |
| Concurrency | `test_worker_pool.py` | Pool bound across Sessions, per-Session share, env sizing and validation, worker returned on exceptions, single-flight tile and buildings downloads, cancellation |
| Startup and catalogue | `test_catalogue_startup.py` | HTTP builds before serving, stdio on first use; build-once; failing Core sections and unreadable Core datasets stop the build, optional datasets are skipped; the dtcc-sim retrier and merge (never on a reader's thread, 30 s loop, copy-and-swap, non-blocking lock); real HTTP subprocess cases. An autouse fixture clears `DTCC_SIM_SERVICE_URL`/`DTCC_REMOTE_SERVICES` and merge state, and the `dtcc_sim` fixtures work whether or not it is installed |
| Sessions | `test_http_sessions.py`, `test_chatbot_sessions.py` | Isolation across Sessions over real HTTP, header required, LRU cap, in-flight Sessions never evicted, one process memory budget, a dropped Session handing its bytes back; an expired chatbot session deleting its artifacts |
| Catalogue | `test_registry.py`, `test_core_dependency.py` | Reflection and parameter schemas; Core declared, pinned to a full SHA, loud failure without it |
| Dispatch and storage | `test_dispatcher.py`, `test_object_store.py`, `test_serializers.py` | Reference, bounds and enum resolution; tuple storage; summaries |
| Memory (T11) | `test_object_store.py`, `test_memory_budget.py` | Accounting by graph walk (the #13 probe within an order of magnitude, shared arrays once, `.x.array`, slots); LRU within a store and across stores under one budget; per-store cap and oversized objects refused; `not_stored` notes from operations, tuples, simulations and GeoJSON; the Run cap |
| Files (T7, #68) | `test_artifacts.py`, `test_path_refusal.py`, `test_artifact_tools.py`, `test_renderer.py` | Per-Session folders, unguessable names, traversal and cross-Session lookups refused; the exact 18 path-taking Core operations, refused before Core runs; render and export returning artifacts with no paths; building exports as GeoJSON (WGS84), GeoPackage and City JSON; every geometry type rendered by matplotlib |
| Cache | `test_disk_cache.py`, `test_crop.py`, `test_builder_calls.py` | Containment, TTL, budget, hashing; `get_buildings` answering a sub-area from a cached download and every cache failure path; the building crop matching Core's footprint rule with real Core buildings; the cache trust check (private creation under umask 002, refusals for other owners, writable dirs, parents and files, symlinks); two caches sharing one directory; builder calls recorded instead of cached |
| Building summaries | `test_buildings_summary.py` | Heights from Core's estimate then measurement, empty stats as `None`, area over every building, bad bounds refused at every entry point, `BuildingCollection` serialisation |
| Domain helpers | `test_analysis.py`, `test_geocode.py`, `test_geojson_store.py` | Field statistics and comparison; hardcoded and Nominatim geocoding; GeoJSON load and query |
| Chatbot | `test_chatbot_app.py`, `test_chatbot_config.py`, `test_chatbot_memory.py` | App with mocked Chroma and SDK; the `/artifacts` route and image/file frames, including FastMCP-wrapped results; MCP config, header and stdio session env; session-filtered memory |

Test names are sentences describing the behaviour (for example `test_a_session_with_a_tool_in_flight_is_never_evicted`). The repo's review guidance asks that tests prove the behaviour they name without sleep-based timing. `test_worker_pool.py` uses gates and gauges rather than sleeps.

## Running

```sh
uv run --no-sync pytest -q                         # everything (CI)
uv run --no-sync pytest -q -m "not external"       # offline
uv run --no-sync pytest -q tests/test_worker_pool.py
uv run --no-sync pytest -q -k session
```

At #68 (2026-10-01) the suite collected 427 tests. `CHANGELOG.md` tracks the count per merged PR.
