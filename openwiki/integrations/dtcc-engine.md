---
type: integration
title: DTCC Engine integration
description: How dtcc-agent runs simulations as DTCC Engine jobs when DTCC_ENGINE_URL is set. It submits, polls with backoff, downloads the result as a .dtccpkg read by Core, maps every Engine answer to a message the model can relay, and replaces the dtcc-sim path and its shared results folder with no fallback.
tags: [engine, simulations, http, integration, dtcc-twin]
verified:
  - by: openwiki/0.5.2
    at: 2026-10-13T19:36:58.526Z
sources:
  - id: openwiki-source-f8694fac7020c12d5f38c343
    resource: repo://dtcc_agent/engine.py
  - id: openwiki-source-32c33c58f635fb0708a0e8c6
    resource: repo://dtcc_agent/runner.py
  - id: openwiki-source-e23a39a8942c83ff4b942657
    resource: repo://dtcc_agent/runtime.py
  - id: openwiki-source-0e41257659137c7a2c27de1c
    resource: repo://tests/test_engine.py
generated: { by: "claude-code", at: "2026-10-13T19:36:58.526Z" }
---

# DTCC Engine integration

The DTCC Engine (`dtcc-twin` `apps/engine`) is the platform's headless job service: it runs registered Core and Sim datasets as asynchronous jobs behind one bearer token and returns each result as a canonical package. On 2026-10-09 the platform agreed that all queuing goes through it. Since #113 (M4/T40, #109) dtcc-agent sends **simulations** there when it is configured. `dtcc_agent/engine.py` is the client; `dtcc_agent/runner.py` chooses the path.

## Configuration and the path choice

| Variable | Default | Effect |
|---|---|---|
| `DTCC_ENGINE_URL` | unset | Set: simulations run as Engine jobs |
| `DTCC_ENGINE_TOKEN` | unset | Bearer token; required with the URL |
| `DTCC_ENGINE_TIMEOUT` | `900` | Seconds a simulation may take before the tool reports a timeout |

`runner.run()`, `list_simulations()` and `get_schema()` use **exactly one** path, chosen by configuration, with no fallback between them, because a silent fallback would hide a broken Engine:

1. **The Engine**, when `DTCC_ENGINE_URL` is set.
2. **The dtcc-sim mini-service**, when `DTCC_REMOTE_SERVICES` or `DTCC_SIM_SERVICE_URL` is set. Core's remote protocol submits over HTTP but reads the result file from a folder both services mount (`SHARED_RESULTS_DIR`), so agent and sim must share a disk.
3. **In process**, when neither is set.

While the Engine is configured, `runner._remote_services()` returns nothing, so dtcc-sim is neither asked nor merged into the catalogue (see [Operation catalogue](../concepts/operation-catalogue.md)). `runtime.start()` calls `engine.check_config()` first: a URL without a token stops the tool server at startup, and `engine.client()` makes the same check under stdio.

Only `urban_heat_simulation` and `air_quality_field`, the agent's simulation set, are offered through the Engine. Buildings, terrain, meshing, rendering and every other operation still run in the agent; whether Core downloads also move to the Engine is measured in #110.

## The job lifecycle

`Engine.run(dataset, parameters)` returns `(model, job_id, package_size)`:

1. **Submit.** `POST /api/v1/jobs` with `{"dataset", "parameters"}`. The Engine always returns a package and refuses a `format` parameter, so any `format` the model passed is dropped. A 202 returns the job id.
2. **Poll.** `GET /api/v1/jobs/{id}` after 1 s, doubling to every 5 s, until the job is `completed`, it `failed`, or `DTCC_ENGINE_TIMEOUT` passes. A 503 (job state briefly unknown) is waited out.
3. **Download.** `GET /api/v1/jobs/{id}/package` is streamed to a temporary file, refused past 2 GiB (`MAX_PACKAGE_BYTES`, since Core asks callers of untrusted packages for a byte budget and no Session could keep more), and read with `dtcc_core.datasets.load_model_package`. The temporary file is deleted afterwards.

The runner wraps the model in a `RemoteSimulationResult` with `task_id` set to the job id, `output_format="dtccpkg"` and `model` set to the package's `VolumeMesh`, so the field statistics path of `run_simulation` and `compare_scenarios` is the same as for dtcc-sim (see [Simulations and task tools](../workflows/simulations.md)). `as_dict()` leaves the model out of the JSON the model sees.

The discovery calls reuse the same client: `datasets()` reads `GET /api/v1/datasets` once per process, `list_simulations` takes each simulation's `description`, and `get_simulation_schema` returns its `describe()["args_schema"]`.

## Errors the user can hear

Every problem becomes an `EngineError` whose message the tool returns as `{"error": ...}`. Tools return it instead of raising: a raised tool error fails the whole chat turn (see [MCP server and tool execution](../architecture/mcp-server-and-tool-execution.md)).

| Engine answer | Message, in short |
|---|---|
| 401 on any call | The Engine rejected the agent's token (`DTCC_ENGINE_TOKEN`); tell the administrator |
| 422 on submit | The Engine refused the parameters, with the Engine's `detail` so the model can correct them |
| 503 on submit with a `job_id` | Not an error: the queue did not confirm the job, which may still run, so it is **polled, never resubmitted** |
| 503 on submit without one | The job queue is unavailable; try again shortly |
| `failed` | The job failed, with its error type |
| 404 while polling | The job is unknown or expired; run it again |
| Timeout | The job did not finish within the timeout and may still be running, naming the job |
| Connection error | The Engine at its URL is unreachable |

## Verified behaviour

Live, against a local Engine built from dtcc-twin `22d0624` with dtcc-sim stopped and no shared folder mounted, the heatwave question answered twice with full temperature tables: 4 jobs of about 30 s each, 4 package downloads, 42 status polls. With a wrong token, the chat answered that the Engine rejected the agent's token and an administrator must fix it, and still answered the part that did not need the Engine.

## Tests

`tests/test_engine.py` runs offline against `httpx.MockTransport`: submit once with backoff and `format` dropped, every error row above, waiting out a 503, the timeout, an unreachable Engine, the startup check, dtcc-sim ignored while the Engine is set, discovery from the Engine, the Engine path chosen even with dtcc-sim configured, a real canonical package written by Core (as the Engine's worker writes one) decoded into `run_simulation`'s statistics, and a rejected token relayed by all four simulation tools.

See also [Deployment and CI](../operations/deployment-and-ci.md) for the Compose variables.
