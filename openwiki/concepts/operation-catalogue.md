---
type: concept
title: Operation catalogue
description: How registry.py reflects over the pinned dtcc-core into a catalogue of named Operations with parameter schemas, built once per process (HTTP at startup, stdio on first use), failing loudly on a broken Core, with dtcc-sim datasets joining later from a background retrier.
tags: [registry, catalogue, dtcc-core, operations, reflection, dtcc-sim]
sources:
  - id: openwiki-source-b3e643290de65ed93425d581
    resource: repo://.github/workflows/dtcc-core-contract.yml
  - id: openwiki-source-69befafd44aed5a2ceb6450c
    resource: repo://docs/adr/0007-agent-keeps-its-own-generic-dispatch.md
  - id: openwiki-source-4163f0ea9e6726ccca521458
    resource: repo://dtcc_agent/registry.py
  - id: openwiki-source-32c33c58f635fb0708a0e8c6
    resource: repo://dtcc_agent/runner.py
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-c0b62da1c8d12500b49cd428
    resource: repo://tests/test_catalogue_startup.py
  - id: openwiki-source-2eddb37f6f7db2fd16d2a3c2
    resource: repo://tests/test_core_dependency.py
  - id: openwiki-source-f7a9ed42f310602857a88be4
    resource: repo://tests/test_path_refusal.py
generated: { by: "claude-code", at: "2026-10-01T20:29:16.810Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-10-01T20:29:16.810Z
---

# Operation catalogue

`dtcc_agent/registry.py` builds the catalogue behind `list_operations`, `describe_operation` and `run_operation`. It maps an Operation name (`builder.build_terrain_raster`, `datasets.point_cloud`) to an `OperationInfo` holding:

- category and subcategory;
- a one-line description (the first docstring line);
- a list of `ParamInfo`;
- the return type;
- search tags;
- the callable itself.

Nothing is hand-listed except a few meshing and reproject names. Everything else comes from reflecting over dtcc-core, which is why the catalogue follows the pinned Core revision. CI logs the count on every run because it has silently drifted before (135 → 133).

## What gets reflected

`_build_registry()` registers, in order:

| # | Source | Name pattern |
|---|---|---|
| 1 | `dtcc_core.builder.__all__` | `builder.<fn>` |
| 2 | `builder.pointcloud.filter` | `builder.pc_filter.<fn>` |
| 3 | `builder.pointcloud.convert` | `builder.pc_convert.<fn>` |
| 4 | `builder.raster.{analyse,filter,stats,interpolation}` | `builder.raster.<fn>` |
| 5 | `builder.meshing` (10 named functions) | `builder.meshing.<fn>` |
| 6 | `dtcc_core.io.__all__` | `io.<fn>` |
| 7 | `dtcc_core.datasets.registry.list_datasets()` | `datasets.<name>` |
| 8 | `reproject.reproject` (6 named functions) | `reproject.<fn>` |

`_register_functions` skips classes and names already registered, so the first registration of a name wins. A name a module lists (the explicit list, or its `__all__`) but lacks, or that is not callable, raises: it means the pinned Core changed underneath. Names found by the `dir()` fallback are callable by construction.

Step 7 registers whatever is in Core's dataset registry at build time. That registry is shared: the optional `dtcc_sim` package (imported just before, if installed) and remote dtcc-sim services register their datasets in it too. dtcc-sim's remote services are never contacted during the build; see below.

## Parameter schemas

- **Functions.** `_extract_params` reads `inspect.signature` and `get_type_hints`. The type is kept as `str(hint)` so Union members survive. A parameter is an **object reference** (`is_object_ref: true` in `describe_operation`) when its type string names a dtcc model type (PointCloud, Mesh, VolumeMesh, Raster, City, Building, Terrain, Tree, Surface, MultiSurface, RoadNetwork, Bounds or Object). The dispatcher then resolves string IDs for it from the Session's store (see [Dispatch, object references and serialization](dispatch-and-object-store.md)).
- **Datasets.** Parameters come from the dataset's `show_options()` JSON schema, with a required `bounds: list[float]` always first.

`ParamInfo.is_path` marks a file or directory path: the name is `path`, `filename` or `outfile` or ends in `_path` or `_dir`, or the type mentions `Path` or a JSON-schema `format: path`. It is derived from the name and type, so it applies to function and dataset parameters alike. `run_operation` refuses such parameters (U1); at the pinned Core that is 18 operations, pinned by `tests/test_path_refusal.py`. See [Artifacts and the file boundary](artifacts-and-file-boundary.md).

`list_operations(category, search)` filters by category and by a case-insensitive substring over name, description and tags, sorted by name.

## Build once, fail loudly

- `build()` builds the catalogue once per process behind a double-checked lock, so two first callers share one build. The HTTP server calls it at startup through `runtime.start()`, before its first request; stdio and in-process callers reach it through `get_registry()` on first use. See [MCP server and tool execution](../architecture/mcp-server-and-tool-execution.md).
- Every section registered from the pinned Core is fatal. A failure raises `CatalogueError("Failed to register <section>: ...")`: the HTTP server exits at startup, and over stdio the first call that reads the catalogue fails naming the section. A failed build is not cached, so the next caller tries again.
- A Core dataset that can't be read (its `show_options()` or description fails) is fatal too. `_is_core_dataset` tells Core's own datasets from others: the class is defined under `dtcc_core` and is not a `RemoteDatasetDescriptor`.
- Everything else is optional and only warns: a `dtcc_sim` dataset or remote dataset that can't be read is left out, and a `dtcc_sim` package that is installed but fails to import warns while Core's datasets are kept. A `dtcc_sim` that simply isn't installed says nothing.

## dtcc-sim's datasets join on demand

When `DTCC_SIM_SERVICE_URL` (or `DTCC_REMOTE_SERVICES`) is set, its services' datasets are not part of the build, so startup never waits on dtcc-sim and a dtcc-sim that is down at boot does not lose its datasets until a restart.

- **Asking.** `build()` starts one daemon thread, `_retry_remote_services`, if a configured service has not registered. It calls `runner._ensure_remote_services_registered()`, then again every 30 s (`_REMOTE_RETRY_SECONDS`) while any service is down, and exits once every service has answered. It warns once, logs later rounds at DEBUG and logs INFO when dtcc-sim answers; dtcc-core's own discovery warning still repeats each round. A dtcc-sim that is down or answers so slowly that no timeout fires (httpx's timeout is per read) ties up this thread, never a request.
- **Merging.** `get_registry()` calls `_merge_remote_datasets()` on every read. It never calls the network: it starts the retrier if one is needed, then adds the datasets of services the runner has registered since the last merge. It copies the catalogue, registers datasets into the copy and rebinds `_REGISTRY`, so a reader iterating the old dict never sees it change. The merge lock is taken without waiting; a reader that finds a merge in progress reads the catalogue as it is. A failed merge is retried on the next read.
- **Pending lookups.** While a configured service hasn't answered, `get_operation("datasets.<x>")` for a missing name says dtcc-sim hasn't answered yet (and still points at `list_operations()`), instead of a plain not found. Other names keep the plain message.
- **The runner's own paths.** `list_simulations`, `get_simulation_schema` and simulation runs call the runner directly and register services synchronously; the next catalogue read merges them without a network call.

Registration itself is Core's `register_remote_service`, which since the pin move to `bb95f2f` (#59) registers a service all or nothing (dtcc-core#128) and skips, with a warning, any dataset whose name already belongs to a non-remote dataset, so a service cannot replace Core's `point_cloud` or `buildings` (dtcc-core#132). That closed #44 and #45; the agent's own repair step for them was removed. Merged dtcc-sim datasets are still never refreshed after a redeploy: restart the agent after redeploying dtcc-sim (#46).

## The Core pin

`pyproject.toml` pins `dtcc-core` to a full commit SHA (`bb95f2f…` since #59), because dtcc_core exposes no `__version__`. The Docker build asserts the installed Core is that commit (#61). The pin is not moved by hand. The `dtcc-core contract` workflow installs a candidate SHA over the locked environment, verifies the installed commit, runs the whole suite as the contract, and prints the catalogue size. A green run is the signal to move the pin. `tests/test_core_dependency.py` asserts that Core is declared and pinned to a full SHA, and that importing without Core raises with a remedy. See [Deployment, configuration and CI](../operations/deployment-and-ci.md).

## Why the agent keeps its own catalogue

The DTCC Twin design names a *Capability Catalog* of authoritative Dataset Definitions, and the planned DTCC Engine specifies the same generic discovery. ADR-0007 (accepted 2026-09-19) keeps `registry.py`, `dispatcher.py`, `runner.py` and `serializers.py` in this repo anyway. The Engine has no code yet, so the duplication is accepted deliberately. ADR-0002 records the direction of convergence on Twin contracts. The descriptors reflected here carry three fields where the Twin definition calls for nine.

## Tests

- `tests/test_registry.py` covers type detection, parameter extraction, `to_dict`, the categories present, lookup and search.
- `tests/test_catalogue_startup.py` covers build-once, every failing Core section, missing or non-callable listed names, Core vs optional datasets, the `dtcc_sim` import cases, failed builds not being kept, and the dtcc-sim path: the build never calls dtcc-sim, readers never wait even on a hanging dtcc-sim, one retrier, the 30 s loop, copy-and-swap, the non-blocking merge lock, and the pending-lookup message.
