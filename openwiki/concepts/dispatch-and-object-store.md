---
type: concept
title: Dispatch, object references and serialization
description: How run_operation resolves parameters, calls a dtcc-core function or dataset, stores the result in the calling Session's ObjectStore under a short ID, and returns an LLM-sized summary, plus the object tools built on that store.
tags: [dispatcher, object-store, serializers, references, pipelines]
sources:
  - id: openwiki-source-88dfce5777b4bb33b4b64cf7
    resource: repo://dtcc_agent/dispatcher.py
  - id: openwiki-source-75161c3a8d68e36736d2343c
    resource: repo://dtcc_agent/object_store.py
  - id: openwiki-source-d8839a242913c8f59a48c041
    resource: repo://dtcc_agent/refs.py
  - id: openwiki-source-3cb4a6487d73410befc45a84
    resource: repo://dtcc_agent/serializers.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
  - id: openwiki-source-ec47a577dec4bba7e1ffe974
    resource: repo://tests/test_object_store.py
  - id: openwiki-source-2474212d3cebf96cd7d1f586
    resource: repo://tests/test_server.py
generated: { by: "claude-code", at: "2026-10-01T20:29:16.810Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-10-01T20:29:16.810Z
---

# Dispatch, object references and serialization

The generic dispatch path lets an LLM chain any catalogued dtcc-core Operation without geometry passing through the model. The pieces are:

- `dispatcher.run_operation`, which executes one Operation;
- `ObjectStore`, which keeps results in memory, one store per Session;
- `serializers`, which turns a result into a summary.

## The `run_operation` flow

The MCP tool `run_operation(name, params, label)` in `server.py` calls `dispatcher.run_operation(name, params, store=_session().objects, cache=_disk_cache)`:

0. **Refuse a Run reference.** Any parameter value shaped like a Run reference (`run_` plus 8 hex characters) is refused with a message pointing at `get_run_summary`, before anything else runs. Only the exact shape counts, so an ordinary string that merely starts with `run_` passes.
1. **Check literal bounds.** A `bounds` parameter given as a list or tuple must be an area: `bounds_error` refuses the wrong length, non-finite values, or a min not below its max, and the call returns `{"error": "Invalid bounds ..."}`. A stored `Bounds` object id or `None`, which some operations take, passes through to step 3.
2. **Look up** the `OperationInfo` in the catalogue. An unknown name returns `{"error": ...}`.
2a. **Refuse file paths** (U1). If any parameter the catalogue marks `is_path` is required, or is given a value other than `null` or `""`, the call is refused before Core runs. See [Artifacts and the file boundary](artifacts-and-file-boundary.md).
3. **Check the disk cache** if the name is in `CACHE_ALLOWLIST`. Datasets go through `load_cached_dataset`, which crops a larger cached area to the request; the `get_buildings` tool uses the same helper. A hit returns immediately with `cache_hit: true` (see [Disk cache](disk-cache.md)).
4. **Call the Operation.**
   - *Datasets* (`_run_dataset`) are called as `ds(**params)`, with a `Bounds` object converted back to a list.
   - *Functions* (`_run_function`) resolve each declared parameter. A `builder.*` call is timed and recorded to `builder_calls.jsonl` (builders are not cached, see [Disk cache](disk-cache.md)):
     - **Object references.** For a parameter whose type is a dtcc object (`is_object_param`), a string value is looked up in the Session's store. PointCloud, Mesh, VolumeMesh, Raster, City, Terrain, Surface and MultiSurface inputs are **deep-copied** first, so an in-place Core function never mutates a stored object. A string that is not a stored Object reference passes through unchanged.
     - **Bounds.** A 4- or 6-element list for a `Bounds`-typed parameter or a parameter named `bounds` becomes a `dtcc_core` `Bounds`.
     - **Enums.** Strings for `GeometryType` parameters become the enum member.
     - **Missing parameters.** A missing required parameter returns an error listing every missing name. Omitted optional parameters take the function's own default.
5. **Store and summarise** (`_store_and_summarize`):
   - A **tuple** stores each element separately and returns `object_refs`.
   - A **list** of Building, Tree or Surface is stored as one object.
   - A **primitive or dict** is returned inline and not stored.
   - **Anything else** is stored and returned with its `object_ref`.
   - A result the store will not keep (larger than the Session may hold) comes back with `object_ref: null` (or a `null` among `object_refs`) and a `not_stored` note, with its summary intact (U4).
6. **Populate the disk cache** on success, datasets through `store_dataset`.

Exceptions from Core are caught and returned as `{"error": "Operation '<name>' failed: ..."}`. Tools return error payloads rather than raising, so the LLM can recover.

## ObjectStore

`object_store.ObjectStore` is a thread-safe dict of entries holding `object`, `type`, `source_op`, `label`, `created`, `last_accessed` and `nbytes`.

- **References** are Object references, `obj_` plus 8 hex characters, minted by `refs.new(refs.OBJECT)`. `list()` reports each entry under `object_ref`.
- **Size** (U4, #13; T11, #66). `_estimate_bytes` walks everything an object reaches, once each: `sys.getsizeof` per Python object, dict keys and values, list, tuple and set items, `__dict__` and `__slots__` attributes, and a numpy array's buffer (a view counts its base). It also follows a dolfinx Function's `.x.array`, which is a property, and skips modules, classes and functions. On real data it tracks the pickled size: 127 Lindholmen buildings count 6.25 MB (4.73 MB pickled, 16 ms to count) where the old estimator, which read only a few named array attributes, said 64 bytes.
- **Shared budget.** Every Session's store takes the one process-wide `MemoryBudget` (`OBJECT_BUDGET_BYTES`, 2 GiB) and shares its lock. A store first evicts its own least recently used entries past its cap, then the budget evicts the least recently used entry in *any* store until the total is under 2 GiB. Access order is a counter, so there are no ties. An idle Session's memory therefore goes to the Sessions in use.
- **Per-store cap.** An HTTP Session's store may hold `SESSION_OBJECT_BYTES` (1 GiB); the local stdio Session's may use the whole budget. A cap never exceeds its budget.
- **Too large to keep.** `store()` returns `None` and keeps nothing for an object bigger than the store's cap, instead of emptying the Session to make room. Callers add a `not_stored` note via `_kept()`: the dispatcher, cache hits, simulations, and the GeoJSON and spatial tools.
- **Clearing.** `clear()` drops every entry and returns its bytes to the budget; the server calls it when it drops an idle Session.
- The budget covers stored results only. Memory while an operation runs is bounded by the worker count (`DTCC_MCP_WORKERS`). See [Sessions and isolation](../architecture/sessions-and-isolation.md).

## Serialization: summaries, never arrays

`serializers.serialize(obj)` dispatches on exact type to per-type summaries:

- **PointCloud:** count, bounds, classification counts, z statistics.
- **Mesh and VolumeMesh:** vertex, face and cell counts.
- **Raster:** shape, cell size, value statistics.
- **Buildings:** a `BuildingCollection` (what `datasets.buildings` returns), a City and a building list report the count and height statistics. A height is Core's estimate, else its measurement (`building_height`); `Building.height` alone is only the measurement, which downloads leave empty.
- **Others:** tree lists, Terrain, RoadNetwork, dolfinx `Function` (via `analysis.summarize_field`), and GeoJSON FeatureCollection.

Generic lists show the first 10 elements. Numpy arrays are reduced to shape, dtype and statistics. `to_markdown(obj)` backs the richer `object_to_text` tool.

## Tools over the store

| Tool | What it does |
|---|---|
| `list_objects`, `inspect_object`, `delete_object` | Browse, summarise and free stored objects |
| `get_field_names` | Discover fields and data attached to an object |
| `export_object` | Write an object to a file in the Session's artifact folder for download (CSV, OBJ, PLY, STL, VTK, glTF; buildings as GeoJSON, GeoPackage or City JSON) |
| `object_to_text` | Markdown description |
| `spatial_query` | Spatial filtering, nearest station, height queries |
| `load_geojson`, `query_geojson`, `summarize_geojson_property` | GeoJSON files from `SHARED_RESULTS_DIR` as stored FeatureCollections (`geojson_store.py`) |
| `render_object` | A PNG in the Session's artifact folder, drawn with matplotlib (`renderer.py`) |

Every tool above that takes an `object_ref` resolves it through one server helper, `_object`: a Run reference is refused as the wrong kind, with a pointer to `get_run_summary`, and an unknown reference returns "not found" with a pointer to `list_objects()`. Both are error payloads, never exceptions. `query_geojson` and `summarize_geojson_property` used to raise `KeyError` on an unknown reference because they checked `ObjectStore.get()` for `None`; they now use the helper too (#57).

## Typed references (ADR-0010)

References carry their kind in the value. `dtcc_agent/refs.py` mints them (`new(kind)`: `obj_` or `run_` plus 8 hex characters) and classifies them (`wrong_kind(ref, expected)` returns an error message when a value has exactly the other kind's shape). Tools take and return `object_ref` and `run_ref`; `run_operation` returns `object_ref` or `object_refs`, no longer `result_id`. A misrouted reference is refused as the wrong kind instead of reported "not found", which used to send the caller to `list_objects()` where a run never appears. Disk-cache ids stay internal and untyped, since they never cross the tool boundary.

A simulation Run records the Object reference of its result and keeps no copy of it: the Object owns the result (U6, decided 2026-09-30). See [Simulations, runs and geocoding](../workflows/simulations.md).

## Tests

- `tests/test_dispatcher.py` covers bounds and enum resolution, reference resolution, tuple storage and cache integration.
- `tests/test_object_store.py` covers size estimation (the #13 GeoJSON probe within an order of magnitude, shared arrays counted once, `.x.array`, single-string slots), LRU eviction, delete, the shared budget across stores, the per-store cap, oversized objects and `clear()`.
- `tests/test_memory_budget.py` covers oversized results from operations, tuples, simulations and GeoJSON, and the Run cap.
- `tests/test_serializers.py` covers the per-type summaries.
- `tests/test_server.py` covers the tool surface and typed references: the prefixes, refusal of the wrong kind by object tools, `get_run_summary` and `run_operation`, the Run to Object link, an evicted result, and the GeoJSON tools reporting an unknown reference. These replaced the M0 characterisation tests that pinned the old, untyped behaviour.
