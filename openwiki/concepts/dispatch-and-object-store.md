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
  - id: openwiki-source-2474212d3cebf96cd7d1f586
    resource: repo://tests/test_server.py
generated: { by: "claude-code", at: "2026-09-30T14:41:08.402Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-09-30T14:41:08.402Z
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
3. **Check the disk cache** if the name is in `CACHE_ALLOWLIST`. Datasets go through `load_cached_dataset`, which crops a larger cached area to the request; the `get_buildings` tool uses the same helper. A hit returns immediately with `cache_hit: true` (see [Disk cache](disk-cache.md)).
4. **Call the Operation.**
   - *Datasets* (`_run_dataset`) are called as `ds(**params)`, with a `Bounds` object converted back to a list.
   - *Functions* (`_run_function`) resolve each declared parameter:
     - **Object references.** For a parameter whose type is a dtcc object (`is_object_param`), a string value is looked up in the Session's store. PointCloud, Mesh, VolumeMesh, Raster, City, Terrain, Surface and MultiSurface inputs are **deep-copied** first, so an in-place Core function never mutates a stored object. A string that is not a stored Object reference passes through unchanged.
     - **Bounds.** A 4- or 6-element list for a `Bounds`-typed parameter or a parameter named `bounds` becomes a `dtcc_core` `Bounds`.
     - **Enums.** Strings for `GeometryType` parameters become the enum member.
     - **Missing parameters.** A missing required parameter returns an error listing every missing name. Omitted optional parameters take the function's own default.
5. **Store and summarise** (`_store_and_summarize`):
   - A **tuple** stores each element separately and returns `object_refs`.
   - A **list** of Building, Tree or Surface is stored as one object.
   - A **primitive or dict** is returned inline and not stored.
   - **Anything else** is stored and returned with its `object_ref`.
6. **Populate the disk cache** on success, datasets through `store_dataset`.

Exceptions from Core are caught and returned as `{"error": "Operation '<name>' failed: ..."}`. Tools return error payloads rather than raising, so the LLM can recover.

## ObjectStore

`object_store.ObjectStore` is a thread-safe dict of entries holding `object`, `type`, `source_op`, `label`, `created`, `last_accessed` and `nbytes`.

- **References** are Object references, `obj_` plus 8 hex characters, minted by `refs.new(refs.OBJECT)`. `list()` reports each entry under `object_ref`.
- **Size** is estimated by `_estimate_bytes`. It sums the `nbytes` of known numpy attributes (`points`, `vertices`, `faces`, `data` and others) and recurses through `children` and `geometry`, with a minimum of 64 bytes.
- **Eviction.** When the total exceeds `max_bytes`, the least recently *accessed* entries are evicted. `get()` refreshes the access time.
- **Budget.** Each Session has its own store: 256 MiB per HTTP Session, 2 GiB for the local Session. See [Sessions and isolation](../architecture/sessions-and-isolation.md).

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
| `export_object` | Write an object to a file (CSV, OBJ, PLY, STL, VTK, glTF and more); default `/tmp/dtcc_exports/<id>.<fmt>` |
| `object_to_text` | Markdown description |
| `spatial_query` | Spatial filtering, nearest station, height queries |
| `load_geojson`, `query_geojson`, `summarize_geojson_property` | GeoJSON files as stored FeatureCollections (`geojson_store.py`) |
| `render_object` | Offscreen PNG through dtcc-viewer (`renderer.py`) |

Every tool above that takes an `object_ref` resolves it through one server helper, `_object`: a Run reference is refused as the wrong kind, with a pointer to `get_run_summary`, and an unknown reference returns "not found" with a pointer to `list_objects()`. Both are error payloads, never exceptions. `query_geojson` and `summarize_geojson_property` used to raise `KeyError` on an unknown reference because they checked `ObjectStore.get()` for `None`; they now use the helper too (#57).

## Typed references (ADR-0010)

References carry their kind in the value. `dtcc_agent/refs.py` mints them (`new(kind)`: `obj_` or `run_` plus 8 hex characters) and classifies them (`wrong_kind(ref, expected)` returns an error message when a value has exactly the other kind's shape). Tools take and return `object_ref` and `run_ref`; `run_operation` returns `object_ref` or `object_refs`, no longer `result_id`. A misrouted reference is refused as the wrong kind instead of reported "not found", which used to send the caller to `list_objects()` where a run never appears. Disk-cache ids stay internal and untyped, since they never cross the tool boundary.

A simulation Run records the Object reference of its result and keeps no copy of it: the Object owns the result (U6, decided 2026-09-30). See [Simulations, runs and geocoding](../workflows/simulations.md).

## Tests

- `tests/test_dispatcher.py` covers bounds and enum resolution, reference resolution, tuple storage and cache integration.
- `tests/test_object_store.py` covers size estimation, LRU eviction and delete.
- `tests/test_serializers.py` covers the per-type summaries.
- `tests/test_server.py` covers the tool surface and typed references: the prefixes, refusal of the wrong kind by object tools, `get_run_summary` and `run_operation`, the Run to Object link, an evicted result, and the GeoJSON tools reporting an unknown reference. These replaced the M0 characterisation tests that pinned the old, untyped behaviour.
