---
type: concept
title: Dispatch, object references and serialization
description: How run_operation resolves parameters, calls a dtcc-core function or dataset, stores the result in the calling Session's ObjectStore under a short ID, and returns an LLM-sized summary, plus the object tools built on that store.
tags: [dispatcher, object-store, serializers, references, pipelines]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-29T13:24:05.367Z
sources:
  - id: openwiki-source-7931b878d950a1ff97af7eb8
    resource: repo://docs/adr/0010-references-are-typed-and-a-run-records-its-object.md
  - id: openwiki-source-88dfce5777b4bb33b4b64cf7
    resource: repo://dtcc_agent/dispatcher.py
  - id: openwiki-source-75161c3a8d68e36736d2343c
    resource: repo://dtcc_agent/object_store.py
  - id: openwiki-source-3cb4a6487d73410befc45a84
    resource: repo://dtcc_agent/serializers.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
  - id: openwiki-source-2474212d3cebf96cd7d1f586
    resource: repo://tests/test_server.py
generated: { by: "claude-code", at: "2026-09-29T13:24:05.367Z" }
---

# Dispatch, object references and serialization

The generic dispatch path lets an LLM chain any catalogued dtcc-core Operation without geometry passing through the model. The pieces are:

- `dispatcher.run_operation`, which executes one Operation;
- `ObjectStore`, which keeps results in memory, one store per Session;
- `serializers`, which turns a result into a summary.

## The `run_operation` flow

The MCP tool `run_operation(name, params, label)` in `server.py` calls `dispatcher.run_operation(name, params, store=_session().objects, cache=_disk_cache)`:

1. **Look up** the `OperationInfo` in the catalogue. An unknown name returns `{"error": ...}`.
2. **Check the disk cache** if the name is in `CACHE_ALLOWLIST`. Datasets go through `load_cached_dataset`, which crops a larger cached area to the request; the `get_buildings` tool uses the same helper. A hit returns immediately with `cache_hit: true` (see [Disk cache](disk-cache.md)).
3. **Call the Operation.**
   - *Datasets* (`_run_dataset`) are called as `ds(**params)`, with a `Bounds` object converted back to a list.
   - *Functions* (`_run_function`) resolve each declared parameter:
     - **Object references.** For a parameter whose type is a dtcc object (`is_object_param`), a string value is looked up in the Session's store. PointCloud, Mesh, VolumeMesh, Raster, City, Terrain, Surface and MultiSurface inputs are **deep-copied** first, so an in-place Core function never mutates a stored object. A string that is not a stored ID passes through unchanged.
     - **Bounds.** A 4- or 6-element list for a `Bounds`-typed parameter or a parameter named `bounds` becomes a `dtcc_core` `Bounds`.
     - **Enums.** Strings for `GeometryType` parameters become the enum member.
     - **Missing parameters.** A missing required parameter returns an error listing every missing name. Omitted optional parameters take the function's own default.
4. **Store and summarise** (`_store_and_summarize`):
   - A **tuple** stores each element separately and returns `result_ids`.
   - A **list** of Building, Tree or Surface is stored as one object.
   - A **primitive or dict** is returned inline and not stored.
   - **Anything else** is stored and returned with its `result_id`.
5. **Populate the disk cache** on success, datasets through `store_dataset`.

Exceptions from Core are caught and returned as `{"error": "Operation '<name>' failed: ..."}`. Tools return error payloads rather than raising, so the LLM can recover.

## ObjectStore

`object_store.ObjectStore` is a thread-safe dict of entries holding `object`, `type`, `source_op`, `label`, `created`, `last_accessed` and `nbytes`.

- **IDs** are `uuid4().hex[:8]`.
- **Size** is estimated by `_estimate_bytes`. It sums the `nbytes` of known numpy attributes (`points`, `vertices`, `faces`, `data` and others) and recurses through `children` and `geometry`, with a minimum of 64 bytes.
- **Eviction.** When the total exceeds `max_bytes`, the least recently *accessed* entries are evicted. `get()` refreshes the access time.
- **Budget.** Each Session has its own store: 256 MiB per HTTP Session, 2 GiB for the local Session. See [Sessions and isolation](../architecture/sessions-and-isolation.md).

## Serialization: summaries, never arrays

`serializers.serialize(obj)` dispatches on exact type to per-type summaries:

- **PointCloud:** count, bounds, classification counts, z statistics.
- **Mesh and VolumeMesh:** vertex, face and cell counts.
- **Raster:** shape, cell size, value statistics.
- **Others:** City, building and tree lists, Terrain, RoadNetwork, dolfinx `Function` (via `analysis.summarize_field`), and GeoJSON FeatureCollection.

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

An unknown object ID returns an error payload, not an exception.

## Reference identity, and where it is heading

Today three identifier spaces all mint eight hex characters: ObjectStore IDs, DiskCache IDs and Run IDs (`str(uuid4())[:8]`). Nothing in a reference says which store it belongs to. `tests/test_server.py` pins this current behaviour: a run id passed to an object tool is *missed, not refused*, and the reverse is also true. A simulation result is also stored twice, as a Run and as an Object, linked only by `label`.

ADR-0010 (accepted 2026-09-19) decides that references carry their kind (`obj_…`, `run_…`) and that a Run records the Object reference it yielded. It lands in the rebuild's first milestone. See [Simulations, runs and geocoding](../workflows/simulations.md).

## Tests

- `tests/test_dispatcher.py` covers bounds and enum resolution, reference resolution, tuple storage and cache integration.
- `tests/test_object_store.py` covers size estimation, LRU eviction and delete.
- `tests/test_serializers.py` covers the per-type summaries.
- `tests/test_server.py` covers the tool surface and the reference-identity characterisation tests.
