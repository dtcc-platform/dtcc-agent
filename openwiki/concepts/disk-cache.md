---
type: concept
title: Disk cache
description: The persistent pickle and JSON-index cache for dataset downloads and builder results. It reuses containing bounds and crops, keys builders by metadata fingerprints, evicts by TTL and size, and is shared across Sessions today.
tags: [cache, disk-cache, datasets, performance, isolation]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-26T09:41:09.647Z
sources:
  - id: openwiki-source-612afbd7ed762fbc6635cafc
    resource: repo://dtcc_agent/crop.py
  - id: openwiki-source-052f7c9f16ee5a8169a3fb7d
    resource: repo://dtcc_agent/disk_cache.py
  - id: openwiki-source-88dfce5777b4bb33b4b64cf7
    resource: repo://dtcc_agent/dispatcher.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
generated: { by: "claude-code", at: "2026-09-26T09:41:09.647Z" }
---

# Disk cache

`dtcc_agent/disk_cache.py` keeps expensive results across processes and Sessions. It stores each object as a pickle under `<cache_dir>/objects/<cache_id>.pkl`, with metadata in `<cache_dir>/index.json`. One process-wide instance, `_disk_cache = DiskCache()`, lives in `server.py`. A `threading.Lock` guards the in-memory index and every index write.

## Configuration

| Setting | Value | Source |
|---|---|---|
| Directory | `DTCC_AGENT_CACHE_DIR`, default `/tmp/dtcc_cache` (Docker sets `/data/cache`) | `CACHE_DIR` |
| TTL | 168 hours (7 days) | `CACHE_TTL_HOURS` |
| Disk budget | 10 GB, oldest first | `CACHE_MAX_SIZE_GB` |

`cleanup()` runs once in `DiskCache.__init__`. It deletes expired entries and then evicts the oldest entries until the total is under budget. Lookups also skip expired entries, so a long-running process never serves one; their files are only deleted at the next startup.

## What gets cached

Only operations in `CACHE_ALLOWLIST` are cached:

- **Downloads, keyed by bounds and source:** `datasets.point_cloud`, `datasets.buildings` and `get_buildings`.
- **Builders over stored objects:** `builder.build_terrain_raster`, `builder.build_terrain_surface_mesh`, `builder.build_city_surface_mesh`, `builder.raster.slope_aspect` and `builder.pc_filter.classification_filter`.

The dispatcher caches only single-object results (`result_id`). Tuple results such as `slope_aspect`'s `(slope, aspect)` are not written, although the operation is allowlisted.

## Two lookup strategies

### Datasets: containment and crop

`dataset_lookup(operation, source, params_hash, bounds)` matches entries on operation, source and a hash of the non-bounds parameters. Among entries whose bounds **contain** the request, it returns the one with the smallest area. The dispatcher (`_check_cache_dataset`) loads it and, if the cached bounds differ from the request, calls `crop.crop_to_bounds`:

- **Objects with a 2-D `points` array** (point clouds) keep only the points inside the bounds.
- **Objects with a `buildings` list** (a City) keep only the buildings whose footprint centroid falls inside.
- **Any other type** is returned uncropped.

One download of a large area therefore serves every neighbourhood inside it.

`get_buildings` (the hardcoded tool) calls `dataset_lookup` directly, but on a hit it **does not crop**: it returns the cached summary dict with only its `bounds` field replaced. A hit from a larger area therefore reports that area's buildings.

### Builders: exact hash over fingerprints

`_check_cache_builder` replaces each object-ref parameter with `content_fingerprint(metadata)` and hashes `{op, params}` with `canonical_params_hash` (bounds removed), then calls `builder_lookup` for an exact match.

**Known collision risk.** Despite the name, `content_fingerprint` hashes only `type`, `source_op`, `nbytes` and `label` from the ObjectStore metadata. It never reads the object's contents. Two different inputs that share those four attributes collide, and can be served each other's derived results. ADR-0004 therefore says builder entries must be session-local. Code does not do that yet: the cache has no Session in its keys. Content hashing is tracked as `TODOS.md` T-001. For the same reason, builders are never given a single-flight key (see [MCP server and tool execution](../architecture/mcp-server-and-tool-execution.md)).

## Failure behaviour

A cache miss, a lookup exception or a write failure never fails the operation. `_check_cache` logs a warning and returns `None`. `_populate_cache` logs and continues. `get_buildings` swallows cache errors and fetches fresh. On a hit the dispatcher stores the loaded object in the calling Session's ObjectStore under the label `(cached)` and adds `cache_hit: true` to the response.

## Tests

- `tests/test_disk_cache.py` covers store and load, containment preferring the smallest area, TTL expiry, budget eviction, hashing, and `get_buildings` hits.
- `tests/test_crop.py` covers point cloud and City cropping.
- The `TestCacheIntegration` cases in `tests/test_dispatcher.py` check that a hit skips the download.
