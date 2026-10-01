---
type: concept
title: Disk cache
description: The persistent pickle and JSON-index cache for dataset downloads and builder results. It reuses a containing download cropped with Core's own footprint rule, keys builders by metadata fingerprints, evicts by TTL and size, and is shared across Sessions today.
tags: [cache, disk-cache, datasets, performance, isolation]
sources:
  - id: openwiki-source-011d18639a7a8582693228a4
    resource: repo://dtcc_agent/builder_calls.py
  - id: openwiki-source-612afbd7ed762fbc6635cafc
    resource: repo://dtcc_agent/crop.py
  - id: openwiki-source-052f7c9f16ee5a8169a3fb7d
    resource: repo://dtcc_agent/disk_cache.py
  - id: openwiki-source-88dfce5777b4bb33b4b64cf7
    resource: repo://dtcc_agent/dispatcher.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
  - id: openwiki-source-116206fc13aadd444daea62c
    resource: repo://tests/test_disk_cache.py
generated: { by: "claude-code", at: "2026-10-01T20:29:16.810Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-10-01T20:29:16.810Z
---

# Disk cache

`dtcc_agent/disk_cache.py` keeps dataset downloads across processes and Sessions. It stores each object as a pickle under `<cache_dir>/objects/<cache_id>.pkl`, with metadata in `<cache_dir>/index.json`. One process-wide instance, `_disk_cache = DiskCache()`, lives in `server.py`, and is built when the module is imported, so a cache the trust check refuses stops the process there.

## Configuration

| Setting | Value | Source |
|---|---|---|
| Directory | `DTCC_AGENT_CACHE_DIR`, default `$XDG_CACHE_HOME/dtcc_agent`, usually `~/.cache/dtcc_agent` (Docker sets `/data/cache`) | `CACHE_DIR` |
| TTL | 168 hours (7 days) | `CACHE_TTL_HOURS` |
| Disk budget | 10 GB, oldest first | `CACHE_MAX_SIZE_GB` |
| Version stamp | cache schema `1` plus the installed dtcc-core commit | `CACHE_STAMP` |

`cleanup()` runs once in `DiskCache.__init__`. It deletes every entry that may not be served (see the version stamp below) and then evicts the oldest entries until the total is under budget. Lookups apply the same test, so a long-running process never serves such an entry; its file is only deleted at the next startup.

## Version stamp: a cold start after a Core upgrade

The cache pickles Core objects, and a pickle needs the Core that wrote it. So every entry records `CACHE_STAMP`: a cache `schema` number, bumped when the entry format or cached payloads change, and the installed dtcc-core commit. `_core_build()` reads the commit from pip's `direct_url.json`, the same field the contract workflow checks. For a local or editable install without VCS information it falls back to Core's version, which cannot see local edits.

One function, `_fresh(entry, now)`, decides whether an entry may be served: its stamp equals the current one, its fields parse, its cache id is valid, and it is younger than the TTL. Both lookups and `cleanup()` use it, and an entry missing a field is simply not fresh. So after a Core upgrade, or with an index written before stamps existed, the cache starts cold and startup deletes the old files, instead of unpickling objects from another Core. An `index.json` that is not a JSON list is treated as empty. Two processes on different Core commits sharing one directory never serve each other's entries, and each deletes the other's at startup (#56, T12).

## Trust: who may change the cache

Loading a pickle runs code, so the cache must be one nobody else can change. `_private_dir` enforces that for the cache dir and `objects/` when `DiskCache` is built, and raises `CacheDirError` (the process does not start) otherwise:

- Missing directories are created `0700` one at a time, so another process never sees one open. Pickles, `index.json` and `index.lock` are created `0600` whatever the umask.
- The dir and every entry in it must belong to the current user and not be group- or world-writable. No entry may be a symlink: the cache never makes one, and a link's target could live where others can write.
- Every parent must belong to root or the user, and may be writable by others only if it is sticky, as `/tmp` is. Otherwise whoever can write a parent could swap the whole cache.
- The cache then uses the **real** path it checked, so a symlinked `DTCC_AGENT_CACHE_DIR` swapped later cannot redirect it. `load()` accepts only an 8-hex-digit cache id and opens pickles with `O_NOFOLLOW`.

The old default, `/tmp/dtcc_cache`, was a directory any local user could create first or write (#51).

## Several processes, one cache

Each process keeps the index in memory. Every edit (`store`, `cleanup`) takes an `flock` on `index.lock`, re-reads `index.json`, changes it and writes it back through a temp file and `os.replace`, so one process never drops entries another has added. Lookups reload the index when its (inode, mtime, size) has changed since they last read it; `os.replace` always installs a new inode. A test with eight processes storing at once kept all 1,600 entries.

## What gets cached

Only the two operations in `CACHE_ALLOWLIST` are cached, both downloads keyed by bounds and source: `datasets.point_cloud` and `datasets.buildings`. The `get_buildings` tool has no entry of its own; it shares the `datasets.buildings` download (see below).

The dispatcher caches only single-object results (an `object_ref`). A download too large to keep in the Session's memory (`object_ref: null`, see [Dispatch and the object store](dispatch-and-object-store.md)) is not written either.

**Builder results are not cached** (U2, #11, decided 2026-09-30; #62). Their old key described each input Object by its `type`, `source_op`, `nbytes` and `label` only, and dropped `bounds`, so two different inputs, or one input with two areas, could be served each other's geometry. The builder cache code (`content_fingerprint`, `builder_lookup`) was removed rather than kept unreachable.

## Two lookup strategies

### Datasets: containment and crop

Two helpers in `dispatcher.py` own every dataset read and write: `load_cached_dataset(name, params, cache)` and `store_dataset(name, params, obj, cache)`. Both key an entry with `_dataset_params_hash`, a hash of the non-bounds parameters in which `source` defaults to `LM` as it does in Core, so a call that omits `source` and one that names it share an entry.

`dataset_lookup(operation, source, params_hash, bounds)` matches entries on operation, source and that hash. Among entries whose bounds **contain** the request, it returns the one with the smallest area. `load_cached_dataset` loads it and, if the cached bounds differ from the request, calls `crop.crop_to_bounds`:

- **Objects with a 2-D `points` array** (point clouds) keep only the points inside the bounds.
- **A Core `BuildingCollection`** keeps exactly the buildings a fresh Core download of those bounds would keep. Core's footprint loader keeps a footprint only when it lies wholly inside the bounds shrunk by 2 m, for LM and OSM alike, so the crop applies the same `create_bounds_filter(..., buffer=-2.0, strategy="contains")` test to each building's unsimplified LOD0 outline. Core tests a multi-part footprint whole and then splits it, so parts sharing a source feature id (`objektidentitet` for LM, `osm_id` for OSM; Core's own `Building.id` is random) are kept or dropped together. A building without an outline is dropped, as Core's size filter drops it.
- **Any other type**, including a Core `City` (whose buildings cannot be replaced), comes back unchanged. For a larger cached area that is treated as a **miss**: reusing it whole would answer for the wrong area.

One download of a large area therefore serves every neighbourhood inside it. A live check on Lindholmen (500 m cached, 200 m asked) gave the same buildings from the crop as from a fresh download: 13 of 127 on LM, 12 of 138 on OSM.

**Known gaps** (issue #49): the crop sees only what the cache holds, so it can still keep one building at the edge that a fresh download drops when a multi-part feature's tiny sibling part (under Core's 15 m² filter) crossed the bounds, or when Core repaired a malformed outline after its own bounds test.

### `get_buildings`

The `get_buildings` tool caches the Core building download, not its summary. It calls `load_cached_dataset("datasets.buildings", ...)`, fetches with `runner.fetch_buildings` and `store_dataset` on a miss, and builds the answer with `runner.summarize_buildings` for the bounds asked, applying `max_buildings` there. The dispatcher's `run_operation("datasets.buildings")` reads and writes the same entries.

### Builders: measured, not cached

Instead of caching, the dispatcher records every `builder.*` call through `dtcc_agent/builder_calls.py`: one JSON line per call in `$DTCC_AGENT_LOG_DIR/builder_calls.jsonl` (Docker: `/data/logs`) with the time, operation, whether it succeeded, its duration and a 16-character key a cache shared across Sessions would have matched. The key keeps `bounds` and describes input Objects by the same metadata the old cache used, so repeated keys are an upper bound on the hits a correct cache would get. Only a hash is written, never the parameters. Without a log directory nothing is written, and a failure to record only logs a warning.

A few weeks of real use decides whether provenance keys (`TODOS.md` T-001) are worth building. Builders get no single-flight key for the same reason they get no cache (see [MCP server and tool execution](../architecture/mcp-server-and-tool-execution.md)).

## Failure behaviour

A cache miss, a lookup exception or a write failure never fails the operation. `_check_cache` logs a warning and returns `None`. `_populate_cache` logs and continues. `get_buildings` logs a failed lookup or write and fetches fresh. On a hit the dispatcher stores the loaded object in the calling Session's ObjectStore under the label `(cached)` and adds `cache_hit: true` to the response.

## Tests

- `tests/test_disk_cache.py` covers the version stamp (entries carry it; a restart on a newer Core starts cold and deletes old files; another Core's entries are never served; an unstamped, malformed or non-list index starts cold), store and load, containment preferring the smallest area, TTL expiry, budget eviction, hashing, and `get_buildings` on exact and containing hits, an entry written by the dispatcher without `source`, an uncroppable cached area, and each failure path.
- `tests/test_crop.py` covers point cloud cropping and the building crop with real Core buildings: the 2 m margin, multi-part features grouped by source id, the unsimplified outline, and buildings without an outline.
- The `TestCacheIntegration` cases in `tests/test_dispatcher.py` check that a hit skips the download and that a two-raster operation never touches the cache.
- `tests/test_builder_calls.py` checks that only the downloads are cached, that a builder call writes a record and nothing to the cache, that the key tells two areas apart and matches the same inputs across Sessions, and that nothing is written without a log directory.
