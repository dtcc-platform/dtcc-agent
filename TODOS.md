# TODOS

Deferred work with enough context to pick up cold. Larger open *decisions* live in
`docs/plans/2026-09-19-rebuild-plan.md` under "Deferred decisions register"; this file
is for work whose shape is already known.

---

## T-001 — Make `content_fingerprint` hash contents, so builder caches can be shared again

**What.** Replace the metadata-only fingerprint in `disk_cache.py:40-53` with one derived
from the object's actual contents.

**Why.** The eng review of 2026-09-19 made four `CACHE_ALLOWLIST` entries session-local —
`builder.build_terrain_surface_mesh`, `builder.build_city_surface_mesh`,
`builder.raster.slope_aspect`, `builder.pc_filter.classification_filter` — because their
cache key hashes only `type`, `source_op`, `nbytes` and `label`, never the contents. Two
sessions whose inputs share those four attributes collide and can be served each other's
derived results. Making them session-local closed that, and cost cross-session reuse of
the most expensive derived geometry in the system. Fixing the fingerprint is the only
safe path back to sharing them.

**Pros.** Recovers the largest remaining cache win. Makes the function's name true.
Removes a collision class that can also serve a session its own wrong result.

**Cons.** Hashing a multi-gigabyte point cloud on every lookup is its own performance
problem — which is exactly why the metadata shortcut exists. Probably needs a cheap
structural digest rather than a full content hash, and getting that right is real work.

**Where to start.** `content_fingerprint` is called from `canonical_params_hash` to key
builder operations whose inputs are Objects referenced by transient ids. `test_disk_cache.py`
has 21 tests covering containment, TTL and eviction, so there is a safety net. **Measure
first**: time a full hash of a realistic Gothenburg point cloud before assuming it is too
slow. The assumption that it is has never been tested.

**Depends on / blocked by.** Nothing. Unblocks re-sharing the four builder entries.

---

## T-002 — Cancellation and backpressure for abandoned sessions

**What.** When a session disconnects, stop or abandon the work running for it. When the
worker pool's queue is full, return a clear too-busy response rather than making the
caller wait indefinitely.

**Why.** Milestone 1 introduces a bounded worker pool, which fixes the memory ceiling but
says nothing about work whose requester has gone. The common pattern is someone asking for
a terrain build, waiting, getting bored, and closing the tab. That computation runs to
completion either way — but once the pool is bounded it is *blocking someone else's
request*, not merely wasting a core. The bounded pool makes abandonment more expensive
than it is today, so this gets more valuable the moment M1 ships, not less.

**Pros.** Recovers worker capacity from abandoned work, the most common way capacity is
wasted. A clear too-busy message beats an unexplained wait. Makes queue depth observable,
which the M2 harness can then measure.

**Cons.** Native computations frequently cannot be interrupted, so honest cancellation may
mean discarding the result rather than stopping the work — which recovers memory only once
the job finishes. Partial benefit for real effort.

**Where to start.** Find out whether `dtcc_core` operations can be interrupted at all.
That answer determines whether this is cancellation or merely result-discarding, and it
changes the design completely. The MCP transport already signals disconnection, so the
trigger exists. There is no concurrency machinery in the repo before M1.

**Depends on / blocked by.** Unblocked: M1a/T8 added the bounded worker pool
(`_workers` and each Session's `workers` share in `dtcc_agent/server.py`, sized by
`DTCC_MCP_WORKERS`). Calls waiting for a worker currently wait with no limit or timeout.
