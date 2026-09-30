# TODOS

Deferred work with enough context to pick up cold. Larger open *decisions* live in
`docs/plans/2026-09-19-rebuild-plan.md` under "Deferred decisions register"; this file
is for work whose shape is already known.

---

## T-001 — Provenance keys, so builder results can be cached (and shared) correctly

**What.** Key a builder result by how its inputs were made, not by what they look like.
Each stored Object records its provenance: a download its operation, parameters (bounds
included), source and Core commit; a builder result its operation, parameters and its
inputs' provenance. The cache key is a hash of that. Objects with no reproducible origin
(loaded files, GeoJSON, filter-tool output, simulations) get none and are never cached.

**Why.** U2 (#11), decided 2026-09-30, switched builder caching off: the old key described
an input by `type`, `source_op`, `nbytes` and `label` only, so different inputs collided,
and it dropped `bounds`, so different areas did. A provenance key is exact without hashing
multi-gigabyte point clouds, survives restarts, and is safe to share across Sessions,
since equal provenance means equal public inputs. That also makes most of T6's cache split
unnecessary.

**Pros.** Recovers builder caching, the most expensive derived geometry in the system, for
everyone. No content hashing.

**Cons.** About 2-3 days: a provenance field set by the dispatcher (downloads, cache hits
and crops, builders), the key, and tests. It relies on Core builders being deterministic
and stored Objects never being mutated (heavy inputs are deep-copied already). A cropped
cached download carries #49's rare edge difference into anything built from it.

**Where to start. Measure first.** `dtcc_agent/builder_calls.py` records every builder call
with its duration and the key the old cache would have matched (bounds kept). Count repeated
keys and the seconds they cost in `builder_calls.jsonl` from real use. Equal keys are an
upper bound on the hits a provenance cache would get. Build this only if that number is
worth it.

**Depends on / blocked by.** Usage data from `builder_calls.jsonl`.

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
(`runtime.workers` in `dtcc_agent/runtime.py` and each Session's `workers` share in
`dtcc_agent/server.py`, sized by `DTCC_MCP_WORKERS`). Calls waiting for a worker currently wait with no limit or timeout.
