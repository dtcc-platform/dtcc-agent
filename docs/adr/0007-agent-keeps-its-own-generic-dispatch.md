---
status: accepted
decided: 2026-09-19 by Spiros, after measuring the DTCC Engine
---

# The agent keeps its own generic dispatch over Core and Sim

`dtcc-agent` keeps `registry.py`, `dispatcher.py`, `runner.py` and `serializers.py`. It does not
become a client of the DTCC Engine, and it does not offer its reflection layer upstream as the
Engine's discovery layer. The duplication between the two is accepted, deliberately, and written
down here so it is a decision rather than an oversight.

## The duplication is real

The Engine's specification asks, three times, for exactly what `registry.py` already does:

- `docs/dtcc-engine-backend-design-v1.md:30-32` — "Generic discovery and invocation of all Core and
  Sim Dataset Definitions … **There is no manually curated list of supported Dataset names.**"
- `:210-213` — "**Engine must not maintain hand-written copies of Dataset argument models.**"
- `:437` (acceptance) — "adding a conforming definition becomes visible after refresh without
  Dataset-specific Engine code."

It names the same upstream seams this repo reflects over (`:375-380`): Core's
`dtcc_core/datasets/registry.py`, `DatasetDescriptor`, and Sim's `dtcc_sim/datasets.py`.

And the Engine **has no code**. `dtcc-twin` at `develop` holds two design documents and an empty
`dtcc-engine/` directory; no Python, no pull requests ever opened, last push 2026-09-06. So this
is not two implementations diverging. It is one implementation, running and tested, and one
specification of the same idea.

## Why we keep it anyway

**The decision was already made, on the 2026-09-15 call**: the MCP server is a self-contained
native module running *in parallel with* the Engine, not through the Twin API, confirmed with
"σωστά σωστά" (14:16-14:52). Reopening it needs a better reason than symmetry.

**The technical reason is the cache, and it is decisive.** `DiskCache.dataset_lookup`
(`disk_cache.py:156`) does not require an exact parameter match. It finds any cached entry whose
bounds *contain* the request, scores candidates by area to pick the least cropping, and crops the
result down. One download of Gothenburg answers every neighbourhood inside it.

That mechanism needs the cached result **as a live Python object**, because cropping is an
operation on geometry. The Engine's contract is asynchronous job submission returning a
`.dtccpkg` archive (`:196-203`). Behind that contract you can cache archives keyed on exact
parameters, and nothing else — the containment-and-crop trick is not expressible. Moving dispatch
upstream would cost the best-performing mechanism in the repo.

**The execution models differ too.** The Engine is submit → poll → download. The agent is
synchronous and in-process, inside a loop where a person is waiting for an answer. Adopting job
polling would add latency to the one place that cannot afford it.

**The output contracts differ.** `serializers.py` exists because a language model has a context
window: a `PointCloud` becomes `{type, num_points, bounds, classification_counts, z_stats}` —
a few hundred bytes standing in for tens of megabytes, and the module docstring is explicit that
it "never dumps raw arrays". The Engine's consumers are Atlas and Table, which want archives and
job identifiers and have no such constraint.

## Consequences, stated plainly

- **We own tracking Core's churn, alone.** 71 Core commits between `5cf56fa` and `18eb176` moved
  the catalogue from 135 operations to 133 with no notification to anyone. Nobody else will
  notice that for us. This is the strongest argument *against* this decision and it is real.
  Redesign item 9 — catalogue as a versioned artifact rather than import-time reflection — is the
  mitigation, because it makes the drift diffable instead of silent.
- **The nine-field descriptor gap stays ours to close** (ADR-0002), and closing it is not
  something the Engine will do on our behalf.
- **A migration may arrive later, not on our schedule.** If the Engine is built and the platform
  wants one dispatch layer, that conversation happens then, with a working reference to point at.
  This ADR is what gets cited in it.
- **The overlap should be said out loud once**, rather than discovered. One sentence to Vasilis,
  not a pull request: the agent implements the Engine's generic-dispatch requirement today, here
  is what it does and does not cover. That is courtesy and an early warning, not a proposal.

## What was rejected

**Offering `registry.py` upstream as the Engine's discovery layer.** Genuinely attractive: the
cost of aligning is at its floor while there is no Engine code and no implementation plan, and it
would move the Core-churn burden onto the component whose job it is. Rejected because the cache
argument above is concrete and the alignment benefit is speculative — the Engine has had no
commits in two weeks, so coupling to it means coupling to that pace, and because it reopens a call
made and confirmed on a call.

**Waiting to become an Engine client.** Rejected: it blocks this rebuild on a repository with no
implementation and no plan.
