---
status: accepted
decided: 2026-09-19 by Spiros
---

# References carry their kind, and a Run records the Object it yielded

Three identifier spaces exist and are indistinguishable: `object_store.py:63` and
`disk_cache.py:122` mint `uuid4().hex[:8]`, `server.py:54` mints `str(uuid4())[:8]`, and all three
are eight hexadecimal characters. The same ObjectStore key is additionally spelled three ways —
`obj_id` internally, `result_id` leaving the dispatcher, `object_id` entering a tool. So nothing
in a reference says which store it belongs to.

**Decided:** references carry their kind in the value (`obj_…`, `run_…`), and the model has exactly
two reference terms — **Object reference** and **Run reference**. A misrouted reference now fails
loudly instead of missing silently.

**And:** a Run yields exactly one Object and **records that Object's reference**. Today
`server.py:52-65` stores one result twice, under a Run reference and a separate Object reference,
linked only by a human-readable `label` that nothing ever queries — and `get_run_summary` returns
no Object reference at all. A person who runs a simulation cannot reach the Object it produced
except by listing every object and reading labels.

## Considered options for the Run/Object relationship

- **A Run *is* an Object with provenance attached** — one entity, one identifier, and `_results`
  is a redundant second store. Rejected: it collapses a distinction the domain needs. "What did I
  run" and "what do I have" are different questions, and a Run is an event while an Object is a
  value.
- **A Run's result should not be in the ObjectStore at all** — the double-store is the bug.
  Rejected: the comment at `server.py:63` records a real requirement, that simulation results feed
  later pipelines. Removing the Object would break the chain a person most wants.
- **Two entities, one explicit link** — chosen. Smallest change, preserves the distinction, and
  makes the link queryable rather than decorative.

## Consequences

**This must happen in milestone 1, or not for a long time.** `object_id` appears 62 times and
every occurrence is in a tool signature the model sees. Renaming while `server.py` is being
replaced is free; renaming afterwards means changing a published tool surface that people and
prompts depend on.

**It is also a precondition for the session boundary.** ADR-0004 scopes state per Session, which
means every reference crossing the tool boundary has to be authorized. A reference that cannot be
classified cannot be authorized, so typed references are load-bearing for isolation rather than
merely tidy.

Cache identity stays internal and never crosses the tool boundary, so it gains no term in the
glossary and no prefix requirement.
