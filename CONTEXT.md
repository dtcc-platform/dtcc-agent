# dtcc-agent — domain glossary

The shared language for this repo. Glossary only: no implementation detail, no decisions.
Decisions live in `docs/adr/`.

Where a term is also defined by `dtcc-twin/DESIGN.md`, that definition wins and is marked
**(Twin)**. Where this repo currently uses a different word for the same thing, the conflict
is recorded rather than hidden.

## Core concepts

**Operation**
A single named capability contributed by DTCC Core or DTCC Sim — `builder.build_terrain_raster`,
`datasets.point_cloud`. The unit the catalogue enumerates and the unit a caller invokes. An
Operation is not a task: it is a step.

**Task tool**
A capability shaped like something a person asks for — *build a twin here*, *compare these two
scenarios* — as opposed to an Operation, which is shaped like something the platform can do.
Task tools are a curated surface over Operations. Today none exist; the distinction is named
here because the shape of that curated surface is not yet decided: the seven Operations the
system prompt currently hardcodes are moving to versioned configuration as-is, and the Task tools
proper will be derived from Scenarios rather than invented.

**Object**
A value produced by an Operation and held for later reference — a point cloud, a raster, a mesh,
a building collection. Objects are referred to by an **Object reference**, never passed by value.

**Object reference**
The identifier that names a stored Object. Carries its kind, so a reference can be classified
without consulting a store.
_Avoid_: object_id, result_id, obj_id.

**Run**
One execution of a simulation, together with the parameters it was given. Distinct from an
Object: a Run is an event, an Object is a value. **A Run yields exactly one Object and records
its Object reference** — the two are separate entities with an explicit link, not one entity
under two names.

**Run reference**
The identifier that names a Run. A Run reference and an Object reference are never
interchangeable.
_Avoid_: run_id.

> **Conflict with the code, not yet resolved there.** Four names are in use across three
> concepts, and the three are indistinguishable by format — `object_store.py:63` and
> `disk_cache.py:122` both mint `uuid4().hex[:8]`, `server.py:54` mints `str(uuid4())[:8]`, and
> all three are eight hexadecimal characters. So nothing in a reference says which store it
> belongs to. The Run/Object link is worse: `server.py:52-65` stores one result twice, under a
> Run reference and a separate Object reference, linked only by a human-readable `label` that
> nothing ever queries, and `get_run_summary` returns no Object reference at all. Both are
> resolved in the rebuild's first milestone, while the surface is being replaced anyway.

**Field**
Named values defined on a geometry — a temperature, a wind speed. A Field is what makes a
simulation result meaningful, and is the thing most easily lost in conversion.
_Avoid_: attribute, property, data layer.

**Association**
Where on a geometry a Field's values sit: `vertex`, `edge`, `face`, `cell`, `sample`, or
`geometry` — the last meaning one value for the whole thing, such as an area statistic. A Field
without an Association cannot be serialized, and two Fields with different Associations are not
interchangeable even when their value counts match.

**Session**
One continuous conversation. **The Session is the isolation unit**: Objects, Runs, conversation
memory and budgets all belong to exactly one Session and are never visible from another. See
ADR-0004.

A Session is identified anonymously and scoped to a browser, so **Session lifetime is not person
lifetime** — the same person returning tomorrow is a different Session and does not reach
yesterday's Objects. When authentication arrives, what changes is who a Session belongs to, not
what a Session is.

> **Only partly true in code, recorded rather than hidden.** Over HTTP, `server.py` now keeps
> Objects and Runs per Session, keyed by the `X-DTCC-Session` header the chatbot sends, and
> `chatbot/memory.py` `retrieve()` filters on `session_id`. What is still missing: `DiskCache`
> keys carry no identity; the Session id is client-supplied and unauthenticated; there is no
> Session expiry, only a cap of 8 live Sessions that drops the least recently used idle one;
> and budgets are an equal share of one object budget rather than per-Session budgets (T11/U7,
> #23). Over stdio and for in-process callers there is a single Session per process.

**Scenario**
A question with a known-good answer, written by someone with domain expertise, used to judge
whether the assistant behaves correctly. A Scenario asserts the *sequence and parameters* the
assistant chose, not only the final number. Scenarios are the acceptance gate; they are not tests
of the platform's arithmetic.

**Provenance**
The record of what produced an answer: which model, which provider, which prompt version, which
catalogue revision, which Operations ran. An answer without Provenance cannot be reproduced or
audited, and once the model is swappable it cannot even be attributed.

**Corpus**
A body of material that retrieval searches — as distinct from conversation memory, which is what
was said before. "It cannot look things up" and "it does not remember me" are different
complaints with different fixes.

## Terms owned by dtcc-twin

**Capability Catalog (Twin)**
The runtime discovery view of authoritative Dataset Definition descriptors contributed by DTCC
Core and DTCC Sim. Twin consumes this view and does not maintain a competing registry.

> This repo currently *reflects over* the Core registry to build its own catalogue, which is the
> same job done a different way. The descriptors it gets back carry three fields where the Twin
> definition calls for nine. See ADR-0002 and ADR-0005.

**Dataset Definition (Twin)**
The authoritative description of something the platform can produce: identity, title, explanation,
parameter schema, semantic result type, coverage, required inputs, execution requirements, output
semantics.

**Dataset Realization (Twin)**
A concrete result produced by executing a Dataset Definition with specific parameters. The Twin
term for what this repo would call the Object a Run produced.

**Front door**
Not a Twin term. Used here for the claim that this repo is the one surface where a person can
ask the platform a question in their own words, rather than knowing which control to operate.
**Settled on 2026-09-15**: Vasilis stated it directly on a 1:1 — the chatbot is part of the
product and is the front door. Decided by authority rather than by the municipality-interview
evidence, which still does not exist. See ADR-0001. It is not yet written into
`dtcc-twin/DESIGN.md`, which mentions no agent or chatbot.
