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
The short identifier that names a stored Object.

> **Vocabulary conflict, unresolved in code.** Three names are in use for two concepts:
> `object_id` and `result_id` are the *same* namespace (an Operation returns `result_id`;
> the same value is accepted anywhere `object_id` is expected), while `run_id` names an entry
> in a *different* store. Canonical terms going forward: **Object reference** for the first,
> **Run reference** for the second. The code has not been changed.

**Run**
One execution of a simulation, together with the parameters it was given. Distinct from an
Object: a Run is an event, an Object is a value. A Run may yield Objects.

**Field**
Values attached to every point of a geometry — a temperature, a wind speed. A Field is what makes
a simulation result meaningful, and is the thing most easily lost in conversion.

**Session**
One continuous conversation with one person. **The Session is the isolation unit**: Objects, Runs,
conversation memory and budgets all belong to exactly one Session and are never visible from
another. See ADR-0004.

> **Not true in code, recorded rather than hidden.** This is the intended definition, not a
> description of today. `dtcc_agent/` contains no notion of a session, user or tenant: the stores
> are module-level singletons (`server.py:32-38`), `ObjectStore.get` does no authorization,
> `list_objects` enumerates every object in the process, LRU eviction is cross-user, and
> `DiskCache` keys carry no identity. `chatbot/sessions.py` scopes only the model's conversation
> transcript and does not reach the stores; `chatbot/memory.py` stamps `session_id` on write but
> `retrieve()` queries with no filter on it. Making the Session real is new work in the rebuild,
> not a refactor — there is no seam to thread an identifier through.

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
