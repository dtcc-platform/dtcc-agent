# dtcc-agent restructure — design

**Status:** proposed, for review. Nothing here is committed to.
**Relationship to other documents:**

- The **inventory** on dtcc-agent#1 records what works today, with evidence, and a phase plan
  written *before* this design. Where the two disagree on sequencing, this document is later.
- **`docs/adr/0001`–`0006`** hold the six decisions that were hard to reverse, surprising, and the
  result of a real trade-off. This document holds the rest — decisions with real rationale that
  simply were not surprising enough to earn an ADR.
- **`to-questionnaire-dtcc-agent-restructure.md`** holds the eight questions nobody on the
  engineering side can answer. Several phases below are gated on them.
- **`CONTEXT.md`** is the glossary. Terms in **bold** below are defined there.

## What changed since the inventory's phase plan

The inventory proposed five phases on the evidence of what was broken. Scoping the restructure on
top of it changed three things:

1. **The harness is being replaced** (ADR-0003). That is a larger change than anything in the
   original plan and it did not appear in it.
2. **The evaluation suite moved from a nice-to-have to a gate**, which changes *where* it sits —
   see the sequencing note below. This is the single most important change in this document.
3. **Deployment acquired real content** — provider split, identity, retention, budgets, gates —
   where the original Phase 3 was a placeholder.

**Revised 2026-09-14, after dtcc-core#85.** Core closed and implemented its model-hardening issue
on 2026-09-13 and migrated five repos with it — Core, Sim, Upload, Atlas and Tangible Twin. This
repo was not one of them. Nothing in this plan is invalidated: the agent's 112 tests still pass
against Core `develop` `464c58d`, every Core module path it imports survives, and the operation
catalogue only grows (131 → 135). Three targeted changes, all recorded in place below:

- **Phase 0 gains one item** — reproject on field-carrying geometry now raises where it used to
  return a result. See the Phase 0 note.
- **Phase 4 shrank and mostly moved upstream** — there is now a blessed native decode path, but
  Sim did not migrate the two datasets this repo exposes. See the Phase 4 note.
- **ADR-0001, ADR-0002, ADR-0003 and ADR-0005 carry revision notes.** ADR-0005's protobuf
  evidence is withdrawn; ADR-0002 gained a partial reference implementation; ADR-0003 got cheaper.

The open question this raises is not technical. Five repos moved together and this one was not
included, and no design document in the new Core mentions the agent, the chatbot or Lurkie. That
is §5a of the questionnaire and it is a blocker: it decides whether this plan targets Core
`develop` or waits for a release that has not happened since January.

**Revised again 2026-09-17, after the 2026-09-14 standup and the 2026-09-15 1:1.**
Transcripts and notes in `~/Projects/dtcc/meetings/`.

Settled, and folded into the ADRs: the chatbot **is product and is the front door** (ADR-0001 is
now accepted, not proposed); **deployment is AWS**; **authentication is already solved and
central across all services** (ADR-0004); **no objection to pydantic-ai** (ADR-0003); the **MCP
server runs alongside Vasilis's DTCC Engine**, not through the Twin API (ADR-0005). The order of
work was confirmed as: land a couple of unblocking commits, then the `mcp` migration or pin so a
clean machine works, then correct the documentation, then the evaluation harness — which is
Phase 0, Phase 1, Phase 2 as written here.

Two changes to this plan's assumptions:

- **The ten follow-up issues are cancelled.** The inventory findings were assessed as minimal and
  low level, most not worth an issue, "fixable in two commits". Phase 0 absorbs them directly
  rather than filing them.
- **Phase 2's answer key is now the critical path.** With the product question answered, the
  harness decision agreed and Phase 0 unblocked, the only thing that still has no owner is the
  domain expert who writes the ten to fifteen evaluation scenarios. Everything from Phase 3
  onward sits behind it. §6 of the questionnaire is promoted accordingly.

**And the biggest change: this is a fresh repository, not a migration in place.** Confirmed
2026-09-17. Phase 0 inverts — pinning `mcp<2` and declaring `dtcc-core` stop being fixes and
become the first `pyproject.toml`, the import smoke test is written before there is anything to
break, and Napoleon's lint/TDD/harness conventions get adopted from commit one instead of
retrofitted. What survives from Phase 0 as actual work: the Session isolation unit and the memory
session filter (ADR-0004), the security items, and the reproject error path from §5b.

Provider funding is also settled: **Vasilis pays for OpenRouter during testing, the university
supplies AWS tokens later for Bedrock** — the test-versus-production split ADR-0003 was built
for. The ChatGPT/Codex subscription cannot substitute: it authenticates by OAuth scoped to the
Codex client (`auth_mode = "chatgpt"`, `OPENAI_API_KEY: null`), and pydantic-ai's providers all
take an `api_key`. Record per-scenario cost and tokens in the Phase 2 harness so the model
comparison is data rather than impressions.

Still needing Vasilis: the new repo's name and owner, whether Lurkie moves across, what happens
to `dtcc-agent#1` and its assignment, and whether the old public repo gets archived. See
§5a-bis.

### The scope, in Vasilis's terms — recorded 2026-09-18

Two deliverables, and they are the frame for everything above:

1. **The agent becomes a standalone application**, not a Claude process. Today the "agent" is a
   Claude Agent SDK loop inside `chatbot/app.py` that spawns `python -m dtcc_agent` as a stdio
   subprocess. It becomes an application with its own service lifecycle and a provider-agnostic
   harness (ADR-0003).
2. **The MCP server is redesigned and enhanced**, not replaced. It stays a self-contained native
   module that "has everything it needs", running **in parallel with the Engine** and not through
   the Twin API — Vasilis, 2026-09-15 at 14:16-14:52, confirmed "σωστά σωστά".

**A misreading corrected before it reached the plan.** From the Engine backend design I inferred
that its scope — "generic discovery and invocation of all Core and Sim Dataset Definitions... no
manually curated list" — overlaps `registry.py`, `dispatcher.py` and `runner.py`, and concluded
the agent should become an Engine HTTP client, which would have rewritten roughly half the MCP
server and removed `dtcc-core` from its dependencies. **That is not the decision.** Capability
overlap between two systems is not subsumption of one by the other, and the topology was settled
on the record in the other direction. Consequences of dropping that inference: Phase 0's "declare
the `dtcc-core` dependency" is correct and stays, because a self-contained module keeps importing
Core; the Engine v1 deferrals (polling-only status, no streaming, no guaranteed cancellation) do
**not** constrain Phase 5, because the agent reads progress from dtcc-sim's own API, which already
reports `{"status": "running", "progress": 0.15, ...}`; and the agent never holds an Engine token.

The overlap is still worth one sentence to Vasilis at some point — two systems doing generic
Dataset dispatch is a duplication question for later, not a reason to revisit a settled call.

### The sequencing insight

The original plan put the evaluation suite late, as an artifact the paper needs. That was wrong.

Replacing the harness (ADR-0003) rewrites the code path that decides *which tools get called with
which parameters*, in a system whose behaviour is stochastic. Without a scenario suite in place
first, there is no way to tell a successful migration from a regression — you would be comparing
two demos by eye. **The evaluation harness must land before the harness replacement, not after.**

The same argument applies to every later capability change: curated Task tools, retrieval, prompt
caching, a model swap. Each is an unmeasurable change until the suite exists.

## Decisions not covered by an ADR

Recorded here because each has a rationale worth keeping, and each would otherwise live only in a
conversation.

**The MCP server runs in-process; a session-keyed disk store waits on #85.**
The object store is currently module-level state inside a subprocess respawned per message, so
**Object references** die between turns and the whole scientific stack reloads every time. Both of
the complaints raised about the chatbot reduce to this. Moving to an in-process server fixes both
in one change. Cross-session survival needs objects on disk, which works for most dtcc-core types
— `disk_cache` already pickles them — but *not* for live simulation results, which hold handles
into the numerical library. That half is blocked on dtcc-core#85 and is scoped accordingly.

**The seven hardcoded Operation schemas move to versioned configuration, unchanged.**
`chatbot/config.py` embeds seven schemas in the system prompt with an instruction not to call
`describe_operation` for them. Curation is therefore already happening, in the least testable
place in the repo. Moving them as-is is mechanical, commits to nothing about the eventual
**Task tool** surface, and makes them coverable by the scenario suite.

**Task tools are derived from Scenarios, not designed up front.**
Designing four task-shaped tools today means inventing the tasks, while the product question
(ADR-0001) is still a hypothesis. The Scenarios a domain expert writes are, by construction, a
statement of what people actually want to do. Take the vocabulary from them.

**A descriptor contract is proposed to dtcc-core, not built unilaterally.**
`list_operations()` returns three fields — `name`, `category`, `description` — where the Twin's
Capability Catalog calls for nine, and `describe_operation` reverse-engineers parameters from
unannotated Python signatures, so most come back with an empty type. This repo is the only
consumer that already needs what the catalog promises, which makes it the natural forcing
function. It does not make it the owner. File the ask with the evidence.

**The repo stays separate until D5 is answered.**
Converging on Twin contracts raises the question of folding into `dtcc-twin`. Doing that now would
answer the product question by architecture instead of by decision. The MCP server also has
standalone value that does not depend on the chatbot existing.

**Distribution is PyPI plus a container.**
A tool whose purpose is to be plugged into an MCP client needs a one-line install; `.mcp.json`
pointing at `python -m dtcc_agent` already assumes it is importable. `dtcc-viewer` is the
counter-example available: not on PyPI, submodule uninitialised, and consequently unreachable by
anyone following the documentation. This costs someone owning releases.

**Cost is measured per completed task, in the evaluation harness.**
Measured per request, a system that takes five turns to finish a job looks cheap. The meaningful
unit is what it costs to answer the question. Putting it in the same harness as quality makes it
reproducible and makes "is this model worth it here" a measurement. Per-request logging follows
in production.

**The retention posture is written before the first deploy, not after.**
Conversations and **Session** memory are the only personal-data surfaces; everything else is
public data. What is retained, for how long, where, and which providers see it is a half-page that
is much cheaper to write now.

**The long-running-work lifecycle is implemented here, against Twin's vocabulary.**
`DESIGN.md` specifies validation, queued, running, progress, cancellation and actionable failure.
`dtcc-twin` has no implementation, so waiting means waiting indefinitely. This repo is the only
surface running multi-minute jobs in front of a person. The data already exists and is discarded:
the dtcc-sim API returns `{"status": "running", "progress": 0.15, "message": "Downloading files
(2/2)..."}` on every poll.

**Every tool call is recorded with Session reference, parameters and cost.**
It is the only way to answer "why did it do that" about a stochastic system, and it is the raw
material from which Task tools get derived. Budgets are per-Session before per-user, because there
is no user identity until authentication exists.

## Phases

Each phase states goal, scope, deliverable, dependency, priority and acceptance.

### Phase 0 — Install, start, isolate · **blocking**

**Goal:** a fresh clone runs, and the service is safe to deploy.
**Scope:** pin `mcp<2` or migrate to `MCPServer`; declare the `dtcc-core` dependency; add an
import-and-list-tools smoke test; make the **Session** the isolation unit and fix the memory
session filter (ADR-0004); resolve the security items.
**Deliverable:** a green CI job that installs from clean and starts the server.
**Depends on:** nothing. **Priority:** highest — nothing else should start.
**Acceptance:** clean machine, install, `python -m dtcc_agent` lists 22 tools; CI fails if it
regresses; security items closed or explicitly accepted by their owner.

> The `mcp<2` pin concerns the **server** side. ADR-0003 replaces the **client** harness. They are
> different packages and do not conflict.

> **Added 2026-09-14 — one new Phase 0 item, from the dtcc-core#85 migration.** Core `develop`
> now raises `NotImplementedError: Fields and semantic regions require an explicit reprojection
> rule` where Core `5cf56fa` silently returned a result. Verified A/B in two clean venvs. This
> repo exposes six `reproject.*` operations to the model and every simulation result carries a
> Field, so the natural "simulate then reproject for display" chain now raises. Whichever way §5b
> of the questionnaire is answered, the tool descriptions and the error path need to change, and
> the change belongs here rather than later because it is a user-visible failure on a documented
> chain. Note this only bites once the `dtcc-core` dependency is declared and pinned (§5a) —
> against the Core revision the inventory tested, the old behaviour still applies.

### Phase 1 — Make the documentation true · ~4 days

**Goal:** every documented example runs as written.
**Scope:** `air_temperature` → `T_ambient`; document `SHARED_RESULTS_DIR`; add
`BuildingCollection` to the renderer dispatch; generate tool and operation counts rather than
hardcoding them; document the three GeoJSON tools; move the seven prompt schemas into versioned
configuration; publish to PyPI.
**Deliverable:** a CI job executing every README example end to end.
**Depends on:** Phase 0. **Priority:** high — it is cheap and it is what a new person hits first.
**Acceptance:** the example script passes in CI on a clean environment.

### Phase 2 — The evaluation harness · **gate** · ~1 week

**Goal:** a way to tell whether a change made things better or worse.
**Scope:** ten **Scenarios** with expected tool sequences and parameters, written by a domain
expert; a runner asserting sequence and parameters, with numeric tolerance where a number is
stable; cost per completed task measured in the same run.
**Deliverable:** `scenarios/` plus a runner, reporting quality and cost together.
**Depends on:** Phase 1. **Blocked on:** a domain expert supplying the answer key — questionnaire §6.
**Priority:** highest after Phase 0, because everything downstream is unmeasurable without it.
**Acceptance:** the suite runs in CI, reports both numbers, and fails on a deliberately broken
scenario.

### Phase 3 — Replace the harness · ~2 weeks

**Goal:** provider portability, and the built-in shell tooling gone.
**Scope:** replace the Claude Agent SDK with pydantic-ai (ADR-0003); configuration-driven provider
and model; OpenRouter for testing, Bedrock for production; rebuild context management; connect to
this repo's own MCP server over the client-side path.
**Deliverable:** the chatbot running on pydantic-ai with provider chosen by configuration.
**Depends on:** Phase 2 — see the sequencing note. **Priority:** high.
**Acceptance:** the scenario suite scores no worse than the pre-migration baseline, on both
providers, with cost reported for each.

### Phase 4 — Close the mini-service capability gap · ~1 week

**Goal:** the deployment mode that ships can answer the question the product exists for.
**Scope:** deserialize the returned volume mesh so `run_simulation` and `compare_scenarios` return
field statistics remotely — or scope them to direct mode explicitly and fail loudly.
**Deliverable:** either statistics in mini-service mode, or a clear refusal plus corrected docs.
**Depends on:** Phase 1, **plus one upstream dtcc-sim change** (see the note below).
**Priority:** medium-high — the README's headline example depends on it.
**Acceptance:** `compare_scenarios` returns a numeric comparison in mini-service mode, or refuses
with a message that matches the documentation.

> **Revised 2026-09-14 — this phase shrank and mostly moved upstream.** dtcc-core#85 landed on
> 2026-09-13 and dtcc-sim adopted the native `.dtcc` exchange in `e24a1f2`, so there is now a
> blessed self-describing decode path: `DTCC.ModelFile`, read through Core's new `io.load_model`,
> with vertex-valued simulation fields carrying `association="vertex"`.
>
> But Sim migrated only `urban_wind_simulation` and `traffic_simulation`. The two datasets this
> repo actually exposes are untouched: `urban_heat_simulation` and `air_quality_field` are still
> `format: Optional[Literal["xdmf"]]` in `dtcc_sim/datasets.py`, and xdmf is multi-file, which is
> why the deployed mode returns no statistics. So the work is no longer "build a deserialization
> story". It is one `format` literal upstream plus an `io.load_model` call here. File the dtcc-sim
> issue before scheduling this phase; without it, the only honest option left is the explicit
> refusal.
>
> Also worth picking up while here: wind and traffic now have clean native results and this repo
> exposes neither. `_SIMULATION_NAMES` in `runner.py` lists two of the four.

### Phase 5 — Lifecycle, provenance and observability · ~2 weeks

**Goal:** a five-minute simulation stops looking broken, and answers become auditable.
**Scope:** the Twin lifecycle — validation, queued, running, progress, cancellation, actionable
failure; per-tool-call audit records with **Session** reference, parameters and cost; per-Session
budgets; full **Provenance** recorded and surfaced per Session.
**Deliverable:** progress and cancellation in the UI; an audit record per call.
**Depends on:** Phase 3 — provenance is only meaningful once the model is swappable.
**Priority:** medium-high. **Acceptance:** a running simulation reports progress and can be
cancelled; every answer can be traced to the model, prompt version and Operations that produced it.

### Phase 6 — Deployment · ~2 weeks

**Goal:** a deployment a second person can perform.
**Scope:** one AWS account with separate dev and production stacks; an identity provider rather
than a shared password; the retention posture written down; the three deploy gates enforced.
**Deliverable:** a documented, repeatable deploy.
**Depends on:** Phases 0, 3, 5. **Blocked on:** account ownership and authentication —
questionnaire §3. **Priority:** medium — gated on answers, not on engineering.
**Acceptance:** someone other than the author deploys from the documentation; all three gates pass
— scenario suite green, security items closed, cost per task measured.

### Phase 7 — Retrieval · ~2 weeks

**Goal:** the assistant can look things up rather than requiring the asker to name an Operation.
**Scope:** a separate `dtcc-docs` MCP server (ADR-0005) indexing the Capability Catalog
descriptors; mandatory citations.
**Deliverable:** a second MCP server, catalogue-only.
**Depends on:** Phase 3. **Blocked for external corpora on:** a licence reviewer — questionnaire §7.
**Priority:** medium. **Acceptance:** "what can you tell me about this neighbourhood" is answered
from the catalogue, with every claim citing a descriptor and revision.

### Phase 8 — Task tools · unscoped

**Goal:** fewer, task-shaped choices over the full catalogue.
**Scope:** derive **Task tools** from the Phase 2 Scenarios; keep the catalogue as a documented
escape hatch. **Depends on:** Phase 2 having produced real Scenarios. **Priority:** deliberately
last — the vocabulary does not exist yet. **Acceptance:** the Scenarios pass using Task tools,
with fewer calls than the Operation-level baseline.

## Running in parallel, owned elsewhere

- **dtcc-core #85 and #87** — engineer around them, push them upstream with the measurements this
  work produced. Questionnaire §5.
- **The descriptor contract proposal** to dtcc-core. No phase; it is an issue and a conversation.
- **D5 — product or research output.** Questionnaire §1. It does not block Phases 0–4. It does
  determine whether Phases 6–8 are worth doing at all.

## Risks

**The largest is sequencing.** If Phase 3 starts before Phase 2, the harness replacement becomes
unfalsifiable and any regression surfaces later as "the chatbot got worse" with no way to
bisect it.

**The second is Phase 2's dependency on someone else.** The scenario answer key needs a domain
expert. If nobody supplies one, the honest fallback is a much smaller suite over deterministic
assertions — tool sequence only, no parameter judgement — which is weaker but still gates Phase 3.

**The third is that D5 stays unanswered.** Phases 6 through 8 all assume the front-door
hypothesis. If the answer turns out to be "research output", most of this document is
over-engineering and the correct plan stops after Phase 4.
