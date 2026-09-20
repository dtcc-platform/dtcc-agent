# dtcc-agent rebuild plan

Written 2026-09-19, from settled ADRs, after a design interview that closed twelve open questions.
This supersedes the sequencing in `2026-09-14-restructure-design.md` and in `dtcc-agent#1` §6.
Those two are kept as history. Where this document and either of them disagree, this one is the
plan of record; where this document is silent, neither of them fills the gap by default.

**Decisions live in `docs/adr/`.** This document sequences work and records what is deliberately
not yet decided. It does not re-argue anything an ADR settles.

## Scope

Two deliverables, agreed on the 2026-09-15 call:

1. **The agent becomes a standalone application** — it stops being a Claude CLI process driven over
   stdio. In practice this is four things, all in scope: a real transport, model independence
   (ADR-0003), a rebuilt UI, and a deployment story.
2. **The MCP server is redesigned and enhanced** — not replaced. It stays a self-contained native
   module running in parallel with the DTCC Engine, not through the Twin API.

On a branch off `develop` (ADR-0009). Roughly 1,900 lines of domain logic are kept; `server.py`
and `chatbot/` are replaced; session scoping, auth, provenance and surface tests are new.

## What is true today, measured

Worth stating because three planning artifacts describe it differently.

- **Two processes, not one.** `chatbot/app.py` is a uvicorn service on 8050 that constructs a
  `ClaudeSDKClient` *per user message* (`app.py:228`), which spawns a `claude` CLI process, which
  talks stdio to `python -m dtcc_agent`. `app.py:14` pops `CLAUDECODE` to defeat the CLI's own
  guard against nesting.
- **The model sees 22 tools, not 133.** The catalogue reaches it through three dispatch tools
  (`server.py:451`, `:482`, `:504`). `run_operation` takes `dict[str, Any]` and type-checks
  nothing.
- **There is no session, user or tenant anywhere in `dtcc_agent/`.** Zero grep hits. Stores are
  module singletons (`server.py:32-38`). `ObjectStore.get` does no authorization, `list_objects`
  enumerates every object in the process, LRU eviction is cross-user, `DiskCache` keys carry no
  identity, and `chatbot/memory.py:64-97` retrieves without filtering on the `session_id` it
  stamps on write.
- **`server.py` is 1,190 lines with zero direct test coverage.**
- **Nothing is deployed.** `localhost:8050`, started by hand. No CI directory, no deploy config.
- **The render path is broken end to end.** `renderer.py:75` writes to a fresh random tempdir;
  `app.py:60` serves a fixed one. The returned `image_path` is unreachable over HTTP.
- **The DTCC Engine does not exist as code** — two design documents and an empty directory,
  no pull requests ever, last push 2026-09-06.

## The nine redesign items

Items 1-3 are structural: a server is stdio or it is not, stores are globals or they are injected.
Items 4-9 are genuinely incremental and land against the harness that can prove they helped.

| # | Item | Milestone |
|---|---|---|
| 1 | Transport — stdio to HTTP | M1 |
| 2 | State ownership — module globals to injected per-session stores | M1 |
| 3 | Auth — none today; required once transport is HTTP | M2 |
| 4 | Validation at the dispatch boundary — generate models from `ParamInfo` | M3+ |
| 5 | Error contract — typed errors, not flattened strings (`dispatcher.py:165-167`) | M3+ |
| 6 | Provenance — model, prompt version, catalogue revision, operations run | M2 |
| 7 | Collapse the direct/mini-service fork — four sites in `runner.py` plus a hidden one in `registry.py:344-352` that makes the catalogue itself mode-dependent and memoised | M3+ |
| 8 | Result lifecycle — `_results` never evicts, exports hardcode `/tmp`, renders unreachable | M3+ |
| 9 | Catalogue as a versioned artifact rather than import-time reflection | M3+ |
| 10 | Tests for the tool surface | M1, precondition |

Items 4-9 are re-evaluated as the work proceeds rather than committed to in order now. The
sequencing above is a starting position, not a contract.

## Milestones

Revised 2026-09-20 by `/plan-eng-review`. Five milestones, not three: M0 was split out so the
behaviour-changing work has a gate to fail against, and M4 was added because deployment was in
the scope statement but in no milestone.

```
        ┌─────────────────────── TODAY ───────────────────────┐
        │  browser ─ws─► chatbot/app.py (uvicorn :8050)       │
        │                     │ per user message              │
        │                     ▼                               │
        │              ClaudeSDKClient ──spawn──► claude CLI   │
        │                                            │ stdio  │
        │                                            ▼        │
        │                              python -m dtcc_agent    │
        │                              (FastMCP, 22 tools)     │
        │                                     │               │
        │                              module-global stores    │
        │                              shared by everyone      │
        └──────────────────────────────────────────────────────┘

        ┌──────────────────── AFTER M1 ────────────────────────┐
        │  browser A ─ws─┐                                     │
        │  browser B ─ws─┤► chatbot/app.py                     │
        │                │      │  one MCP session PER browser │
        │                │      ▼                              │
        │                │  ClaudeSDKClient ──► claude CLI     │
        │                │        (still, until M3)            │
        │                │           │ streamable-http         │
        │                │           ▼                         │
        │                │    dtcc-agent MCP server            │
        │                │       │                             │
        │                │   lifespan: catalogue built ONCE    │
        │                │       │                             │
        │                │   ┌───┴────────────────┐            │
        │                │   ▼                    ▼            │
        │                │  per-session stores   shared cache  │
        │                │  (objects, runs,      (downloads    │
        │                │   memory, exports)     ONLY)        │
        │                │       │                             │
        │                │   bounded worker pool ──► dtcc-core │
        │                └─────────────────────────────────────┘
```

### M0 — CI and characterisation tests. Zero behaviour change.

**Forward.** A GitHub Actions workflow (follow `dtcc-sim`'s `ci-build-tests.yml`). Tests
characterising the **current** 22-tool surface. Declare `dtcc-core` pinned to a commit SHA, as
`dtcc-sim` does at its `pyproject.toml:19`, and copy its `dtcc-core-contract.yml` so a candidate
Core SHA is tested against the agent before the pin moves. Turn the silent `ImportError` at
`registry.py:211-214` into a startup failure.

**Acceptance.** CI runs on push and pull request and is green. A fresh install with no `dtcc-core`
fails loudly instead of serving an empty catalogue. The characterisation tests describe today's
behaviour, including behaviour that is wrong — they are the baseline, not an endorsement.

**Rollback.** Nothing to roll back; no behaviour changes.

**Prerequisite.** Merge PR #2 first. This branch's `pyproject.toml:21` still reads `mcp>=1.0.0`
with no ceiling, so CI cut from `develop` before #2 lands resolves `mcp` 2.x and fails at import
for a reason already fixed next door.

### M1 — transport, state, references

**Forward.** HTTP transport via `mcp.run(transport="streamable-http")`. Per-session stores
injected through FastMCP's `lifespan`, with the corrected split from ADR-0004. Typed references
and the Run-to-Object link (ADR-0010). Catalogue built once in `lifespan` rather than memoised on
first use. `_results` gets the ObjectStore's LRU and byte budget. A global memory budget with a
per-session cap. A bounded worker pool for all Core execution. Session-owned artifact directories
plus an authenticated download route, replacing the caller-supplied paths in `export_object` and
`load_geojson`. `dtcc_core`'s internal `asyncio.run()` calls wrapped in
`anyio.to_thread.run_sync`, and `nest_asyncio` deleted. Lurkie opens one MCP session per browser
session. Dockerfile and compose become two services.

**Acceptance.** The 122 existing tests stay green. The replacement for `server.py` has direct
coverage. **Two real MCP clients over HTTP**, each creating objects, runs and conversation
memory, and neither sees the other's. The shared download cache still hits across both. A Run
reference passed where an Object reference is expected is refused, not missed. `get_run_summary`
returns the Object reference its Run yielded. A tool that downloads data works under uvloop. No
tool accepts a caller-supplied filesystem path. `docker compose up` produces a working
two-service system.

**Rollback.** Transport is selected by configuration and the stdio entry point stays alive, so
rollback is a config flip, not a revert. The `.mcp.json` route keeps working throughout.

**Not in scope.** Changing which tools exist or what they return. Removing the `claude` CLI — see
below.

**Corrected from the first draft:** M1 previously claimed "Lurkie works with no `claude` CLI on
the machine". That is not achievable here. Changing the MCP transport changes how the SDK reaches
this server; it does not remove `ClaudeSDKClient`, which is still Claude Code as a library and
still spawns the CLI. **The CLI stays until M3.**

### M2 — auth, provenance, measurement

**Forward.** Admission control on the HTTP surface: minting a session requires a deployment-level
shared secret. Session identifier and subject identifier are separate fields from the start
(ADR-0004). A provenance record per answer: model, prompt version, catalogue revision, operations
run. The measurement layer of the evaluation harness reading it (ADR-0008).

**Acceptance.** A request with no session is refused, and a session cannot be minted without the
admission secret. Every answer carries a complete provenance record. The harness reports latency,
tokens and cost per task over a fixed question set.

**Rollback.** Admission control is a config toggle. Provenance is additive — a record nobody reads
breaks nothing.

### M3 — pydantic-ai, caching, Lurkie

**Forward.** Replace the Claude Agent SDK (ADR-0003); the `claude` CLI dependency disappears here.
Prompt caching (ADR-0006), which deletes the seven hardcoded schemas in `chatbot/config.py`.
Lurkie's rebuild against the stabilised API.

**Acceptance.** The harness reports before-and-after latency and cost across the migration. A
second provider runs the same question set. Lurkie works with no `claude` CLI installed.

**Corrected from the first draft:** M3 previously required "the scenario suite scores no worse
than the M2 baseline". M2 delivers only the measurement layer; the correctness answer key is
explicitly the half that needs Anders or Nuri and has no committed delivery date (ADR-0008). **M3
cannot be gated on a suite that may not exist.** Either the answer key gets an owner and a date
before M3 starts, or M3's acceptance is latency, cost and provider portability only — and the
behavioural-equivalence claim is not made. Cost and latency cannot establish equivalence.

**Rollback.** The largest risk in the plan and the hardest to reverse. Keep the Agent SDK path
behind configuration for one milestone rather than deleting it on the day the replacement lands.

**Dependency note:** Lurkie's rebuild is listed here, but D1, D2 and D3 — whether it renders
geometry, which engine, and whether it becomes an Atlas panel — are all scheduled *after* M3.
Rebuilding a UI before deciding whether it should exist is the wrong order. **Either resolve
D1-D3 before M3's UI work begins, or M3 ships only the runtime change and the UI rebuild becomes
its own milestone.** Flagged rather than silently sequenced.

### M4 — deployment

**Gated on M2.** An internet-reachable service without admission control is not deployable.

**Forward.** The AWS path: image build and publish, configuration, secrets, lifecycle, and
whatever the access answer turns out to be.

**Open, and not ours to answer:** who may deploy. Still unanswered from the questionnaire.

## Deferred decisions register

Open on purpose, with what decides each and when it comes up. Nothing here blocks M1.

### D1 — Does Lurkie render geometry, or only text and static images?

**The dominating question for the UI, and it is not a framework question.** This product's
answers are geometry and fields. A chat box that renders only markdown is a weak surface for a
digital twin, and the visual path is currently broken anyway (item 8).

*Decides it:* whether an interactive map and 3D view are required, or a static image is enough.
*When:* after M3, when the streaming interface is known. *Evidence already gathered:*
`engine-bench` benchmarks deck.gl/MapLibre, vtk.js, Cesium, PlayCanvas and three.js against a
real Gothenburg tile with a temperature field on a 50,729-vertex volume mesh — measured findings,
not opinions. **Start from `engine-bench/NOTES.md`, not from a framework comparison.**

### D2 — If Lurkie renders geometry, which engine?

*Decides it:* D1 first, then the measured tradeoffs in `engine-bench` — vtk.js volume-plus-surface
compositing works once the opacity ramp is retuned; Cesium's `VoxelPrimitive` is `@experimental`
and runs about one frame per second; deck.gl's `ScenegraphLayer` never binds glTF `COLOR_0` while
`SimpleMeshLayer` does; PlayCanvas needs no vertex-colour workaround.
*When:* with D1. *Note:* this overlaps the engine work already in flight for `dtcc-twin#1`, so
the decision should be made once for both rather than twice.

### D3 — Vanilla, no-build framework, SPA, adopted chat UI, or no UI at all?

Five real options: keep vanilla with more structure; a no-build framework (Preact+htm, Alpine,
htmx); a full SPA with a build step; adopt an existing chat UI (Open WebUI, Chainlit,
assistant-ui) and stop owning chat entirely; or have no separate UI and become a panel inside
Atlas.

*Decides it:* D1 and D2 mostly settle it — the rendering requirement dominates the component
model. The Atlas option is a question for Vasilis and a yes makes the other four moot.
*When:* after M3. *Until then:* stay vanilla. "No build step" means `docker compose up` with no
Node toolchain, and that is worth more than component ergonomics while the API underneath is
still moving.

### D4 — Real authentication: which identity provider, and when?

*Decides it:* someone specifying central platform authentication. Nobody has: the Engine design
defers per-consumer tokens, roles, permissions and quotas from v1
(`dtcc-engine-backend-design-v1.md:74`).
*When:* not in this rebuild. *Mitigation:* M2's session identifier is carried end to end so that
substituting an authenticated subject is a substitution, not a re-plumb (ADR-0004).

### D5 — Does the tool surface move from 22 generic tools toward Task tools?

*Decides it:* the scenarios. `CONTEXT.md` names Task tools and records that none exist; the
restructure design's position — that they are derived from Scenarios rather than designed up
front — survives and is the right one.
*When:* after the correctness layer of the harness exists (ADR-0008). *Until then the surface is
frozen*, which is why M1 explicitly excludes changing it.

### D6 — Does the agent ever become an Engine client?

*Decides it:* whether the Engine gets built, and whether the platform then wants one dispatch
layer. ADR-0007 is what gets cited in that conversation.
*When:* not on our schedule. *Owed now:* one sentence to Vasilis that the overlap exists, so it
is an early warning rather than a discovery.

### D7 — Does Core grow a field reprojection rule?

*Decides it:* `dtcc-core#112`, filed 2026-09-18 and awaiting an answer.
*When:* on reply. *Default if no:* the agent strips fields and reprojects geometry only, saying so
in the tool description. The third option — refusing reproject on field-carrying inputs — is
rejected as worse for the product.

### D8 — Native `.dtcc` output for heat and air quality

*Decides it:* `dtcc-sim#8`, filed 2026-09-18.
*When:* on reply. *Blocks:* field statistics in mini-service mode, and therefore a real numeric
`compare_scenarios`. Nothing in M1-M3 waits on it.

### D9 — Nine descriptor fields against three

`DESIGN.md:344-346` specifies nine; `registry.py` yields three (ADR-0002).
*Decides it:* whether convergence on Twin's contract is worth the work absent a Twin
implementation to converge with. *When:* alongside item 9, catalogue-as-artifact.

## What this supersedes

- `docs/plans/2026-09-14-restructure-design.md` — its Phases 0-8 sequencing, its fresh-repository
  premise (ADR-0009), and its Phase 0 acceptance criterion of "`python -m dtcc_agent` lists 22
  tools", which was inherited from a measurement of the old surface. Its reasoning is kept.
- `dtcc-agent#1` §6 and §7 — the five-phase plan and the ten follow-up issues, both already
  withdrawn publicly on 2026-09-18.
- `to-questionnaire-dtcc-agent-restructure.md` — a list of questions for a meeting. The answers it
  collected are now in ADRs; the questions it never got answered are in the register above.


## NOT in scope

Considered during the 2026-09-20 engineering review and explicitly deferred.

| Item | Why deferred |
|---|---|
| Hashing cache contents properly | Needs measurement before design; the metadata shortcut exists for a reason. `TODOS.md` T-001. |
| Cancellation and backpressure | Needs to know whether `dtcc_core` operations are interruptible at all. Depends on M1's pool existing. `TODOS.md` T-002. |
| Removing the deep copy at `dispatcher.py:149` | Requires knowing which of 133 operations mutate their inputs. Nobody does. |
| Task tools | Derived from Scenarios, which need the correctness layer. Register D5. |
| Nine-field descriptors | No Twin implementation to converge with yet. Register D9. |
| Real identity provider | Nobody has specified central authentication; the Engine defers it from v1. Register D4. |
| Becoming an Engine client | The Engine has no code. ADR-0007. Register D6. |
| A durable job queue | The bounded pool is sufficient at this scale; a queue is not automatically necessary. |

## What already exists

The review found three cases where the plan was about to build something the SDK or a sibling
repo already provides.

| Sub-problem | Already exists | Plan's use |
|---|---|---|
| HTTP transport | `FastMCP.run(transport="streamable-http")`, verified in `mcp 1.30.0` | **Now reused.** The first draft read as if this were bespoke. |
| Authentication | `FastMCP(auth=AuthSettings, token_verifier=TokenVerifier)` — a protocol with one method | **Now reused.** You implement a verifier, not an auth layer. |
| Per-server state construction | FastMCP's `lifespan` | **Now reused** for stores and the catalogue. |
| Session lifetime and limits | `session_idle_timeout`, `max_sessions`, `event_store` | **Now reused** instead of hand-rolled. |
| Core pinning and contract testing | `dtcc-sim`'s SHA pin plus `dtcc-core-contract.yml` | **Now copied** rather than invented. |
| CI workflow shape | `dtcc-sim`'s `ci-build-tests.yml` | **Now followed.** |
| LRU with a byte budget | `ObjectStore`, with 21 tests behind it | **Now reused** for `_results` rather than a second mechanism. |
| Spatial containment caching | `DiskCache.dataset_lookup` | Kept. It is the reason ADR-0007 keeps dispatch in-process. |

## Failure modes

For each new codepath, one realistic production failure and whether the plan covers it.

| Codepath | Failure | Test? | Handled? | Visible? |
|---|---|---|---|---|
| HTTP transport | Core download raises under uvloop | Yes (M1) | Yes — thread wrap | Loud |
| Per-session stores | Lurkie hands every browser one session | Yes — two-client test | Yes | Loud |
| Shared cache | One session served another's derived geometry | Yes (M1) | Yes — only downloads shared | Silent if wrong → **was a critical gap, now closed** |
| Session artifacts | A session reads another's export off disk | Yes (M1) | Yes — session-owned dirs | Loud |
| Bounded pool | Concurrent builds exhaust memory | Partial | Yes — pool cap | Loud (OOM) |
| Abandoned work | Closed tab's job blocks the pool | No | No | **Silent — T-002** |
| Catalogue in lifespan | Misconfiguration yields a partial catalogue | Yes (M1) | Yes — fail at startup | Loud |
| Cache version stamp | Stale pickle after a Core upgrade | Yes (M1) | Yes — discard on mismatch | Loud (cold start) |
| Missing `dtcc-core` | Empty catalogue, no error | Yes (M0) | Yes — startup failure | Loud |
| Provenance | Record is incomplete, harness numbers unattributable | Yes (M2) | Yes | Loud |

**One critical gap remains: abandoned work (T-002)** — no test, no handling, silent. Accepted
deliberately; it needs M1's pool to exist first.

## Worktree parallelization

| Step | Modules touched | Depends on |
|---|---|---|
| M0-CI | `.github/`, `pyproject.toml` | PR #2 |
| M0-tests | `tests/` | M0-CI |
| M1-transport | `dtcc_agent/server.py`, `__main__.py` | M0 |
| M1-state | `dtcc_agent/object_store.py`, `disk_cache.py`, `server.py` | M0 |
| M1-refs | `dtcc_agent/` (wide), `tests/` | M0 |
| M1-lurkie | `chatbot/` | M1-transport |
| M1-deploy | `Dockerfile`, `docker-compose.yml` | M1-transport, M1-lurkie |

```
Lane A: M0-CI → M0-tests           (sequential, shared tests/ + config)
Lane B: M1-transport → M1-lurkie → M1-deploy   (sequential, transport gates both)
Lane C: M1-state                   (independent of transport)
Lane D: M1-refs                    (independent, but touches everything)
```

**Execution order.** Lane A alone first — everything gates on a green build. Then B and C in
parallel worktrees. **Lane D must not run in parallel with B or C:** the reference rename touches
`server.py`, `object_store.py`, `dispatcher.py` and every test, which is the union of what B and C
touch. Land D before them or after them, never alongside.

**Conflict flag.** Lanes B and C both touch `server.py`. B replaces the transport and entry point;
C replaces the store wiring. Same file, different regions — coordinate or sequence.

## Implementation Tasks

Synthesized from this review's findings. Each derives from a specific finding.

- [ ] **T1 (P1, human: ~15min / CC: ~5min)** — repo — Merge PR #2 before cutting M0
  - Surfaced by: Architecture 6A — `pyproject.toml:21` has `mcp>=1.0.0` unbounded on this branch
  - Files: none (a merge)
  - Verify: `develop` contains `mcp>=1.0.0,<2`
- [ ] **T2 (P1, human: ~2d / CC: ~3h)** — packaging — Declare and SHA-pin `dtcc-core`; copy `dtcc-sim`'s contract workflow; make a missing Core fail at startup
  - Surfaced by: Architecture 8A — `dtcc-core` is not a declared dependency anywhere; `registry.py:211-214` returns silently
  - Files: `pyproject.toml`, `.github/workflows/dtcc-core-contract.yml`, `dtcc_agent/registry.py`
  - Verify: fresh venv, no Core → startup fails loudly; contract workflow runs on dispatch
- [ ] **T3 (P1, human: ~1w / CC: ~1d)** — CI — GitHub Actions + characterisation tests for the current 22-tool surface
  - Surfaced by: Step 0 / D2 — no `.github/` at all; `server.py` is 1190 lines with zero coverage
  - Files: `.github/workflows/ci-build-tests.yml`, `tests/test_server.py`
  - Verify: CI green on push and PR
- [ ] **T4 (P1, human: ~2d / CC: ~2h)** — runtime — Thread-wrap Core's internal `asyncio.run()`; delete `nest_asyncio`
  - Surfaced by: Architecture 1A — verified `ValueError: Can't patch loop of type <class 'uvloop.Loop'>`
  - Files: `dtcc_agent/dispatcher.py`, `dtcc_agent/__main__.py`, `pyproject.toml`
  - Verify: a download tool succeeds under `uvicorn --loop uvloop`
- [ ] **T5 (P1, human: ~3d / CC: ~4h)** — transport/session — streamable-http, one MCP session per browser, stores via `lifespan`
  - Surfaced by: Architecture 2A — after M1 the MCP client is Lurkie, not the browser
  - Files: `dtcc_agent/server.py`, `chatbot/app.py`, `chatbot/config.py`
  - Verify: two real MCP clients, neither sees the other's objects
- [ ] **T6 (P1, human: ~1d / CC: ~2h)** — cache — Share only `datasets.point_cloud`, `datasets.buildings`, `get_buildings`; the rest session-local
  - Surfaced by: Outside voice 1 / 12A — `content_fingerprint` hashes metadata, not contents
  - Files: `dtcc_agent/disk_cache.py`, `docs/adr/0004-*.md`
  - Verify: a builder result created in one session is never served to another
- [ ] **T7 (P1, human: ~4d / CC: ~6h)** — filesystem — Session-owned artifact dirs + authenticated download route; remove caller-supplied paths
  - Surfaced by: Outside voice 2 / 13A — `export_object` does `open(filepath, "w")` unvalidated; `load_geojson` reads any `.json`
  - Files: `dtcc_agent/server.py`, `dtcc_agent/geojson_store.py`, `dtcc_agent/renderer.py`, `chatbot/app.py`
  - Verify: no tool accepts an arbitrary path; a render is actually reachable over HTTP
- [ ] **T8 (P1, human: ~2d / CC: ~3h)** — execution — Bounded worker pool for all Core execution
  - Surfaced by: Outside voice 5 / 15A — zero concurrency controls; `dispatcher.py:149` deep-copies before execution
  - Files: `dtcc_agent/dispatcher.py`, `dtcc_agent/server.py`
  - Verify: N concurrent builds never exceed the configured pool
- [ ] **T9 (P1, human: ~2d / CC: ~3h)** — references — `object_ref` / `run_ref`, typed and prefixed; Run records its Object reference
  - Surfaced by: ADR-0010 + Test review regressions 1-3
  - Files: `dtcc_agent/` (wide), `tests/`
  - Verify: a Run reference where an Object reference is expected is refused
- [ ] **T10 (P2, human: ~2d / CC: ~3h)** — registry — Build the catalogue in `lifespan`; drop the memoised global
  - Surfaced by: Code quality 7A — `registry.py:380-387` latches whichever mode won the first request
  - Files: `dtcc_agent/registry.py`, `dtcc_agent/server.py`
  - Verify: catalogue contents are identical regardless of first-request timing
- [ ] **T11 (P2, human: ~1d / CC: ~2h)** — memory — LRU + budget on `_results`; global budget with a per-session cap
  - Surfaced by: Performance 10A, 11A
  - Files: `dtcc_agent/server.py`, `dtcc_agent/object_store.py`
  - Verify: unbounded simulation runs do not grow the process without limit
- [ ] **T12 (P2, human: ~1d / CC: ~1h)** — cache — Stamp schema + Core version; discard on mismatch; read fields with `.get()`
  - Surfaced by: Architecture 5A — pickled Core objects, 7-day TTL, no version guard
  - Files: `dtcc_agent/disk_cache.py`
  - Verify: an index written pre-upgrade produces a cold start, not an exception
- [ ] **T13 (P2, human: ~1d / CC: ~2h)** — deployment — Two-service Dockerfile and compose
  - Surfaced by: Architecture 4A — `CMD ["uvicorn", "chatbot.app:app"]`, one service
  - Files: `Dockerfile`, `docker-compose.yml`
  - Verify: `docker compose up` yields a working system
- [ ] **T14 (P2, human: ~2d / CC: ~3h)** — auth — Admission secret; separate subject and session identifiers
  - Surfaced by: Outside voice 4 / 14A — anonymous tokens anyone can mint are not authentication
  - Files: `dtcc_agent/server.py`, `chatbot/app.py`, `docs/adr/0004-*.md`
  - Verify: a session cannot be minted without the secret

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | not run | — |
| Outside Review | codex (auto, plan-review) | Independent 2nd opinion | 1 | completed | 7 findings, 7 folded |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | issues_open | 15 issues, 1 critical gap |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | not run | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | not run | — |

**OUTSIDE COVERAGE:** provider `codex` (model `gpt-6-astra`), phase `plan-review`, status
`completed`. 7 findings, all folded. Three were internal contradictions corrected directly
(M1's CLI-free acceptance, M3's dependence on a deferred correctness layer, the UI rebuild
preceding D1-D3). Four became decisions: shared-cache isolation, filesystem escape, admission
boundary, execution concurrency. **The cache finding invalidated a claim in ADR-0004 that this
review had accepted**, which is the clearest case for running the outside voice at all.

**CROSS-MODEL:** native review (claude) found 11 issues; codex found 7, of which 0 duplicated the
native set. Overlap zero — the two passes found disjoint problems. Native found infrastructure and
sequencing (uvloop, undeclared Core, no rollback, memoised catalogue); codex found boundary
correctness (cache contents, filesystem paths, admission, execution). Model identity known for
both.

**VERDICT:** ENG CLEARED with concerns — 15 findings, all resolved and folded into a revised
5-milestone plan. CEO, Design and DX reviews not run. One accepted critical gap (abandoned work,
`TODOS.md` T-002). Plan is implementable; start at T1.

NO UNRESOLVED DECISIONS
