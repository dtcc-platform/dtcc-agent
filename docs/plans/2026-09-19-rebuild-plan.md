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

## Eng review 2026-09-23 — M1 only

Second eng-review pass, scoped to M1 (T4-T13). M0 is complete and merged (PRs #7, #8):
`dtcc-core` declared and pinned to `18eb176`, CI green on `develop` and on PRs, and
`tests/test_server.py` holds 61 characterisation tests of the current 22-tool surface.
ADR-0001..0010 and the M0/M4 split are accepted and were not reopened.

### Scope record

feature answers: none proposed (no feature cuts offered; all ten M1 tasks retained);
structure: A) Smaller arrangement (D1, answered 2026-09-23);
accepted scope: M1 splits into two milestones carrying the same ten tasks and the same
contracts. **M1a** = T4, T5, T6, T7, T8, T10, T11, T13. **M1b** = T9, T12 (amended by R1/D2,
which moved T8 into M1a). M1b may land before or after M1a, but not concurrently: both
milestones rewrite `server.py` (see C2).
pending remedies: S1, S2, S6, S7, S8.

### Scope Challenge findings

| # | Sev | Conf | Location | Finding | Disposition |
|---|---|---|---|---|---|
| S1 | P1 | 9/10 | `renderer.py:75` + `chatbot/app.py:62` | `render_object` writes to a per-call `mkdtemp("dtcc_screenshots_")`, the mount serves the fixed `/tmp/dtcc_screenshots`, and nothing builds a `/renders/` URL. Rendered images are unreachable over HTTP today | pending -> T7 |
| S2 | P1 | 9/10 | mcp 1.26.0 | `StreamableHTTPSessionManager` has no idle timeout, max-sessions or expiry. ADR-0004's session boundary and the QA plan's idle case rest on a built-in that does not exist | pending |
| S3 | P2 | 9/10 | plan T4 file list | Names `dispatcher.py`, which has no async code. The wrap belongs in `server.py`'s tools as `async def` | correction accepted |
| S4 | P2 | 9/10 | plan T10 rationale | "Latches whichever mode won the first request" is unreachable; the mode is env-static | correction accepted |
| S5 | P2 | 10/10 | ADR-0004 prose | Says four builders; there are five. `build_terrain_raster` takes `pc`, a user-supplied PointCloud | correction accepted |
| S6 | P2 | 8/10 | M1 acceptance | T4 is invisible until T5 and no existing test can catch it; the uvloop download acceptance is unverifiable as written | pending |
| S7 | P3 | 8/10 | `disk_cache.py:28-37` | `CACHE_ALLOWLIST` mixes namespaces: `get_buildings` is an MCP tool name, the other seven are Core operation names | pending |
| S8 | P2 | 9/10 | `AGENTS.md` | Still claims `dtcc-core` is undeclared and "M0 fixes this" — false since #7 merged | pending |

### What already exists (reuse, do not rebuild) — [Layer 1]

FastMCP 1.26.0 already provides `run(transport="streamable-http")`, `lifespan`,
`stateless_http`, `auth` / `token_verifier`, `event_store`, `streamable_http_path`, `host`
and `port`. T5 wires these; it does not build transport or state plumbing. `ObjectStore`
already carries the LRU and byte budget T11 wants, so T11 extends it rather than adding a
parallel mechanism.

### Architecture / code-quality findings (native)

| # | Sev | Conf | Location | Finding |
|---|---|---|---|---|
| A1 | P1 | 9/10 | `Dockerfile:3,28` + `docker-compose.yml` | `DTCC_CORE_REF` defaults to the floating `develop` branch and Core is installed at line 28, before `pip install -e .` at line 29. The container either bypasses M0's SHA pin or silently ignores its own build arg. Which one wins is **unverified** (needs a container build); both outcomes are defects. T13 edits this file |
| A2 | P1 | 9/10 | anyio limiter, `dispatcher.py:149-150,61-65`, `object_store.py:55` | anyio's default thread limiter is 40. T4 raises peak concurrent Core execution from 1 to 40, each deep-copying heavy geometry against a 2 GiB budget. Resolved by R1/D2 |
| A3 | P1 | 9/10 | mcp 1.26.0 | No session expiry, so sessions live for process lifetime. Per-session stores (T5) make the *number* of sessions an unbounded growth path. T11 caps each session, not the count |
| A5 | P2 | 8/10 | M1a vs T14 | M1a opens an HTTP MCP surface with no admission control until M2. `docker-compose.yml` already sets `DTCC_AGENT_HOST=0.0.0.0` |
| A6 | P2 | 8/10 | ADR-0004 / T5 | The MCP session is between the chatbot backend and the server, not the browser. "One MCP session per browser" needs a browser-to-MCP-session mapping the plan assigns to no component |
| A7 | P1 | 9/10 | `chatbot/config.py:66-72` | **T13 depends on T5, and the plan says so nowhere.** `get_mcp_server_config()` returns `type: stdio` and spawns `sys.executable -m dtcc_agent` as a child process. The container cannot split into two services until that config points at an HTTP URL, and `Dockerfile:37` runs only `uvicorn chatbot.app:app`. Third hidden ordering constraint, after T4-before-T5 and T8-with-T4 |
| A8 | P2 | 9/10 | `chatbot/config.py:66-72` | Because the MCP server is one stdio subprocess of the chatbot, **every browser session today shares one `_object_store` and one `_results`**. This is the concrete isolation defect ADR-0004 exists to close, and it has no test |
| A4 | P3 | 9/10 | `.mcp.json` | Hardcodes `/Users/vasnas/miniconda3/bin/conda` and `fenicsx-env`. **Downgraded from P2** after reading `chatbot/config.py`: the runtime path is `get_mcp_server_config()`, which uses `sys.executable` and is portable. `.mcp.json` is a developer-client file only, so M1's rollback sentence names the wrong artifact rather than resting on a broken one |
| C1 | P2 | 10/10 | `server.py` | The identical object-lookup + "not found. Use list_objects()" block appears 7 times. T9 should collapse it into one resolver; the characterisation tests pin the message |
| C2 | P2 | 9/10 | `server.py` (1190 lines) | M1a (T5/T7/T10/T11) and M1b (T9, wide) both rewrite it. Sequential is fine; parallel worktrees would conflict badly |

### Performance findings (native)

| # | Sev | Conf | Location | Finding |
|---|---|---|---|---|
| P1 | P2 | 9/10 | `registry.py:379-385` | Measured: **1.355s** cold catalogue build, 0.4 microseconds memoised. The first request after every restart pays it, and a build failure surfaces as a request error. This replaces T10's incorrect stated rationale (S4) |
| P2 | P1 | 9/10 | see A2 | 40x concurrency multiplier on the memory-heaviest path |
| P3 | P2 | 8/10 | `server.py:34` | `_results` is unbounded while `ObjectStore` is capped, yet `_store_result` writes the same result to both |
| P4 | P2 | 8/10 | `dispatcher.py:150` | `deepcopy` doubles peak memory per operation before any concurrency multiplier |

### Outside voice (codex, gpt-6-astra) — 9 findings

Run read-only against this repo. Full output in the review transcript. Three findings were
re-verified natively by probe before being recorded here; the verification changed two of them.

| # | Sev | Conf | Location | Finding | Native verification |
|---|---|---|---|---|---|
| X1 | P1 | 10/10 | `registry.py:334`, dispatcher | T7 leaves a filesystem bypass through `run_operation`. Core's I/O operations are in the catalogue and the dispatcher forwards their path arguments, so removing paths from `export_object` and `load_geojson` does not establish the boundary | **Confirmed and worse.** Probed the live registry: **16** operations take a path-like parameter, not the two codex named — `io.load_3dbag`, `io.load_model`, `io.save_model`, `io.load_mesh`, `io.save_mesh`, `io.load_volume_mesh`, `io.save_volume_mesh`, `io.load_pointcloud`, `io.save_pointcloud`, `io.load_raster` and six more. The `save_*` half is arbitrary file **write**. T7 as written closes 2 of 18 path surfaces |
| X2 | P1 | 8/10 | `chatbot/memory.py:63`, `chatbot/app.py:236` | Conversation isolation has no implementation task. Memory retrieval is global and is injected into fresh conversations; neither per-session MCP stores nor two MCP clients exercise that path | Not independently probed. Consistent with A8 (one shared MCP subprocess today) |
| X3 | P1 | 8/10 | `disk_cache.py:40` | Session-local builder caches still return incorrect geometry. Codex probed two key collisions: different objects with identical type/source/size/label collide, and the same raster with different requested `bounds` hashes identically because bounds are removed unconditionally | Not re-probed. Consistent with the confirmed fact that `content_fingerprint` hashes metadata only |
| X4 | P1 | 8/10 | `server.py:124`, `runner.py:304` | A shared-cache hit does not prove valid spatial reuse. `get_buildings` containment hits replace only the returned `bounds` while counts and statistics still describe the larger area, and the cached summary lacks the coordinates needed to reconstruct the smaller result | Not re-probed. This is the ADR-0007 crop-from-containing-bounds behaviour producing a wrong answer, not just a stale one |
| X5 | P1 | 10/10 | `object_store.py:18` | `_estimate_bytes` ignores dict/list contents, so T11's budget cannot establish its guarantee | **Confirmed by probe.** A GeoJSON-like dict whose JSON is 64,168 bytes is estimated at **64 bytes** — a 1000x undercount. Every budget built on this estimator is fictional for dict and list results |
| X6 | P2 | 8/10 | `server.py:52` | T9 and T11 share an unresolved ownership decision. Independent LRUs can evict an Object while its Run retains the payload, so `object_ref` would dangle; deleting the Object need not release memory | Consistent with P3 and with the characterisation test that pins the double-store |
| X7 | P2 | 10/10 | mcp `lowlevel/server.py:657`, `streamable_http_manager.py:181,248` | **T10 targets the wrong lifecycle.** FastMCP's lifespan is entered inside `Server.run()`, and the HTTP manager calls `app.run()` per session (stateful) or per request (stateless). Building the catalogue there runs it per session, not once per process | **Confirmed by reading the installed source.** Line 181 is `run_stateless_server` (per request); line 248 is `run_server` after the session id is registered (per session). Neither is process-scoped |
| X8 | P2 | 7/10 | `renderer.py:67`, Dockerfile | T7/T13 omit the rendering runtime dependency: `dtcc_viewer` plus a GLFW/OpenGL context is not provisioned, so an artifact route can pass while rendering still fails | Not probed. Compounds S1, where the artifact route is broken independently |
| X9 | P2 | 8/10 | `tests/test_server.py` | The compatibility acceptance is internally inconsistent: T7 changes path inputs, T9 changes reference fields, yet M1 excludes changes to what tools return. The characterisation tests pin the current field names | Consistent with the REGRESSION RULE flag on T9 |

### Correction to this review's own finding P1

**X7 overturns the conclusion I drew from the 1.355s measurement.** I recorded that the cold
catalogue build was "T10's real justification". It is not. Because the lifespan is entered per
session (or per request in stateless mode), moving the build into lifespan converts a one-time
1.355s process cost into 1.355s **per session**. T10 as written makes the measured problem
roughly N times worse, where N is the session count.

The measurement stands; the inference from it does not. Process-scoped initialisation belongs at
process start, not in the MCP lifespan. The same applies to T8's worker limiter and T11's global
budget: placed in lifespan they would become per-session and bound nothing globally. This does not
disturb R1/D2, which decided which milestone owns T8, not where its limiter lives.

### Cross-model tension

Native review found 22 issues (8 Scope Challenge, 10 Architecture/Code Quality, 4 Performance).
Codex found 9. **Overlap: zero.** For the third consecutive run on this plan the outside voice
returned a disjoint set, and this time it also corrected a native finding.

The split is consistent: native found sequencing, infrastructure and lifecycle coupling (the
T4/T5/T8/T13 ordering constraints, the Dockerfile pin bypass, the 40-thread multiplier, the
missing session expiry). Codex found correctness inside the boundaries those tasks draw (paths
that bypass the boundary, cache keys that collide, a byte estimator that under-reports by 1000x,
a lifespan that is not process-scoped). Neither pass would have found the other's set.

Model identity known for both: native claude, outside codex `gpt-6-astra`.

## Decision ledger

### R1: Which milestone owns T8 (bounded worker pool)
Finding: A2, P1, confidence 9/10, anyio default thread limiter + `dispatcher.py:149-150`, reviewer: native (claude)
Plan baseline: T8 assigned to M1b (transport-independent) by D1, answered 2026-09-23, Smaller arrangement.
Runtime evidence: `anyio.to_thread.current_default_thread_limiter().total_tokens` == 40, probed in
this repo's venv. `dispatcher.py:149-150` deep-copies when `_should_copy` matches, and
`_should_copy` (dispatcher.py:61-65) covers PointCloud, Mesh, VolumeMesh, Raster, City, Terrain,
Surface and MultiSurface. `ObjectStore.__init__` (object_store.py:55) defaults `max_bytes` to
2 GiB. Today Core execution is serialised on the event loop, so effective concurrency is 1.

Comparison grid:

| Choice | Current (after D1) | A | B |
|---|---|---|---|
| R1 T8 milestone | M1b | M1a, alongside T4 | M1b, unchanged |
| Peak concurrent Core ops after M1a ships | 40 (unbounded by us) | bounded by the configured pool | 40 until M1b lands |
| T4 ships without an execution bound | yes | no | yes |
| M1b contents | T8, T9, T12 | T9, T12 | T8, T9, T12 |
| M1a contents | T4, T5, T6, T7, T10, T11, T13 | T4, T5, T6, T7, T8, T10, T11, T13 | unchanged |
| Other D1 commitments | same ten tasks, same contracts | unchanged | unchanged |

Question D2:
D2 - Move T8 (bounded worker pool) into M1a?

Project/branch/task: dtcc-agent on `develop`, reopening one row of the M1 arrangement approved in D1.

ELI10: Right now every dtcc-core operation runs one at a time, because it runs directly on the
server's event loop. T4 fixes the uvloop crash by pushing that work onto worker threads, and the
library's default allowance is forty of them. So T4 does not just fix a crash, it quietly raises
how many heavy geometry operations can run at once from one to forty, and each of those copies a
point cloud or mesh in memory first. T8 is the task that puts a ceiling back. In D1 you placed T8
in the second milestone, which would ship the raised ceiling without the limit.

Stakes if we pick wrong: M1a ships a fortyfold concurrency increase on the memory-heaviest path in
the system, against a 2 GiB object store. The likely symptom is the server being OOM-killed under
a handful of simultaneous users, which looks like a crash rather than a capacity limit.

Recommendation: A because T4 is what raises the ceiling, so the task that bounds it belongs in the
same milestone.

Note: options differ in kind, not coverage - no completeness score. Both keep all ten tasks and
every D1 commitment; only T8's milestone changes.

Pending remedies not decided here: S1, S2, S6, S7, S8, A1, A3, A4, A5, A6, C1, C2.
Header: T8 milestone
Options:
A) Move T8 into M1a (recommended)
M1a becomes T4, T5, T6, T7, T8, T10, T11, T13. M1b becomes T9 and T12. The execution bound ships
in the same milestone as the thread-wrap that makes it necessary, so peak concurrent Core
operations is a number we chose rather than anyio's default of 40. Costs one more task in the
larger milestone (human: ~2d / CC: ~3h).

B) Keep T8 in M1b
M1a stays T4, T5, T6, T7, T10, T11, T13 and M1b stays T8, T9, T12, exactly as D1 approved. M1a
ships with Core execution bounded only by anyio's default 40-thread limiter. Acceptable if M1a is
not exposed to concurrent users before M1b lands, which requires someone to hold that line.

State: approved
Actual answer: A) Move T8 into M1a (D2, answered 2026-09-23)
Accepted scope: T8 (bounded worker pool) moves from M1b to M1a. **M1a** = T4, T5, T6, T7, T8, T10,
T11, T13. **M1b** = T9, T12. The execution bound ships in the same milestone as the thread-wrap
that raises concurrency, so peak concurrent Core operations is a configured number rather than
anyio's default of 40. All ten tasks and every other D1 commitment are unchanged.
History: D1 (2026-09-23) placed T8 in M1b as a transport-independent task, with
`State: pending / Actual answer: unanswered / Accepted scope: none` before D2. Reopened on A2,
which established that T8 is coupled to T4 through the thread limiter, not to the transport.

Approval readiness: PASS. Checked IDs: D1 (M1 arrangement, answered 2026-09-23, Smaller
arrangement) and R1/D2 (T8 into M1a, answered 2026-09-23, option A). No other remedy is marked
accepted: every finding S1-S8, A1-A8, C1-C2, P1-P4 and X1-X9 remains pending, and their open
choices are listed as U1-U11. This review was asked to report for amendment, so no remedy was
auto-approved.

### Test review (M1)

```
CODE PATHS                                              COVERAGE
[M1a] T4  tool -> anyio.to_thread -> Core asyncio.run
  +-- uvloop + real download succeeds                   [GAP] [->E2E] no network test exists
  +-- nest_asyncio removed, stdio still works           [GAP]
[M1a] T5  streamable-http + lifespan + session stores
  +-- two clients, neither sees the other's objects     [GAP] [->E2E] test_chatbot_sessions.py
  |                                                             covers a chatbot id registry,
  |                                                             NOT MCP isolation
  +-- conversation memory isolated across turns         [GAP] [->E2E] global today (X2)
  +-- rollback via config to stdio                      [GAP]
[M1a] T6  cache split + key correctness
  +-- 3 downloads shared, 5 builders session-local      [GAP]  no test references CACHE_ALLOWLIST
  +-- nested-bounds cold vs warm equivalence            [GAP]  wrong answers today (X4)
  +-- distinct objects, same metadata, must not collide [GAP]  collides today (X3)
[M1a] T7  artifact dirs + download route
  +-- no tool accepts a caller path                     [GAP]  16 registry ops do (X1)
  +-- a render is reachable over HTTP                   [GAP]  broken today (S1)
[M1a] T8  bounded worker pool                           [GAP]  no concurrency test anywhere
[M1a] T10 process-scoped catalogue                      [**  TESTED] test_registry.py (24) covers
                                                               contents, not lifecycle (X7)
[M1a] T11 per-session cap + global budget
  +-- store-level LRU + byte budget                     [*** TESTED] test_object_store.py:70,76
  +-- accounting reflects real size                     [GAP]  1000x undercount (X5)
[M1a] T13 two-service compose                           [GAP]  pin bypassed (A1), needs T5 (A7)
[M1b] T9  typed references                              [*** TESTED] test_server.py pins TODAY's
                                                               behaviour - 4 tests MUST flip
[M1b] T12 cache versioning                              [**  TESTED] TTL + restart covered,
                                                               version mismatch not

COVERAGE: 3/17 paths tested (18%)  |  GAPS: 14 (5 E2E)
QUALITY: ***:2  **:2  |  Legend: *** behaviour+edge+error, ** happy path, * smoke
```

**REGRESSION RULE — T9 is the flagged risk.** Four characterisation tests in `test_server.py`
assert today's wrong behaviour (unrelated ids, missed-not-refused, bare-list envelope). T9 must
flip them deliberately. Deleting them instead would erase the only written description of what
the rebuild changed. This contract is **unapproved**; it needs its own decision before T9 starts.

### Failure modes

| New path | Realistic production failure | Test? | Error handling? | User sees |
|---|---|---|---|---|
| T4 thread-wrap | Download raises under uvloop | no | no | **CRITICAL GAP** — silent until first download |
| T5 session stores | Two users share one store | no | no | **CRITICAL GAP** — silent data leak |
| T6 builder cache | Wrong geometry served from a colliding key | no | no | **CRITICAL GAP** — a confidently wrong answer |
| T6 shared cache | Nested-bounds hit returns the larger area's statistics | no | no | **CRITICAL GAP** — a confidently wrong answer |
| T7 paths | Path reaches a Core I/O op through `run_operation` | no | no | **CRITICAL GAP** — arbitrary read and write |
| T11 budget | Dict/list result under-counted 1000x, budget never trips | no | no | **CRITICAL GAP** — OOM presented as a crash |
| T10 lifespan | Catalogue rebuilt per session, 1.355s each | no | no | slow, visible, not silent |
| T13 container | Floating Core installed instead of the pin | no | no | silent version drift |

**Eight critical gaps** (no test, no error handling, silent or wrong-answer). Six of the eight
produce a wrong answer or a leak rather than a crash, which is the worst failure shape for a tool
whose output a person will act on.

### NOT in scope

- M2, M3, M4 and T14. Only M1 was reviewed.
- ADR-0001..0010 and the M0/M4 split: accepted, not reopened.
- Rewriting `content_fingerprint` into a real content hash. Codex's cheaper suggestion (object
  identity plus revision and complete parameters) is recorded but unapproved.
- Fixing the 16 path-taking Core operations at the Core level. That is a dtcc-core change, not a
  dtcc-agent one; the remedy here is a boundary in the dispatcher.

### Worktree parallelization

| Step | Modules touched | Depends on |
|---|---|---|
| T4 thread-wrap | `dtcc_agent/` (server, main) | — |
| T8 worker pool | `dtcc_agent/` (dispatcher, server) | T4 |
| T5 transport/session | `dtcc_agent/`, `chatbot/` | T4 |
| T6 cache | `dtcc_agent/` (disk_cache) | T5 |
| T7 artifacts/paths | `dtcc_agent/`, `chatbot/` | T5 |
| T10 catalogue | `dtcc_agent/` (registry, server) | T5 |
| T11 memory | `dtcc_agent/` (server, object_store) | T5 |
| T13 two-service | root (Dockerfile, compose), `chatbot/config.py` | **T5 (A7)** |
| T9 typed refs | `dtcc_agent/` wide, `tests/` | — |
| T12 cache version | `dtcc_agent/` (disk_cache) | — |

Lane A: T4 -> T8 -> T5 -> {T6, T7, T10, T11} -> T13 (sequential, shared `dtcc_agent/`)
Lane B: T12 (independent, small)

**Execution order:** T12 may run in its own worktree at any time. Everything else is one
sequential lane, because T5 fans out to five dependents and every one of them rewrites
`server.py`. T9 (M1b) is a separate lane in time, not in parallel: it also rewrites `server.py`.

**Conflict flags:** `server.py` (1190 lines) is touched by T5, T7, T8, T9, T10 and T11. Parallel
worktrees across those tasks would conflict continuously. Run them sequentially and keep M1b
strictly before or after M1a.

## Implementation Tasks

Synthesized from this review's findings. Each derives from a specific finding above.
**Every task below is unapproved.** They are the shape of the remedy, not accepted scope.

- [ ] **T15 (P1, human: ~4d / CC: ~6h)** — security — Constrain caller-supplied paths in generic dispatch, not only in two tools
  - Surfaced by: X1 — probed: 16 registry operations take a path-like parameter; the `save_*` half is arbitrary write
  - Files: `dtcc_agent/dispatcher.py`, `dtcc_agent/registry.py`, `dtcc_agent/server.py`
  - Verify: a read and a write attempted through `run_operation` are both refused
- [ ] **T16 (P1, human: ~2h / CC: ~15min)** — build — Make the container install the pinned Core
  - Surfaced by: A1 — `Dockerfile:3,28` defaults `DTCC_CORE_REF=develop` and installs before `pip install -e .`
  - Files: `Dockerfile`, `docker-compose.yml`
  - Verify: built image reports the pinned SHA via `direct_url.json`
- [ ] **T17 (P1, human: ~1d / CC: ~2h)** — lifecycle — Put process-scoped init at process start, not in the MCP lifespan
  - Surfaced by: X7 — lifespan is entered per session (`streamable_http_manager.py:248`) or per request (`:181`)
  - Files: `dtcc_agent/server.py`, `dtcc_agent/registry.py`
  - Verify: catalogue built once per process; measured cost paid once, not per session
- [ ] **T18 (P1, human: ~3d / CC: ~4h)** — cache — Correct the cache keys before reusing builder results
  - Surfaced by: X3, X4 — metadata-only fingerprints collide; nested-bounds hits return the larger area's statistics
  - Files: `dtcc_agent/disk_cache.py`, `dtcc_agent/server.py`, `dtcc_agent/runner.py`
  - Verify: cold vs warm equivalence for nested bounds; two same-metadata different-content objects never substitute
- [ ] **T19 (P1, human: ~1d / CC: ~2h)** — memory — Make byte accounting reflect real size
  - Surfaced by: X5 — probed: a 64,168-byte dict estimates at 64 bytes
  - Files: `dtcc_agent/object_store.py`
  - Verify: dict, list and `.x.array` results are counted within an order of magnitude of real size
- [ ] **T20 (P1, human: ~1d / CC: ~2h)** — tests — A real download test under uvicorn+uvloop
  - Surfaced by: S6 — T4's acceptance has nothing that could verify it
  - Files: `tests/`
  - Verify: the test fails with `nest_asyncio` restored and passes with the thread-wrap
- [ ] **T21 (P1, human: ~1d / CC: ~2h)** — artifacts — Make rendered images reachable over HTTP
  - Surfaced by: S1 — `renderer.py:75` mkdtemp vs `chatbot/app.py:62` fixed mount; no `/renders/` URL is ever built
  - Files: `dtcc_agent/renderer.py`, `chatbot/app.py`
  - Verify: a browser loads the PNG the tool reports
- [ ] **T22 (P2, human: ~2d / CC: ~3h)** — session — Decide and implement session lifetime
  - Surfaced by: S2, A3 — mcp 1.26.0 has no idle timeout, max-sessions or expiry
  - Files: `dtcc_agent/server.py`
  - Verify: sessions do not accumulate without limit
- [ ] **T23 (P2, human: ~2d / CC: ~3h)** — isolation — Give conversation memory an owner
  - Surfaced by: X2 — `chatbot/memory.py:63` retrieves globally into fresh conversations
  - Files: `chatbot/memory.py`, `chatbot/app.py`
  - Verify: two chatbot sessions, multiple turns each, no cross-talk
- [ ] **T24 (P2, human: ~1d / CC: ~2h)** — references — Settle Run/Object payload ownership with T9
  - Surfaced by: X6, P3 — independent LRUs can evict an Object while its Run holds the payload
  - Files: `dtcc_agent/server.py`, `dtcc_agent/object_store.py`
  - Verify: no dangling `object_ref` after eviction; deleting an Object releases its memory
- [ ] **T25 (P2, human: ~1d / CC: ~2h)** — deployment — Provision the rendering stack
  - Surfaced by: X8 — `dtcc_viewer` plus a GLFW/OpenGL context is not in the image
  - Files: `Dockerfile`, `pyproject.toml`
  - Verify: the final container generates and serves a real PNG
- [ ] **T26 (P2, human: ~1d / CC: ~1h)** — deployment — Point the chatbot at an MCP URL before splitting the container
  - Surfaced by: A7 — `chatbot/config.py:66-72` returns `type: stdio`; `Dockerfile:37` runs only the chatbot
  - Files: `chatbot/config.py`, `Dockerfile`, `docker-compose.yml`
  - Verify: the two services talk over HTTP; T13 cannot land before this
- [ ] **T27 (P2, human: ~2h / CC: ~15min)** — quality — Collapse the seven duplicated lookup blocks
  - Surfaced by: C1 — the identical block appears 7 times in `server.py`
  - Files: `dtcc_agent/server.py`
  - Verify: suite green; the characterisation tests pin the message
- [ ] **T28 (P2, human: ~15min / CC: ~2min)** — docs — Correct `AGENTS.md`
  - Surfaced by: S8 — still says Core is undeclared and "M0 fixes this"
  - Files: `AGENTS.md`
  - Verify: the file describes `develop` as it is
- [ ] **T29 (P3, human: ~2h / CC: ~15min)** — cache — Resolve the allowlist namespace mix
  - Surfaced by: S7 — `get_buildings` is a tool name, the other seven are operation names
  - Files: `dtcc_agent/disk_cache.py`
  - Verify: every allowlist entry resolves in one namespace
- [ ] **T30 (P2, human: ~1d / CC: ~1h)** — contract — State M1's intentional contract changes
  - Surfaced by: X9 — T7 and T9 change inputs and fields while M1 excludes changes to returns
  - Files: `docs/plans/`, `tests/test_server.py`
  - Verify: each changed field has a named before/after and an updated assertion
- [ ] **T31 (P2, human: ~2h / CC: ~15min)** — security — Keep the MCP surface off public interfaces until T14
  - Surfaced by: A5 — M1a opens HTTP with no admission control; compose sets `0.0.0.0`
  - Files: `docker-compose.yml`, `chatbot/config.py`
  - Verify: the MCP port is not publishable without an explicit override
- [ ] **T32 (P3, human: ~15min / CC: ~2min)** — docs — Fix M1's rollback sentence
  - Surfaced by: A4 — it names `.mcp.json`, a machine-bound developer file; the runtime path is `get_mcp_server_config()`
  - Files: `docs/plans/2026-09-19-rebuild-plan.md`
  - Verify: the rollback names the artifact that actually carries it

### Unresolved decisions that may bite you later

Every remedy below is **unapproved**. Two decisions were answered this review (D1 split, D2 T8
into M1a); nothing else was accepted, because the request was to report for amendment.

- **U1 (X1/T15)** — how far the path boundary goes: all 16 path-taking operations, an allowlist, or a sandboxed root.
- **U2 (X3+X4/T18)** — fix the cache keys, or disable builder caching until they are correct.
- **U3 (X7/T17)** — where process-scoped init lives once lifespan is ruled out.
- **U4 (X5/T19)** — what accuracy the byte budget must reach, and what happens to an oversized result.
- **U5 (X2/T23)** — which component owns conversation memory.
- **U6 (X6/T24)** — Run/Object payload ownership and eviction semantics.
- **U7 (S2/T22)** — session lifetime, given no built-in expiry exists.
- **U8 (X9/T30)** — the intentional contract changes, and how the four pinned characterisation tests flip.
- **U9 (S1+X8/T21,T25)** — whether rendering is fixed in M1a or deferred with the feature disabled.
- **U10 (A1/T16)** — verify which Core install wins in the container before choosing the fix.
- **U11 (A5/T31)** — the interface the M1a MCP surface binds to before T14.

### Completion summary

- Step 0: Scope Challenge — scope reduced per recommendation (M1 split into M1a/M1b; no features cut)
- Architecture Review: 8 issues found
- Code Quality Review: 2 issues found
- Test Review: diagram produced, 14 gaps identified
- Performance Review: 4 issues found
- NOT in scope: written
- What already exists: written
- TODOS.md updates: 0 items proposed (no TODO file change proposed; all work is task-shaped)
- Failure modes: 8 critical gaps flagged
- Unresolved decisions: 11 in this review
- Outside voice: codex (`gpt-6-astra`), completed, 9 findings, zero overlap with the native set,
  and it corrected one native finding (X7 vs P1)
- Parallelization: 2 lanes, 1 parallel / 1 sequential
- Lake Score: N/A — both answered choices differed in kind, not coverage

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | not run | — |
| Outside Review | codex (auto, plan-review) | Independent 2nd opinion | 2 | completed | 9 findings, 0 folded (all pending) |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 2 | ISSUES OPEN (PLAN) | 22 issues, 8 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | not run | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | not run | — |

**OUTSIDE COVERAGE:** provider `codex` (model `gpt-6-astra`), phase `plan-review`, status
`completed`, run 2026-09-23 against M1. 9 findings, **none folded** — this pass was asked to
report for amendment, so every remedy stays pending. Three findings were re-verified natively
before recording: X1 confirmed and widened (16 path-taking operations, not 2), X5 confirmed
exactly (a 64,168-byte dict estimates at 64 bytes), X7 confirmed by reading the installed mcp
source. The 2026-09-19 run (7 findings, all folded) is retained in the count.

**CROSS-MODEL:** native (claude) found 22 issues; codex found 9; **overlap zero**, the third
consecutive disjoint result on this plan. Native found sequencing, infrastructure and lifecycle
coupling — the T4-before-T5, T8-with-T4 and T13-after-T5 ordering constraints, the Dockerfile pin
bypass, the 40-thread concurrency multiplier, the absent session expiry. Codex found correctness
inside the boundaries those tasks draw — paths that bypass the boundary, cache keys that collide,
a byte estimator under-reporting by 1000x, a lifespan that is not process-scoped. Codex also
**corrected a native finding**: X7 overturned the inference this review drew from its own 1.355s
catalogue measurement. Model identity known for both.

**VERDICT:** ENG REVIEW NOT CLEAR — 22 native issues, 9 outside findings, 8 critical failure-mode
gaps, 11 unresolved decisions, none accepted. M0 remains complete and green. eng review required
before M1 implementation starts.

**UNRESOLVED DECISIONS:**
- U1 (X1/T15) — how far the path boundary goes: all 16 path-taking operations, an allowlist, or a sandboxed root
- U2 (X3+X4/T18) — fix the cache keys, or disable builder caching until they are correct
- U3 (X7/T17) — where process-scoped init lives, now that lifespan is ruled out
- U4 (X5/T19) — required accuracy of the byte budget, and behaviour on an oversized result
- U5 (X2/T23) — which component owns conversation memory
- U6 (X6/T24) — Run/Object payload ownership and eviction semantics
- U7 (S2/T22) — session lifetime, given mcp 1.26.0 has no built-in expiry
- U8 (X9/T30) — M1's intentional contract changes, and how the four pinned characterisation tests flip
- U9 (S1+X8/T21,T25) — whether rendering is fixed in M1a or deferred with the feature disabled
- U10 (A1/T16) — verify which Core install wins in the container before choosing the fix
- U11 (A5/T31) — which interface the M1a MCP surface binds to before T14
