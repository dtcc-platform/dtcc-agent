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

### M1 — transport, state, tests

Scope: HTTP transport; per-session stores injected rather than global, with the hybrid split from
ADR-0004; tests for the tool surface; Lurkie changed minimally to speak HTTP instead of spawning
a subprocess.

Acceptance: the 122 existing tests stay green; `server.py`'s replacement has direct coverage; two
concurrent sessions cannot see each other's objects, runs or conversation memory; the public
bounds-derived cache still hits across sessions; Lurkie works with no `claude` CLI on the machine.

Not in scope: changing which tools exist, or what they return.

### M2 — auth and provenance

Scope: authentication on the HTTP surface; anonymous session tokens carried end to end
(ADR-0004); a provenance record per answer; the measurement layer of the evaluation harness
reading it (ADR-0008).

Acceptance: an unauthenticated request is refused; every answer carries model, prompt version,
catalogue revision and the operations that ran; the harness reports latency, tokens and cost per
task for a fixed question set.

### M3 — pydantic-ai

Scope: replace the Claude Agent SDK (ADR-0003); prompt caching (ADR-0006), which deletes the
seven hardcoded schemas in `chatbot/config.py`; Lurkie's real rebuild against the stabilised API.

Acceptance: the harness reports before-and-after latency and cost across the migration; the
scenario suite scores no worse than the M2 baseline; a second provider runs the same suite.

Sequenced last on purpose: this is the change most likely to produce a long debugging tail, and
the harness should be measuring a stable system before the model runtime moves underneath it.

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
