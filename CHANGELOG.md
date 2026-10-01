# Changelog

Everything delivered on dtcc-agent since the rebuild started, newest first. Each entry says
what changed, why it matters, and how we know it works, so it can be walked through in a
team meeting without opening the code.

**Status:** ✅ merged to `develop` · 🔍 open pull request, in review · ⏳ decision or task still open

Last updated: 2026-10-01.

---

## At a glance

| When | What | Status |
|---|---|---|
| 2026-10-01 | Memory is counted from what objects really hold, and one budget is shared fairly by every user ([#66](https://github.com/dtcc-platform/dtcc-agent/pull/66), T11, fixes [#23](https://github.com/dtcc-platform/dtcc-agent/issues/23), implements U4) | 🔍 |
| 2026-10-01 | Rendered images appear in the chat, exports download from it, and no path typed in chat reaches the filesystem ([#63](https://github.com/dtcc-platform/dtcc-agent/pull/63), T7, fixes [#21](https://github.com/dtcc-platform/dtcc-agent/issues/21) and [#38](https://github.com/dtcc-platform/dtcc-agent/issues/38), decides U9) | ✅ |
| 2026-10-01 | The chat works with the current Claude client again: an outdated library failed every message ([#64](https://github.com/dtcc-platform/dtcc-agent/pull/64)) | ✅ |
| 2026-09-30 | Builder results are no longer cached, so a cache can't hand back the wrong geometry; builder calls are recorded to show whether correct keys are worth building ([#62](https://github.com/dtcc-platform/dtcc-agent/pull/62), decides U2 [#11](https://github.com/dtcc-platform/dtcc-agent/issues/11)) | ✅ |
| 2026-09-30 | The Docker image installs the dtcc-core we pin and test, not Core's moving `develop` ([#61](https://github.com/dtcc-platform/dtcc-agent/pull/61), fixes [#41](https://github.com/dtcc-platform/dtcc-agent/issues/41), decides U10 [#14](https://github.com/dtcc-platform/dtcc-agent/issues/14)) | ✅ |
| 2026-09-30 | The wiki describes typed references, the cache version stamp and Core's registration fixes ([#60](https://github.com/dtcc-platform/dtcc-agent/pull/60)) | ✅ |
| 2026-09-30 | dtcc-core moved to `bb95f2f`, with the three fixes we reported; the agent's stopgaps for two of them are gone ([#59](https://github.com/dtcc-platform/dtcc-agent/pull/59)) | ✅ |
| 2026-09-30 | A half-broken dtcc-sim service no longer leaves a new stray dataset behind on every retry ([#58](https://github.com/dtcc-platform/dtcc-agent/pull/58), fixes [#44](https://github.com/dtcc-platform/dtcc-agent/issues/44)) | ✅ |
| 2026-09-30 | References say what they name (`obj_…`, `run_…`), a wrong one is refused, and a run hands back its result's reference ([#57](https://github.com/dtcc-platform/dtcc-agent/pull/57), T9, fixes [#26](https://github.com/dtcc-platform/dtcc-agent/issues/26)) | ✅ |
| 2026-09-30 | After a dtcc-core upgrade the disk cache starts cold instead of loading the old Core's objects ([#56](https://github.com/dtcc-platform/dtcc-agent/pull/56), T12, fixes [#27](https://github.com/dtcc-platform/dtcc-agent/issues/27)) | ✅ |
| 2026-09-29 | A dtcc-sim service can no longer replace a Core dataset by reusing its name ([#55](https://github.com/dtcc-platform/dtcc-agent/pull/55), fixes [#45](https://github.com/dtcc-platform/dtcc-agent/issues/45)) | ✅ |
| 2026-09-29 | The README's token check works: it no longer runs a `verify_auth.py` that never existed ([#54](https://github.com/dtcc-platform/dtcc-agent/pull/54), fixes [#40](https://github.com/dtcc-platform/dtcc-agent/issues/40)) | ✅ |
| 2026-09-29 | Slope and aspect no longer check a cache they can never be stored in ([#53](https://github.com/dtcc-platform/dtcc-agent/pull/53), fixes [#42](https://github.com/dtcc-platform/dtcc-agent/issues/42)) | ✅ |
| 2026-09-29 | The wiki describes the cache as it is after #50 and #51 ([#52](https://github.com/dtcc-platform/dtcc-agent/pull/52)) | ✅ |
| 2026-09-29 | Building heights are real, bad bounds are refused, and the disk cache can't be tampered with ([#51](https://github.com/dtcc-platform/dtcc-agent/pull/51)) | ✅ |
| 2026-09-29 | Building counts for a smaller area inside a cached one are right, via either tool ([#50](https://github.com/dtcc-platform/dtcc-agent/pull/50), fixes [#39](https://github.com/dtcc-platform/dtcc-agent/issues/39)) | ✅ |
| 2026-09-28 | Restart the agent after redeploying dtcc-sim, decided on #46 ([#48](https://github.com/dtcc-platform/dtcc-agent/pull/48)) | ✅ |
| 2026-09-27 | A generated wiki of the codebase, for people and agents ([#47](https://github.com/dtcc-platform/dtcc-agent/pull/47)) | ✅ |
| 2026-09-27 | The catalogue is built once per process, and a broken Core install stops the server ([#43](https://github.com/dtcc-platform/dtcc-agent/pull/43)) | ✅ |
| 2026-09-26 | A limit on how much Core work runs at once, fair between users, and one download per tile ([#36](https://github.com/dtcc-platform/dtcc-agent/pull/36)) | ✅ |
| 2026-09-26 | The automated reviewer gets repo context, and trials broader code suggestions ([#37](https://github.com/dtcc-platform/dtcc-agent/pull/37)) | ✅ |
| 2026-09-25 | dtcc-core pin moved to Core's latest `develop`, picking up the upstream fixes ([#35](https://github.com/dtcc-platform/dtcc-agent/pull/35)) | ✅ |
| 2026-09-25 | Session isolation over HTTP: each user's objects, runs and memory kept apart ([#33](https://github.com/dtcc-platform/dtcc-agent/pull/33)) | ✅ |
| 2026-09-25 | Every tool runs off the event loop, so Core downloads work under the web server ([#32](https://github.com/dtcc-platform/dtcc-agent/pull/32)) | ✅ |
| 2026-09-24 | Automated first-pass review on every pull request ([#31](https://github.com/dtcc-platform/dtcc-agent/pull/31)) | ✅ |
| 2026-09-24 | M1 reviewed a second time, split into M1a and M1b, and the whole programme put on GitHub ([#9](https://github.com/dtcc-platform/dtcc-agent/pull/9)) | ✅ |
| 2026-09-21 | **M0 done:** dtcc-core pinned, CI running, 61 tests describing today's tool surface ([#7](https://github.com/dtcc-platform/dtcc-agent/pull/7), [#8](https://github.com/dtcc-platform/dtcc-agent/pull/8)) | ✅ |
| 2026-09-21 | Working conventions for issues, triage and agents ([#6](https://github.com/dtcc-platform/dtcc-agent/pull/6)) | ✅ |
| 2026-09-21 | Glossary, ten architecture decisions (ADRs) and the rebuild plan ([#3](https://github.com/dtcc-platform/dtcc-agent/pull/3)) | ✅ |
| 2026-09-21 | The server starts on a fresh install again ([#2](https://github.com/dtcc-platform/dtcc-agent/pull/2)) | ✅ |
| 2026-09-18 | Four Core and Sim defects reported upstream; all four fixed by the Core team, and now in our build | ✅ |
| 2026-09-14 | Assessment of what works today ([#1](https://github.com/dtcc-platform/dtcc-agent/issues/1)) | ✅ |

**Tests:** 112 before the rebuild → 189 after M0 → 194 with #32 → 212 with #33, all passing on the new Core pin → 229 with #36 → 280 with #43 → 296 with #50 → 327 with #51 → 328 with #53 → 329 with #55 → 335 with #56 → 342 with #57 → 344 with #58 → 343 with #59 → 344 with #62 → 401 with #63 → 418 with #66.

---

## 🔍 In review

### Memory is counted from what objects really hold, and one budget is shared fairly by every user · 2026-10-01 · [#66](https://github.com/dtcc-platform/dtcc-agent/pull/66)

**Before:** the memory budget counted only a few named arrays. 127 Lindholmen buildings, about
5 MB in memory, counted as 64 bytes, and a GeoJSON file the same. So the 2 GB budget never
limited the objects people actually fetch (#13). Each of the 8 Sessions got a fixed 256 MB
share, even when the other seven were idle, and simulation Runs piled up without limit.

**Now:**
- Sizes count everything an object holds: arrays, dict and list contents, and Core geometry. The
  same buildings count as 6.25 MB, in 16 ms.
- All Sessions share one 2 GB budget. When it is full, the least recently used object in any
  Session goes first, so an idle Session's memory goes to the people using the agent. One Session
  may hold up to 1 GB.
- A single result bigger than that is not kept. The user still gets its summary, with a note
  that it was too large to keep and no reference for later steps (U4). Before, it would have
  emptied the Session to make room.
- A Session keeps its last 100 simulation Runs.

**How we know it works:** the #13 probe (a 64 KB GeoJSON dict once counted as 64 bytes) is now a
test and counts within an order of magnitude. New tests cover eviction across Sessions, the
per-Session cap, a dropped Session handing its memory back, oversized results from operations,
simulations and GeoJSON, and the Run cap. On real data the new count tracks the pickled size:
6.25 vs 4.73 MB for buildings, 1.56 vs 1.54 MB for a point cloud. 418 tests pass.

---

## ✅ Merged

### Rendered images appear in the chat, exports download from it, and no path typed in chat reaches the filesystem · 2026-10-01 · [#63](https://github.com/dtcc-platform/dtcc-agent/pull/63)

**Before:** `render_object` had never produced an image anyone could see. Its drawing library,
dtcc-viewer, was not installed, so it failed quietly. Even when it did draw, the file landed in
a random temporary folder that the chat's `/renders` route never served, and nothing told the
page an image existed (#38). `export_object` wrote wherever it was told, all users' exports shared
one folder, and `load_geojson` read any file on the machine. Eighteen Core operations reachable
through `run_operation` (the `io.load_*` / `io.save_*` family and two debug-folder options) took a
file path straight from chat, half of them for writing (#21, U1).

**Now:**
- Every Core operation that takes a file path is refused in `run_operation` (U1). A test lists
  the eighteen, so a Core upgrade that adds one fails the suite.
- Each chat session gets its own private folder. `render_object` and `export_object` write there
  under an unguessable name and return that name, never a path. The chat page shows an image in
  the conversation and a file as a download link. Another session's link returns nothing, and a
  session's files are deleted when it expires. Until central sign-in (T14), the link itself is
  the credential.
- `load_geojson` reads only from the shared results folder, where dtcc-sim writes.
- Rendering uses matplotlib (U9): footprints, roads and lines are drawn in plan view, meshes and
  point clouds in 3D, rasters as an image. It needs no graphics card or display, so it works the
  same on a laptop and in the container. dtcc-viewer was set aside because it requires Core's
  moving `develop`, which conflicts with our pin. Whether the page draws geometry itself is still
  D1/D2 in M3; only the drawing code would change.

**How we know it works:**
- In a real browser, the chat was asked for the Lindholmen buildings: it downloaded them,
  rendered 13 footprints, and the picture appeared in the conversation.
- The same image link from another session returns 404. `io.save_mesh` with a path is refused,
  and so is `load_geojson("/etc/passwd")`.
- The live run caught two faults the unit tests had missed, both now fixed and tested:
  `datasets.buildings` returns a building collection that no renderer handled, and the Claude
  client wraps each tool result in one more layer, which hid the image from the page.
- 401 tests pass.

**Still open:**
- Exporting a building collection is not supported yet (#65).
- `load_geojson` has no tool listing the shared results folder, so the file name has to come
  from the user or the simulation.

### Builder results are no longer cached, so a cache can't hand back the wrong geometry · 2026-09-30 · [#62](https://github.com/dtcc-platform/dtcc-agent/pull/62)

**Before:** four builders (terrain raster, terrain mesh, city mesh, classification filter) had
their results cached on disk under a key that described each input only by its type, size,
source and label, and ignored the area asked for. Two different point clouds of the same size,
or the same point cloud with two different areas, got the same key, so a builder could be
answered with another input's or another area's geometry, with no warning (U2, #11).

**Now:** only the two downloads, point clouds and buildings, are cached. Every builder call
runs for real, and each one is recorded in `builder_calls.jsonl` in the log folder with how
long it took and the key a correct cache would have matched. A few weeks of real use tells us
how often the same builder call repeats and how many seconds a cache would save. That decides
whether to build provenance keys (`TODOS.md` T-001). The unused builder cache code is removed.

**How we know it works:** new tests check that only the downloads are cached, that a builder
call writes a record and nothing to the cache, that the recorded key tells two areas apart and
matches the same inputs across sessions, and that nothing is written without a log folder.
344 tests pass.

### The Docker image installs the dtcc-core we pin and test, not Core's moving `develop` · 2026-09-30 · [#61](https://github.com/dtcc-platform/dtcc-agent/pull/61)

**Before:** the Dockerfile installed dtcc-core from a build argument that defaulted to Core's
`develop` branch, then installed the agent. Building the image showed the first install stuck:
the image reported `requested_revision: develop`, not our pinned commit. So the one artifact
that reaches production ran whatever Core `develop` was at build time, which the contract
workflow never tested. It matched our pin only because Core's `develop` happens to equal it
today (#41, U10).

**Now:** the separate Core install and its `DTCC_CORE_REF` build argument are gone, from the
Dockerfile, `docker-compose.yml` and `build_docker.sh`. Core comes from the commit pinned in
`pyproject.toml`, and the build fails if pip installed anything else.

**How we know it works:** images built before and after the change, with the installed Core
read from each. The build check fails when pointed at a different pin.

### dtcc-core moved to `bb95f2f`, with the three fixes we reported; the agent's stopgaps for two of them are gone · 2026-09-30 · [#59](https://github.com/dtcc-platform/dtcc-agent/pull/59)

**Before:** pinned to `9b4e9b9`. Three Core bugs we reported were open, and the agent carried
its own repairs for two of them (#55, #58).

**Now:** pinned to `bb95f2f`, the head of Core's `develop`, which fixes:
- [dtcc-core#126](https://github.com/dtcc-platform/dtcc-core/issues/126): two downloads of the
  same tile no longer share a temporary file.
- [dtcc-core#128](https://github.com/dtcc-platform/dtcc-core/issues/128): a dtcc-sim service is
  registered all or nothing, so a half-broken reply leaves nothing behind.
- [dtcc-core#132](https://github.com/dtcc-platform/dtcc-core/issues/132): a dtcc-sim service can
  no longer replace a built-in dataset of the same name.

The agent's repair step from #55 and #58 is removed, along with its lock, and the gap in it that
PR-Agent flagged on #58 goes with it. The pin also brings Core's rewritten footprint cleaning.
The disk cache starts empty once, because entries record the Core that wrote them (#56).

**How we know it works:** the contract workflow passed on `bb95f2f`. The #45 and #44 tests now
run Core's own `register_remote_service` with only the HTTP reply faked, so they fail if a
later pin brings either bug back. A live Lindholmen run gives the same buildings as before, and
the same from the cache as from a fresh download: 13 of 127 on LM, 12 of 138 on OSM. 343 tests
pass (one test covered a case Core's all-or-nothing registration makes impossible).

### A half-broken dtcc-sim service no longer leaves a new stray dataset behind on every retry · 2026-09-30 · [#58](https://github.com/dtcc-platform/dtcc-agent/pull/58)

**Before:** when a dtcc-sim service's list of datasets had one good entry and then a broken
one, Core registered the good one but reported that nothing was registered. The agent asked
again every 30 seconds and on every `list_simulations` call, and each attempt left one more
copy behind. `list_simulations` showed the good dataset while `list_operations` did not (#44).

**Now:** after a failed discovery the agent undoes what it registered, and puts back anything
it replaced, with a warning naming the datasets and the service. A half-broken service shows
nothing until it is fixed, and is asked again as before. Registrations run one at a time, so a
failed attempt on one thread cannot undo a successful one on another. This extends #55's
restore of Core datasets into one repair step. The real fix is in Core
([dtcc-core#128](https://github.com/dtcc-platform/dtcc-core/issues/128)).

**How we know it works:** eight retries of a half-broken service leave no copy behind, and a
dataset it replaced is put back. Both tests fail without the fix. 344 tests pass.

### References say what they name, a wrong one is refused, and a run hands back its result's reference · 2026-09-30 · [#57](https://github.com/dtcc-platform/dtcc-agent/pull/57)

**Before:** runs and stored objects both had ids of 8 hex characters, so nothing could tell
them apart. Passing a run's id to an object tool said "not found" and pointed at
`list_objects()`, where it would never appear. A run kept a second copy of its result that was
never evicted, and `get_run_summary` gave no way to reach the stored result. Two GeoJSON tools
crashed on an unknown id instead of reporting it (T9, #26).

**Now:**
- Object references are `obj_…` and run references are `run_…`. Tools take and return
  `object_ref` and `run_ref`; `run_operation` returns `object_ref` instead of `result_id`.
- A run reference passed where an object is expected (or the reverse, or as an operation
  parameter) is refused with a message naming the right tool.
- A run records its result's `object_ref` and keeps no copy: the stored object owns the result
  (U6, decided 2026-09-30), so the memory budget covers simulation results. When the object has
  been evicted, `get_run_summary` says so and asks for a re-run.
- `query_geojson` and `summarize_geojson_property` report an unknown reference as an error.

**How we know it works:** the four M0 tests that pinned the old behaviour were flipped to
ADR-0010's contract, and nine new tests cover the prefixes, the run→object link, refusals in
both directions and through `run_operation`, an evicted result, and the GeoJSON tools. 342 tests
pass.

### After a dtcc-core upgrade the disk cache starts cold instead of loading the old Core's objects · 2026-09-30 · [#56](https://github.com/dtcc-platform/dtcc-agent/pull/56)

**Before:** the disk cache keeps Core objects for 7 days and, since #51, survives restarts in
`~/.cache/dtcc_agent`. Nothing recorded which Core wrote an entry. After a Core upgrade the
agent would load objects saved by the old Core, which could crash the request or quietly give
a wrong answer. An index in an older or unexpected format could also crash a lookup (T12, #27).

**Now:**
- Every entry records the cache format and the dtcc-core commit that wrote it.
- An entry from another Core or format, or one with missing or broken fields, is never served,
  and startup removes it with its file. After an upgrade the cache simply starts cold.
- One check decides whether an entry may be served, used by both lookups and cleanup.

**How we know it works:** six new tests: entries carry the stamp; a restart on a newer Core
starts cold and deletes the old files; another Core's entries in a shared folder are never
served; an index written before stamps, one with broken entries, and one that is not a list
each start cold and keep working. 335 tests pass.

### A dtcc-sim service can no longer replace a Core dataset by reusing its name · 2026-09-29 · [#55](https://github.com/dtcc-platform/dtcc-agent/pull/55)

**Before:** Core's dataset registry replaces an entry of the same name. A dtcc-sim service
that advertised `point_cloud` or `buildings` took over Core's: `run_operation` and
`get_buildings` would then fetch from that service instead of Core, and the only trace was a
Core log line (#45).

**Now:** when the agent registers a dtcc-sim service, it puts back any Core dataset the
service replaced and logs a warning naming the dataset and the service. The service's other
datasets join as before.

**How we know it works:** a new test registers a service advertising `point_cloud` and a new
`flood_sim`, and checks that Core's registry and the catalogue keep Core's `point_cloud` while
`flood_sim` joins. It fails without the fix. 329 tests pass.

### Slope and aspect no longer check a cache they can never be stored in · 2026-09-29 · [#53](https://github.com/dtcc-platform/dtcc-agent/pull/53)

**Before:** `builder.raster.slope_aspect` was on the disk-cache allowlist, but it returns two
rasters and the cache only stores single results. Every call looked the cache up, missed,
recomputed, and stored nothing (#42).

**Now:** it is off the allowlist, so it skips the lookup. Nothing it returns changes. Caching
two-part results is left for U2 ([#11](https://github.com/dtcc-platform/dtcc-agent/issues/11)),
which decides whether builder results are cached at all.

**How we know it works:** a new test runs a two-raster operation under that name and checks
the cache is never touched. 328 tests pass.

### Building heights are real, bad bounds are refused, and the disk cache can't be tampered with · 2026-09-29 · [#51](https://github.com/dtcc-platform/dtcc-agent/pull/51)

**Before:** every `get_buildings` answer said the buildings were 0 m tall, because Core keeps
its height estimate where we did not look. `run_operation("datasets.buildings")` returned no
count or heights at all. The total footprint area covered only the buildings listed. An
upside-down or zero-size area was answered from any larger cached area. The disk cache lived
in `/tmp/dtcc_cache`, where another user on the machine could plant a file the agent would
load and run, and two agent processes sharing the cache lost each other's entries.

**Now:**
- Heights come from Core's estimate (or its measurement), in `get_buildings` and in
  `run_operation` alike. Live on Lindholmen: 3.0 to 28.5 m, mean 15.9 m, where it said 0.
- The total footprint area covers every building in the area.
- `run_operation`, `get_buildings`, `run_simulation` and `compare_scenarios` refuse bounds
  that describe no area, with a message saying why.
- The cache lives in `~/.cache/dtcc_agent` by default, is created private, and the agent
  refuses to start on a cache another user could change. Docker's `/data/cache` is unaffected.
- Several processes can share one cache without losing entries.

**How we know it works:** 327 tests pass, 31 of them new, including 8 processes writing one
cache at once (1600 of 1600 entries kept). A Codex adversarial review ran three rounds and a
Claude review one; every finding was fixed.

### Building counts for a smaller area inside a cached one are right · 2026-09-29 · [#50](https://github.com/dtcc-platform/dtcc-agent/pull/50)

**Before:** asking `get_buildings` about an area inside one already cached returned the
cached area's answer with only its bounds changed: the building count, the building list
and the height statistics all described the larger area (#39). `run_operation` on
`datasets.buildings` had the same bug for a different reason: its crop kept every building.

**Now:**
- `get_buildings` caches the building download, not its summary, crops it to the area asked
  for, and summarises it per request. A different `max_buildings` reuses the same download.
- The crop keeps exactly what a fresh download of that area would: Core's own rule, the whole
  footprint inside the bounds less 2 m, with multi-part buildings kept or dropped together.
- `get_buildings` and `run_operation("datasets.buildings")` share one cache entry, so
  whichever runs first saves the other a download.
- A cached area that cannot be cropped is downloaded again rather than reused whole.

**How we know it works:** a live run on Lindholmen (500 m cached, a 200 m area inside it) gave the same buildings from the cache as from a fresh download: 13 of 127 on LM, 12 of 138 on OSM. 295 tests pass, 15 of them new, built from real Core buildings:
a sub-area inside a cached one, buildings crossing the edge, multi-part buildings, and each
cache failure path. Reviewed by five specialist passes, a Claude adversarial pass and three
Codex rounds. Two rare edge cases remain where the cache can count one building more at the
edge than a fresh download (tiny or malformed source shapes), filed as
[#49](https://github.com/dtcc-platform/dtcc-agent/issues/49).

### A generated wiki of the codebase · 2026-09-27 · [#47](https://github.com/dtcc-platform/dtcc-agent/pull/47)

**Before:** understanding a part of the agent meant reading the code, the ADRs and the
README and piecing them together.

**Now:**
- `openwiki/` holds 12 pages: an overview, a quickstart that routes common tasks to the right
  page, and pages on the MCP server, Sessions, the operation catalogue, dispatch, the disk
  cache, simulations, the chatbot, deployment and CI, the tests, and a map of the ADRs.
- Every factual statement is backed by a link to the lines of code it describes, kept in
  `openwiki/.claims/`. When that code changes, the statement is flagged for rechecking.
- It is regenerated by hand with `/openwiki` every few pull requests. There is no scheduled
  job: that needed an OpenAI key and a third-party action, and would have added a third AI
  vendor.
- AGENTS.md tells agents the wiki exists and that code and tests win when they disagree.

**How we know it works:** the pages were regenerated after T10 (#43) merged, so they
describe the catalogue as it is on `develop`. Every factual statement passed OpenWiki's
check that its linked code exists.

### The catalogue is built once per process (M1a/T10) · 2026-09-27 · [#43](https://github.com/dtcc-platform/dtcc-agent/pull/43)

**Before:** the list of operations the agent can run (133 of them) was built the first time
anyone asked for it. That takes about 1.2 seconds, so the first user after every restart waited
for it. If part of it failed to load, the server logged a warning and served a smaller list,
which is how 135 operations quietly became 133 once. Since T4 two first requests could also
both build it at the same time.

**Now:**
- The HTTP server builds the list when it starts, before it answers anyone, and prints how
  many operations it has.
- If any part that comes from the pinned dtcc-core fails, the HTTP server stops at startup
  and says which part failed. A broken install can no longer look like a working one. That
  now includes a Core that lists an operation it no longer has, and a Core dataset whose
  options can't be read; both used to be skipped without a word.
- Over stdio the list is built the first time it's needed. The chatbot starts a new stdio
  server for every message, and most messages never need the list, so building it at startup
  would add over a second to each one. A broken part then fails that first call, naming it.
- Datasets from a dtcc-sim service are no longer loaded at startup. When the list is built, a
  background thread asks dtcc-sim, and asks again every 30 seconds while it's down (the agent
  warns once, though dtcc-core still logs its own warning on each attempt); the list picks its datasets up on the next read after
  it answers. Until then, asking for one of them says dtcc-sim hasn't answered yet, rather
  than that it doesn't exist. Before, if dtcc-sim was down when the server started,
  its datasets were missing until a restart, and startup waited up to 5 seconds per
  unreachable service. Now the two services can start in either order, and no request ever
  waits on dtcc-sim, even one that hangs.
- Datasets that don't come from the pinned Core (the optional `dtcc_sim` package, a dtcc-sim
  service) are optional. One whose options can't be read is left out with a warning, and
  never stops the server. A broken `dtcc_sim` package now warns and keeps Core's datasets;
  before, it dropped every dataset, or for a broken dependency said nothing at all.
- The worker limit from T8 now lives in the same startup module, `dtcc_agent/runtime.py`.

**How we know it works:**
- A real HTTP server, used by three sessions, builds the list once, before its port answers.
- With a broken part, the HTTP server prints the part and exits with code 3.
- With dtcc-sim pointed at an address that never answers, startup took 0.7 s (5.8 s before
  this change). With a fake dtcc-sim that never answers, three reads of the list return at
  once and dtcc-sim is asked only once, from the background thread.
- Each test fails when the part it covers is removed: the HTTP startup hook, stdio not
  building at startup, the lock that stops two first requests building it twice, the rule
  that a Core failure stops the build, the 30-second retry, only one background thread
  asking, and swapping in a new list rather than changing the one others are reading.

### A limit on concurrent Core work (M1a/T8) · 2026-09-26 · [#36](https://github.com/dtcc-platform/dtcc-agent/pull/36)

**Before:** since T4, up to 40 tools could run at once, the default size of the thread pool.
Each Core operation can copy a large input before working on it, so a handful of users
running builds together could run the server out of memory. That shows up as a crash, not as
"busy, please wait". Separately, two users asking for the same new tile at the same moment
would each download it, and Core's downloader can fail when two downloads of one tile
overlap ([dtcc-core#126](https://github.com/dtcc-platform/dtcc-core/issues/126)).

**Now:**
- At most 4 tool calls run at once across the whole server. The number is set with
  `DTCC_MCP_WORKERS`, so it can be sized to the machine's memory.
- One user may use at most half of them, so a busy user never leaves the others waiting.
  Over stdio, where there is only one user, the limit is the whole 4.
- Two requests for the same dataset and bounds never download at the same time, whichever tool
  they come through. The second waits, then usually reads the cache. While it waits it takes
  none of the server's 4 workers, only one of its own user's share, and it can be cancelled.
  Overlapping or nested areas are still downloaded separately.
- Builds are not shared this way. Their cache key can match two different inputs, so sharing
  would hand one user another user's result. Fixing that key is decision U2
  ([#11](https://github.com/dtcc-platform/dtcc-agent/issues/11)).
- Rendering still runs on the main thread, outside the limit, as the graphics library needs.

**How we know it works:**
- Six calls from six users at once, with a limit of 2: never more than 2 run together.
- Six calls from one user, with room for 10: never more than 2 run together.
- Two users asking for one tile at once, one spelling the coordinates as whole numbers and one
  as decimals: one download, and the second user gets a cache hit.
- A user whose share is busy never holds up another user asking for the same tile.
- While one download is in progress and a second request waits for it, an unrelated request
  from a third user still gets a worker and finishes. A cancelled wait leaves nothing behind.
- Each of these tests fails when the part it covers is removed from the code.

**Trade-offs to know:**
- A quick tool, such as listing your objects, waits behind long builds when every worker is
  busy.
- Fairness is per session id, and the server trusts the id the client sends until central
  authentication lands (U11, [#15](https://github.com/dtcc-platform/dtcc-agent/issues/15)).

### The automated reviewer gets repo context · 2026-09-26 · [#37](https://github.com/dtcc-platform/dtcc-agent/pull/37)

**Before:** PR-Agent reviewed with no knowledge of this repo, and it only suggested code changes
for outright bugs, so since #33 its suggestions had come back empty.

**Now:** a `.pr_agent.toml` tells it what the agent is, where the glossary and decisions live,
and what to check hardest: keeping each user's data separate, the concurrency rules from T8, and
failures that happen quietly. As a trial, it also suggests maintainability fixes, not only bugs.
Ending the trial is one line.

**How we know it works:** asked to review #36 again, it flagged a timing-based test and suggested
three code changes where it had suggested none. The test fix and one of the code changes were
applied; the other two did not hold up.

### dtcc-core pin moved to Core's latest `develop` · 2026-09-25 · [#35](https://github.com/dtcc-platform/dtcc-agent/pull/35)

**Before:** we were pinned to Core `18eb176` from 18 September. Two problems:
- It was older than the fixes the Core team made for the defects we reported.
- Core's `develop` history was rewritten after we pinned, so that commit is no longer on any
  Core branch. A commit on no branch can be deleted by GitHub, and then a fresh install of this
  repo would fail.

**Now:** pinned to `9b4e9b9`, the head of Core's `develop` on 24 September. Nothing else in the
dependency lock moved.

**How we know it works:**
- The contract workflow, which tests a candidate Core before the pin moves, passes: the
  installed Core is the right commit, 194 tests pass, and the catalogue stays at 133
  operations.
- With T5 on top, all 212 tests pass on the new Core.
- One fix checked before and after, through the agent itself: reprojecting a mesh that
  carries a data field fails on the old pin ("Fields and semantic regions require an explicit
  reprojection rule") and works on the new one, keeping the field.

**Also fixed:** the contract workflow could never pass for any Core. It skipped installing the
chatbot's dependencies, so the test run stopped before testing anything. It now installs the
same things as the main CI.

### Session isolation over HTTP (M1a/T5) · 2026-09-25 · [#33](https://github.com/dtcc-platform/dtcc-agent/pull/33)

**Before:** every person using the chatbot shared one object store and one list of runs. One
user could see, use or delete what another user had built. Conversation memory also searched
everyone's past chats.

**Now:**
- The MCP server can run over HTTP (`DTCC_MCP_TRANSPORT=http`). stdio stays the default, so
  switching back is a configuration change.
- The chatbot sends its session id with every tool call (the `X-DTCC-Session` header). Each
  session gets its own objects and runs. A call without a session id is refused.
- Conversation memory only searches the current session's past messages.
- At most 8 sessions are kept in memory at once, each with a 256 MiB share, so the server
  stays under the same 2 GiB it had before. A session in the middle of a tool call is never
  dropped.

**Why a header and not the connection:** the chatbot opens a new connection for every
message. Anything tied to the connection would be forgotten after each message.

**How we know it works:**
- Two real clients over HTTP cannot see each other's objects, and a session still finds its
  objects on the next message.
- Tested end to end with the real Claude Code client: the right session sees its object, the
  other sees nothing.
- The review caught a leak before merge: the default HTTP mode kept 2 server tasks alive for
  every chat message, forever. Measured again after the fix: 0.
- 18 new tests, including one proving the default stdio mode still works end to end.

**Still open, each with an owner:**
- Only reachable through `localhost` until the deployment task sets the allowed host names
  (T13, [#24](https://github.com/dtcc-platform/dtcc-agent/issues/24)).
- The session id is not a password. Anyone who can reach the port and knows an id can read
  that session. Who may connect is decision U11 ([#15](https://github.com/dtcc-platform/dtcc-agent/issues/15)).
- Exported files still share one folder (T7, [#21](https://github.com/dtcc-platform/dtcc-agent/issues/21)),
  and the builder cache is still shared between sessions (T6, [#20](https://github.com/dtcc-platform/dtcc-agent/issues/20)).
- The 8-session cap is a stopgap. The real memory budget, with session expiry, is T11
  ([#23](https://github.com/dtcc-platform/dtcc-agent/issues/23)).

### Tools run off the event loop (M1a/T4) · 2026-09-25 · [#32](https://github.com/dtcc-platform/dtcc-agent/pull/32)

**Before:** dtcc-core starts its own event loop inside every lidar and GeoPackage download.
The server ran tools on its main loop and relied on a patch (`nest_asyncio`) to allow that.
The patch does not work with the faster loop the web server uses, so the first real download
after moving to HTTP would have failed.

**Now:**
- Every tool body runs on a worker thread, where Core can start its own loop safely. The
  patch is gone.
- Rendering (`render_object`) stays on the main thread. The graphics library needs that, and
  on macOS anything else crashes the whole server.
- Deleting an object is now one safe step. Two tools can run at the same time now, and the
  old check-then-delete could fail halfway.

**How we know it works:** with both download caches emptied, `develop` fails with
`asyncio.run() cannot be called from a running event loop`. This branch downloads the tiles
and returns the point cloud, also when served by the web server over HTTP.

**Found in review and handed on:**
- Two users downloading the same new tile at once can make one of the downloads fail. The bug
  is in Core's downloader, filed as [dtcc-core#126](https://github.com/dtcc-platform/dtcc-core/issues/126).
- Up to 40 tools can now run at once. Putting a limit on that is T8
  ([#19](https://github.com/dtcc-platform/dtcc-agent/issues/19)), in the same milestone.

### Automated first-pass review · 2026-09-24 · [#31](https://github.com/dtcc-platform/dtcc-agent/pull/31)

[PR-Agent](https://docs.pr-agent.ai) now reviews every pull request when it opens, using
Gemini. It posts a short review and code suggestions, and answers `/review`, `/improve` and
`/ask` comments.

- **An extra read, not a gate.** It never approves or blocks, and our own reviews are
  unchanged. It reads only the diff in one pass, so its suggestions are hints to check.
- **Only people with write access can trigger it by comment.** The repo is public, and
  otherwise anyone could spend the API key.
- **Installed from PyPI rather than as a GitHub Action,** because the organisation only
  allows actions it owns, GitHub's own, or Marketplace-verified ones.

Its first real suggestion, on #32, was valid. It was applied in a stricter form: only the one test that needs uvloop skips, rather than falling back to a loop that would let it pass untested.

### M1 reviewed and split; the programme tracked on GitHub · 2026-09-24 · [#9](https://github.com/dtcc-platform/dtcc-agent/pull/9)

A second engineering review of milestone M1, with Codex as an independent second reviewer.
It produced 31 findings.

- **M1 split into M1a** (transport, session state, execution) **and M1b** (typed
  references, cache versioning). Same ten tasks, nothing cut. They must not be built in
  parallel, because both rewrite `server.py`.
- **The concurrency limit (T8) moved into M1a,** next to the change that makes it necessary.
- **The work is now on GitHub:**
  - 6 milestones and 21 issues, with sub-issues and pull request links.
  - 6 decision issues that block M1a.
  - No task issues for M2 to M4 yet. They are not specified, and guessed issues would read
    as agreed scope.
- **Codex overturned one of our own conclusions.** FastMCP's startup hook runs once per
  session, not once per process. That changed the design of three tasks. Verified in the
  library source before accepting.

### M0: dtcc-core pinned, CI running, today's behaviour recorded · 2026-09-21 · [#7](https://github.com/dtcc-platform/dtcc-agent/pull/7), [#8](https://github.com/dtcc-platform/dtcc-agent/pull/8)

**The problem:** dtcc-core was not listed as a dependency. When it was missing, eight places
in the code quietly skipped their work. A fresh install started without errors and offered an
empty catalogue. It looked like a working server with nothing in it.

**What changed:**
- dtcc-core is declared and pinned to one exact commit (`18eb176`), so everyone runs the
  same Core.
- A missing Core now fails loudly at startup, and the error says how to fix it.
- A contract workflow tests a new Core version before the pin is moved.
- CI runs on every push and pull request.
- 61 characterisation tests record what the 22 tools do today, including what they do wrong.
  When M1 changes that behaviour on purpose, these tests fail on purpose and get updated. That
  way no change in behaviour goes unnoticed.

**Result:** 189 tests passing, catalogue of 133 operations, CI green.

### Working conventions · 2026-09-21 · [#6](https://github.com/dtcc-platform/dtcc-agent/pull/6)

Written conventions for GitHub issues, triage labels and the project glossary, read by both
Claude and Codex. `AGENTS.md` holds them; `CLAUDE.md` points to it.

### Glossary, decisions and the rebuild plan · 2026-09-21 · [#3](https://github.com/dtcc-platform/dtcc-agent/pull/3)

- **`CONTEXT.md`:** a glossary. Where Twin already defines a term, its definition wins.
- **Architecture decisions (ADRs):**

  | ADR | Decision |
  |---|---|
  | 0001 | The agent is the conversational front door |
  | 0002 | Converge on Twin's contracts |
  | 0003 | Replace the Claude Agent SDK with pydantic-ai |
  | 0004 | The Session is the isolation unit |
  | 0005 | Retrieval is a separate MCP server |
  | 0006 | Prompt caching shaped for Anthropic |
  | 0007 | The agent keeps its own generic dispatch |
  | 0008 | Evaluation is one harness with two layers |
  | 0009 | Rebuild on a branch, not a fresh repository |
  | 0010 | References are typed, and a run records its object |

- **The rebuild plan:** milestones M0 to M4 in `docs/plans/2026-09-19-rebuild-plan.md`.

### The server starts on a fresh install again · 2026-09-21 · [#2](https://github.com/dtcc-platform/dtcc-agent/pull/2)

A clean install picked up version 2 of the `mcp` library, which removed the module the server
imports, so `python -m dtcc_agent` crashed. The tests still passed, because none of them
loaded the server. Fixed by pinning `mcp` below version 2, and by adding a test that loads the
server, so this can't go unnoticed again.

### Upstream fixes in dtcc-core and dtcc-sim · reported 2026-09-18, fixed 2026-09-22/23, in our build 2026-09-25

Found while checking the agent against the current Core. All four were fixed by the Core team.

| Issue | Problem |
|---|---|
| [dtcc-core#110](https://github.com/dtcc-platform/dtcc-core/issues/110) | Building a terrain raster from ground points rejected every downloaded point cloud |
| [dtcc-core#111](https://github.com/dtcc-platform/dtcc-core/issues/111) | Buildings with no lidar points on the roof were silently deleted |
| [dtcc-core#112](https://github.com/dtcc-platform/dtcc-core/issues/112) | No rule for reprojecting geometry that carries data fields |
| [dtcc-sim#8](https://github.com/dtcc-platform/dtcc-sim/issues/8) | Two simulations could not write the native `dtcc` output format |

**In our build since #35.** The Core fixes arrived with the pin move to `9b4e9b9`. The Sim fix
lives in dtcc-sim and doesn't depend on our pin.

### Assessment of what works · 2026-09-14 · [#1](https://github.com/dtcc-platform/dtcc-agent/issues/1)

A checked write-up of the starting point. The component tests were green, but the server
could not start from a fresh install. The docs and dependencies had drifted, results from
the mini-service were incomplete, and three README examples could not be reproduced. Posted on
#1 and used as the input to the plan above.

---

## ⏳ Open decisions

These block tasks in M1a. Each issue carries the evidence needed to decide.

| Decision | Blocks | Status |
|---|---|---|
| U3: where once-per-process startup lives ([#12](https://github.com/dtcc-platform/dtcc-agent/issues/12)) | T8, T10, T11 | ✅ Decided 2026-09-24: at process startup, not in FastMCP's per-session hook |
| U6: who owns a simulation result, the run or the stored object (rebuild plan) | T9 | ✅ Decided 2026-09-30: the stored object owns it; the run keeps its reference (ADR-0010, #57) |
| U1: how far the filesystem boundary goes ([#10](https://github.com/dtcc-platform/dtcc-agent/issues/10)) | T7 | ✅ Decided 2026-09-30: refuse path arguments in `run_operation`; `export_object` and `load_geojson` get a proper route in T7 |
| U2: fix the cache keys, or turn builder caching off ([#11](https://github.com/dtcc-platform/dtcc-agent/issues/11)) | T6 | ✅ Decided 2026-09-30: off for now, and builder calls recorded to measure whether provenance keys (T-001) are worth it |
| U4: how accurate the memory budget must be ([#13](https://github.com/dtcc-platform/dtcc-agent/issues/13)) | T11 | ✅ Decided 2026-09-30: accurate byte counts for stored types; an oversized result returns its summary unstored; the budget covers the stored results |
| U10: which dtcc-core install wins in the container ([#14](https://github.com/dtcc-platform/dtcc-agent/issues/14)) | T13 | ✅ Decided 2026-09-30: the pin wins; the build arg won before, fixed by #61 |
| U11: which network interface the MCP server listens on, and who may connect ([#15](https://github.com/dtcc-platform/dtcc-agent/issues/15)) | T13 | ✅ Decided 2026-09-30: loopback only until auth (T14, M2) |
| U9: fix rendering in M1a, or switch it off until images reach the page (rebuild plan) | T7 | ✅ Decided 2026-09-30: fix it, drawn with matplotlib rather than dtcc-viewer (#63) |

## What's next

1. **The rest of M1a:** per-session file folders with path arguments refused (T7, U1) are
   merged (#63), and the memory budget (T11, U4) is in review (#66). The two-service container
   on loopback (T13, U10 and U11) remains; it must give both services the same artifacts folder. Exporting a
   building collection is queued as #65.
2. **M1b is done:** typed references with a run linked to its object (T9, #57) and cache
   versioning (T12, #56).

## For discussion with the team

- **PR-Agent.** It runs on every PR on one Gemini API key. Who owns that key, and is an
  extra automated read worth it for the team?
