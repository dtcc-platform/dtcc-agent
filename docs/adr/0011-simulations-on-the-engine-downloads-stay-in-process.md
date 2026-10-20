---
status: accepted
decided: 2026-10-20 by Spiros, from T41's measurement (#110)
supersedes: ADR-0007
---

# Simulations run on the DTCC Engine; interactive Core downloads stay in-process

**Decided:** the agent is a DTCC Engine client for **simulations**. Building and point-cloud
downloads, and everything built from them (meshing, cropping, rendering), **stay in-process**
behind the agent's disk cache. The agent keeps its own discovery and dispatch over Core
(`registry.py`, `dispatcher.py`, `serializers.py`), as ADR-0007 decided.

## What changed since ADR-0007

ADR-0007 decided the agent does not become an Engine client, when the Engine had no code. Since
then:

- **The Engine exists:** dtcc-twin `apps/engine`, dtcc-engine 0.1.0. It runs any registered Core or
  Sim dataset as a job: submit, poll, download a canonical package.
- **The team agreed on 2026-10-09 that all queuing goes through the Engine.** Interactive Core work
  would move too if a same-day measurement allowed it.
- **Simulations moved (#109, #113).** With `DTCC_ENGINE_URL` set they run as Engine jobs, and the
  folder shared with dtcc-sim goes away. Exactly one path runs, with no silent fallback.

So the half of ADR-0007 that said "no Engine client" no longer holds. The other half (keep the
agent's own dispatch, and the reasons for it) does, and this ADR keeps it.

## The measurement (T41, #110)

Point-cloud downloads, in-process against Engine jobs: one session on 2026-10-19, arms alternating
(ADR-0008), against a local Engine at dtcc-twin `d508e10`. Full report:
`eval/runs/m4-engine-vs-inprocess/README.md`.

| Engine workers | In-process, median answer | Engine, median answer | Change |
|---|---|---|---|
| 1 (dtcc-twin default) | 15,6 s | 21,6 s | +38% |
| 4 | 13,8 s | 17,9 s | +30% |

The rule set in #110 was "move if at most 15% slower". Both results are above it.

- **Why:** ADR-0007's cache argument, now measured. When a request is too large, the model fetches it in pieces. In-process, most pieces are crops of an earlier download, served from the disk cache in about 1 s (13 of 19 calls). Every Engine job is fresh, at a median of 34 s against 7 to 8 s for a fresh in-process download.
- **A single download on a warm Engine is fast** (3,2 s). The gap grows with the number of downloads per answer, which is what the cache saves.
- **Correctness and cost:** both paths return identical point clouds, and cost the same.

**Buildings were not measured:** the Engine can't package a `BuildingCollection`
(dtcc-platform/dtcc-core#160). Building questions lean on the cache even more, so this decision
covers them too until they can be measured.

## Consequences

- **Two places run Core.** The agent's pin (`a78df16`) and the Engine's Core (0.9.8dev) can differ. A simulation's input geometry and a download the agent shows beside it may come from different Core versions. The agent's pin moves through the contract workflow as before; the Engine's Core is the Engine's.
- **The deployed agent needs its own outbound access** to Lantmäteriet and OSM, and CPU and memory for Core work. `docs/deployment.md` already lists both.
- **The experimental switch stays.** `DTCC_ENGINE_CORE=1` runs buildings and point clouds as Engine jobs. It is off by default and must stay unset in a deployment. It is kept so buildings can be measured once dtcc-core#160 is fixed, and removed after that measurement.
- **ADR-0007's churn cost stays ours.** Tracking Core's catalogue is still the agent's job.

## Revisit when

- dtcc-core#160 is fixed: measure buildings with the switch, then remove it.
- The Engine gains a result cache that can answer a smaller area from a larger result, or a deployed Engine measures within 15%.
- The 15% itself is revisited: #110 set it provisionally.

## What was rejected

- **Moving the downloads to the Engine now, as agreed on 2026-10-09.** It makes a typical answer 30 to 38% slower, for no user-visible gain in a one-machine pilot.
- **Engine downloads behind the agent's disk cache.** This keeps the crop-from-cache hits and pays the Engine only on a miss. Not chosen now, because a miss is about 4 times slower than in-process and there is nothing the Engine path adds for the user yet. It's the first option to measure if the Engine needs to own all Core work later.
