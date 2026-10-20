# T41 (#110): point-cloud downloads, in-process vs DTCC Engine

**Result: the Engine path is 30–38% slower in median answer latency, above the provisional
15% limit. By #110's rule, point-cloud downloads stay in-process.** Buildings could not be
measured: the Engine can't return them yet (dtcc-platform/dtcc-core#160).

## Setup

- **Date:** 2026-10-19, 23:24–23:48 UTC, all runs in one session.
- **Agent:** commit `0457229` (branch `feat/engine-core-switch`), image `dtcc-agent:t41-0457229`, Core `a78df16`.
- **Engine:** local, dtcc-twin `develop` `d508e10`, dtcc-engine 0.1.0, Core 0.9.8dev.
- **Model:** `eu.anthropic.claude-sonnet-5-5` on Bedrock, the same for both arms.
- **Arms:** the same image, one stack at a time, each with its own empty `/data` at the start.
  - **A, in-process:** `DTCC_ENGINE_CORE` unset. Downloads run in the tool server with the disk cache.
  - **B, Engine:** `DTCC_ENGINE_CORE=1`. `datasets.point_cloud` runs as an Engine job, without the disk cache.
- **Order (ADR-0008):** passes A, B, B, A with 3 runs each, then a second pair B, A with the Engine worker at concurrency 4 instead of dtcc-twin's default of 1.
- **Questions:** those in `eval/questions.json` that download point clouds but not buildings: `q03-terrain-slope` and `q09-too-large`. The building questions (q01, q02, q04, q05, q07) can't run on arm B; see "Buildings" below.

## Answer latency

Median over each arm's successful runs, cold and warm together. Max in brackets.

| | A, in-process | B, Engine | Change |
|---|---|---|---|
| **Engine at concurrency 1** (passes A, B, B, A; 6 runs per question per arm) | | | |
| `q03-terrain-slope` | 14,9 s (20,6) | 17,1 s (24,7) | +15% |
| `q09-too-large` | 16,9 s (32,7), 5/6 ok | 77,1 s (124,7) | +356% |
| **Both questions** | **15,6 s** | **21,6 s** | **+38%** |
| **Engine at concurrency 4** (passes B, A; 3 runs per question per arm) | | | |
| `q03-terrain-slope` | 12,2 s (15,5) | 17,0 s (18,8) | +40% |
| `q09-too-large` | 17,1 s (26,5) | 43,8 s (85,6) | +156% |
| **Both questions** | **13,8 s** | **17,9 s** | **+30%** |

Arm A's one failed run was a Bedrock `ModelHTTPError` on its first `q09`, not a download failure.

Harness reports: `pooled-a.md` and `pooled-b.md` (concurrency 1); `engine-concurrency-4/{a,b}/` (concurrency 4); per-pass files in `a/` and `b/`.

## Per-call download time

From each arm's provenance log (`operations.jsonl`), successful `datasets.point_cloud` calls only.

| | A: fresh download | A: from disk cache | B: Engine job |
|---|---|---|---|
| Concurrency 1 | median 8,1 s, max 17,6 s (n=6) | median 1,0 s (n=13) | median 34,0 s, max 109,4 s (n=21) |
| Concurrency 4 | 7,4 s (n=1) | median 1,1 s (n=7) | median 34,8 s, max 70,2 s (n=8) |

Two effects add up:

1. **The disk cache.** `q09` asks for too large an area. The 10 km² cap refuses it, and the model then fetches several smaller pieces, up to 5 in one answer. In-process, most pieces are crops of an earlier download, served from the disk cache in about 1 s. On the Engine, each piece is a fresh job.
2. **Each Engine job costs more than an in-process download.** It includes submitting, polling (1 s, doubling to 5 s), writing the canonical package on the worker, and downloading and reading it in the agent. With one Engine worker, an answer's parallel pieces also queue behind each other. Four workers cut `q09`'s worst case from 125 s to 86 s, but a fresh job still took a median of 34,8 s.

A single download on a warm Engine is fast: `q03`'s one call took 3,2 s. The gap grows with the number of downloads per answer, which is what the cache saves.

## Correctness

The same bounds (`[319950, 6398950, 320150, 6399150]`, EPSG:3006) fetched both ways return the same point cloud: 66,483 points, with identical coordinates after sorting.

## Cost

The same in both arms: $0,33 per arm at concurrency 1, $0,14 per arm at concurrency 4. The model's work doesn't change, only where the download runs.

## Buildings

Not measured. A `buildings` job computes the buildings (13 LOD1 for the bounds above, matching #107) and then fails while packaging them: `NotImplementedError: Canonical object BuildingCollection is unsupported`. Filed as dtcc-platform/dtcc-core#160. Building questions lean on the disk cache even more (`q02-cached-subarea` exists to test it), so the same two effects would apply.

## What this does not cover

- A deployed Engine (more CPU, an Engine near its data) wasn't measured. These are a laptop's numbers, with both arms emulating amd64 under Docker Desktop.
- An Engine with its own result cache, or an agent caching Engine results, would change effect 1. Neither exists today.
