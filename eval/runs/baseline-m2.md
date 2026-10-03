# M2 baseline — measurement run 2026-10-03 13:40 UTC

- Harness commit: `0069448` (the agent's code is `develop` at `fd4e74d`) · runs per question: 3 · questions: 10
- Model: claude-sonnet-4-5-20250929 (all models used: claude-haiku-4-5-20251001, claude-sonnet-4-5)
- Prompt version: `5fdfaaf2e87b` · SDK: claude-agent-sdk 0.2.163
- Core commit: `bb95f2f8c338e6433a1e6a48375cf634710ff421` · catalogue: 133 operations
- dtcc-sim: available (probe names simulations)
- Spent: $2.36 of a $10.00 cap

Each cell is median / max over the question's successful runs; failed runs are left out
and counted under OK and Problems. Run 1 is cold, later runs warm. Input tokens include
cache reads and writes. Cost is the SDK's `total_cost_usd`.

| Question | OK | Cold latency (s) | Warm latency (s) | Input tok | Output tok | Cost ($) | Tools | Ops | Problems |
|---|---|---|---|---|---|---|---|---|---|
| q01-building-count | 3 / 3 | 40.1 / 40.1 | 23.1 / 24.3 | 69,439 / 69,689 | 885 / 952 | 0.049 / 0.071 | 3 / 3 | 2 / 2 |  |
| q02-cached-subarea | 3 / 3 | 25.6 / 25.6 | 26.1 / 26.1 | 69,392 / 69,476 | 904 / 959 | 0.049 / 0.070 | 3 / 3 | 2 / 2 |  |
| q03-terrain-slope | 3 / 3 | 47.1 / 47.1 | 59.1 / 60.2 | 150,447 / 175,429 | 1,922 / 2,540 | 0.125 / 0.126 | 7 / 8 | 6 / 7 |  |
| q04-render-buildings | 3 / 3 | 32.9 / 32.9 | 44.6 / 56.0 | 107,303 / 185,234 | 1,246 / 2,036 | 0.091 / 0.110 | 5 / 9 | 3 / 4 |  |
| q05-export-geojson | 3 / 3 | 28.1 / 28.1 | 34.8 / 37.0 | 88,638 / 89,300 | 1,154 / 1,232 | 0.062 / 0.081 | 4 / 4 | 3 / 3 |  |
| q06-discover-operations | 3 / 3 | 23.6 / 23.6 | 27.8 / 29.3 | 68,565 / 111,334 | 1,097 / 1,283 | 0.073 / 0.088 | 3 / 5 | 2 / 4 |  |
| q07-trees | 3 / 3 | 39.5 / 39.5 | 40.7 / 42.6 | 109,169 / 129,511 | 1,642 / 1,874 | 0.092 / 0.096 | 7 / 9 | 5 / 6 |  |
| q08-refused-path | 3 / 3 | 13.8 / 13.8 | 11.9 / 12.2 | 16,074 / 16,074 | 352 / 407 | 0.012 / 0.034 | 0 / 0 | 0 / 0 |  |
| q09-too-large | 2 / 3 | 80.4 / 80.4 | 77.7 / 77.7 | 126,466 / 126,573 | 2,270 / 2,367 | 0.107 / 0.120 | 6 / 6 | 5 / 5 | 1 error |
| q10-heatwave | 2 / 3 | 106.4 / 106.4 | 109.0 / 109.0 | 278,996 / 316,570 | 3,088 / 3,294 | 0.201 / 0.210 | 14 / 17 | 12 / 13 | 1 error |

**Totals:** 28 of 30 runs ok · $2.31 · 1,160 s of answering

## Notes

This is the baseline M3 compares against (ADR-0003, ADR-0008). Measurement only: nothing
here says whether an answer was right.

**Setup.**
- **Machine:** Docker Compose on an Apple Silicon Mac (Docker VM: 8 GB, arm64), with the
  image built for `linux/amd64` and run under emulation. Latencies on a native amd64 host
  will be lower.
- **Access:** both T14 secrets set.
- **Cache:** a fresh data directory, so run 1 started with an empty download cache.
- **dtcc-sim:** the `dtcc-sim:local` image built 2026-09-08. The probe found its
  simulations.
- **Model and cost:** the CLI also uses Claude Haiku for internal steps, hence two models
  above. Cost is the SDK's list-price figure, not an invoice.

**The two failed runs (run 3) are one incident:**
- **q09 crashed the tool server.** It asked for a point cloud over about 5 km of central
  Gothenburg. The model downloaded 123 LAZ tiles, and the MCP server was killed out of
  memory (`OOMKilled`, exit 137) partway through. Runs 1 and 2 of the same question
  finished.
- **The chat went down with it.** The chatbot shares that container's network (T13), and
  port 8050 is published there, so the chat became unreachable. q10's run 3 could not
  connect: that failure is collateral, not q10's own. Nothing restarted the server, and the
  chatbot kept reporting healthy.
- **q09's turn cost is missing from the totals.** Its provenance record was still written
  (`is_error`, `WebSocketDisconnect`, cost unknown), but no cost ever came back for it.

The T11 memory budget covers stored results, not memory while an operation runs (U4), so it
does not prevent this. q09 is kept in the set on purpose: M3 should report how it handles
the same question.
