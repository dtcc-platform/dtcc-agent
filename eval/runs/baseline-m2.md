# M2 baseline — measurement run 2026-10-03 22:51 UTC

- Commit: `1ff99d9` (`develop` with #75–#83) · runs per question: 3 · questions: 10
- Model: claude-sonnet-4-5-20250929 (all models used: claude-haiku-4-5-20251001, claude-sonnet-4-5)
- Prompt version: `5fdfaaf2e87b` · SDK: claude-agent-sdk 0.2.163
- Core commit: `bb95f2f8c338e6433a1e6a48375cf634710ff421` · catalogue: 133 operations
- dtcc-sim: available (probe names simulations)
- Spent: $1.95 of a $10.00 cap

Each cell is median / max over the question's successful runs; failed runs are left out
and counted under OK and Problems. Run 1 is cold, later runs warm. Input tokens include
cache reads and writes. Cost is the SDK's `total_cost_usd`.

| Question | OK | Cold latency (s) | Warm latency (s) | Input tok | Output tok | Cost ($) | Tools | Ops | Problems |
|---|---|---|---|---|---|---|---|---|---|
| q01-building-count | 3 / 3 | 36.3 / 36.3 | 23.5 / 24.8 | 26,217 / 26,275 | 791 / 844 | 0.089 / 0.090 | 3 / 3 | 2 / 2 |  |
| q02-cached-subarea | 3 / 3 | 18.7 / 18.7 | 22.1 / 22.1 | 25,708 / 25,806 | 820 / 830 | 0.089 / 0.097 | 3 / 3 | 2 / 2 |  |
| q03-terrain-slope | 3 / 3 | 41.0 / 41.0 | 43.1 / 46.1 | 44,748 / 45,232 | 1,865 / 2,037 | 0.074 / 0.077 | 7 / 8 | 6 / 7 |  |
| q04-render-buildings | 3 / 3 | 25.4 / 25.4 | 30.0 / 35.7 | 22,273 / 28,613 | 960 / 1,207 | 0.036 / 0.046 | 4 / 5 | 3 / 3 |  |
| q05-export-geojson | 3 / 3 | 24.0 / 24.0 | 28.2 / 30.6 | 22,533 / 22,562 | 979 / 1,009 | 0.038 / 0.046 | 4 / 4 | 3 / 3 |  |
| q06-discover-operations | 3 / 3 | 30.8 / 30.8 | 27.9 / 28.8 | 31,219 / 32,004 | 1,058 / 1,233 | 0.070 / 0.077 | 5 / 6 | 4 / 5 |  |
| q07-trees | 3 / 3 | 35.9 / 35.9 | 30.2 / 31.2 | 23,378 / 24,231 | 1,115 / 1,337 | 0.044 / 0.051 | 6 / 6 | 5 / 5 |  |
| q08-refused-path | 3 / 3 | 13.6 / 13.6 | 14.9 / 17.2 | 2,911 / 2,911 | 372 / 383 | 0.008 / 0.017 | 0 / 0 | 0 / 0 |  |
| q09-too-large | 3 / 3 | 40.5 / 40.5 | 57.3 / 59.9 | 17,535 / 22,036 | 1,884 / 2,375 | 0.060 / 0.061 | 3 / 4 | 2 / 3 |  |
| q10-heatwave | 3 / 3 | 110.0 / 110.0 | 113.1 / 122.3 | 139,766 / 140,982 | 3,165 / 3,246 | 0.154 / 0.154 | 16 / 17 | 11 / 13 |  |

**Totals:** 30 of 30 runs ok · $1.93 · 1,157 s of answering

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
- **Models:** the CLI also uses Claude Haiku for internal steps, hence two models above.
- **Clean run:** the MCP server never restarted and was never killed, all 31 turns
  (including the probe) wrote their provenance records, and the agent used only
  `ToolSearch` and dtcc-agent tools.

**Why it was re-run.** An earlier run the same day, on `fd4e74d` (#78), found three problems,
all now fixed:
- **#79:** q09 ran the MCP server out of memory and took the chat down with it. That run
  scored 28 of 30. Since #82 a LiDAR download over 10 km² is refused, and the agent asks for
  a smaller area. q09 now answers in about 40–60 s.
- **#80:** geocoding sent "central Gothenburg" to New Zealand. Since #81 lookups are
  limited to Sweden.
- **#83:** the agent was given the CLI's general-purpose built-in tools.

**What changed between the two runs:**
- **Total:** $2.31 → $1.93.
- **Prompts:** median input tokens fell about 3×, because the built-in tool definitions are
  gone. For example, q04 went from 107k to 22k.
- **Per-question cost moves with prompt caching, not just prompt size.** Cache writes cost
  more than plain input, and cache reads far less. So a question whose turn happened to
  write more of its prompt to the cache can cost more with fewer tokens. q01 cost $0.049
  before ($67k read, 2k written) and $0.089 now ($14k read, 12k written).
- **Comparing with M3:** use medians over several runs, and token counts as well as cost.

Cost is the SDK's `total_cost_usd` at list price, not an invoice.
