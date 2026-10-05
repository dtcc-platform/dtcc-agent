# M3 runtime gate: pydantic-ai on Bedrock, two runs pooled (2026-10-05 13:29 and 13:38 UTC)

- Pooled from 2 runs: `20261005T132917Z-57b6507`, `20261005T133828Z-57b6507`
- Commit: `57b6507` · runs per question: 6 · questions: 10
- Model: eu.anthropic.claude-sonnet-5-5 (all models used: eu.anthropic.claude-sonnet-5-5)
- Prompt version: `5fdfaaf2e87b` · SDK: pydantic-ai-slim 2.54.0
- Runtime: pydantic-ai · provider: bedrock · cost source: genai-prices 0.1.9 (regional.anthropic.claude-sonnet-5-v1:0)
- Core commit: `bb95f2f8c338e6433a1e6a48375cf634710ff421` · catalogue: 133 operations
- dtcc-sim: available
- Spent: $1.47 across the runs

Each cell is median / max over the question's successful runs; failed runs are left out
and counted under OK and Problems. Run 1 is cold, later runs warm. Input tokens include
cache reads and writes. Cost is computed as the cost source above says.

| Question | OK | Cold latency (s) | Warm latency (s) | Input tok | Output tok | Cost ($) | Tools | Ops | Problems |
|---|---|---|---|---|---|---|---|---|---|
| q01-building-count | 6 / 6 | 15.4 / 15.5 | 4.8 / 5.7 | 25,066 / 25,305 | 337 / 362 | 0.011 / 0.013 | 2 / 2 | 2 / 2 |  |
| q02-cached-subarea | 6 / 6 | 5.9 / 6.1 | 5.8 / 6.0 | 25,524 / 25,530 | 485 / 506 | 0.015 / 0.016 | 2 / 2 | 2 / 2 |  |
| q03-terrain-slope | 6 / 6 | 17.3 / 17.9 | 14.1 / 17.5 | 53,898 / 53,931 | 1,244 / 1,483 | 0.032 / 0.035 | 7 / 7 | 7 / 7 |  |
| q04-render-buildings | 6 / 6 | 10.2 / 11.7 | 9.5 / 9.7 | 52,781 / 52,970 | 830 / 941 | 0.048 / 0.049 | 4 / 5 | 4 / 5 |  |
| q05-export-geojson | 6 / 6 | 7.2 / 7.8 | 6.9 / 7.8 | 33,018 / 33,068 | 528 / 539 | 0.014 / 0.015 | 3 / 3 | 3 / 3 |  |
| q06-discover-operations | 6 / 6 | 7.4 / 8.3 | 8.2 / 9.2 | 18,011 / 18,011 | 1,066 / 1,164 | 0.020 / 0.023 | 3 / 3 | 3 / 3 |  |
| q07-trees | 6 / 6 | 14.2 / 14.6 | 10.6 / 11.7 | 39,678 / 39,704 | 1,046 / 1,109 | 0.030 / 0.031 | 4 / 4 | 4 / 4 |  |
| q08-refused-path | 6 / 6 | 5.0 / 5.2 | 4.7 / 5.0 | 7,887 / 7,887 | 431 / 467 | 0.007 / 0.007 | 0 / 0 | 0 / 0 |  |
| q09-too-large | 6 / 6 | 28.1 / 47.0 | 29.3 / 49.7 | 36,043 / 45,315 | 1,848 / 2,004 | 0.037 / 0.039 | 6 / 6 | 6 / 6 |  |
| q10-heatwave | 6 / 6 | 61.9 / 63.4 | 62.7 / 66.2 | 60,494 / 68,975 | 1,502 / 1,614 | 0.037 / 0.044 | 6 / 7 | 6 / 7 |  |

**Totals:** 60 of 60 runs ok · $1.47 · 981 s of answering
## The gate: passed

Against `baseline-m3-sdk.md` (the Agent SDK on Bedrock, same model, same 10 questions, two
runs pooled), as the M3 epic (#29) defines it:

| Question | SDK warm (s) | pydantic-ai warm (s) | Change |
|---|---|---|---|
| q01-building-count | 11.1 | 4.8 | −56% |
| q02-cached-subarea | 11.0 | 5.8 | −47% |
| q03-terrain-slope | 17.1 | 14.1 | −18% |
| q04-render-buildings | 12.8 | 9.5 | −26% |
| q05-export-geojson | 11.5 | 6.9 | −40% |
| q06-discover-operations | 11.1 | 8.2 | −26% |
| q07-trees | 15.8 | 10.6 | −33% |
| q08-refused-path | 7.0 | 4.7 | −32% |
| q09-too-large | 39.0 | 29.3 | −25% |
| q10-heatwave | 60.9 | 62.7 | +3% |

1. **Median of the per-question warm medians: 12.1 s → 8.9 s (−27%).** Must be no higher: passed.
2. **No question more than 25% slower.** The worst is q10 at +3%, and most of its time is
   dtcc-sim running the simulation: passed.

Per run: $0.95 → $0.73 and 619 s → 491 s of answering.

## Notes

**What changed and what didn't.** Runtime only: same model (`eu.anthropic.claude-sonnet-5-5`),
same provider and Region, same prompt (`5fdfaaf2e87b`), same tools, same machine and setup
(including the same other containers in the Docker VM). The image was built with
`EXTRAS=chatbot`: no Agent SDK and no claude CLI inside.

**Where the time went.**
- **No process per message.** The SDK started the claude CLI and did an MCP handshake for every
  message (ADR-0003); that cost is gone. It shows most on short questions: q01's warm median
  went from 11.1 s to 4.8 s.
- **One fewer model request per turn** (median 4, was 5). The SDK loaded the dtcc-agent tools
  through its `ToolSearch` tool before using them; pydantic-ai gives the model all 22 at once.
  Input tokens per turn therefore rise on some questions (the tool definitions are always sent,
  and cached), while output falls 14%.

**Thinking.** Neither runtime sets a thinking option; both leave Sonnet 5.5 at its default,
which thinks. Provenance does not record thinking, so the two budgets cannot be compared
directly. Output tokens (thinking is billed as output) are 14% lower on pydantic-ai, which
the dropped `ToolSearch` request accounts for in part.

**Cost.** pydantic-ai turns are priced by `chatbot/prices.py` with genai-prices 0.1.9, which
resolves Sonnet 5.5 to Bedrock's regional Sonnet 5 entry ($2.20 / $11 per million tokens in and
out). Checked against the SDK's own `total_cost_usd` over the 60 reference turns: within 2%.
The two cost columns are therefore comparable, but they are not computed the same way.

**Clean runs.** Neither container restarted, all 62 turns (including two probes) wrote
provenance, none errored or retried, and every answer used only Sonnet 5.5. Regenerate with
`python -m eval.measure --pool eval/runs/m3-runtime/*.jsonl`.
