# #88 model switch: Sonnet 5, one run (2026-10-06)

`DTCC_AGENT_MODEL=eu.anthropic.claude-sonnet-5` on image `develop-30b315c`, no code change. Report only, no gate.
Specced as Sonnet 4.6; the company Bedrock account refuses every 4.x model until Anthropic's use-case form is
filed. Compare with `m3-sonnet-5-5-same-day.md`, run right after on the same image.

- Commit: `30b315c` · runs per question: 3 · questions: 10
- Model: eu.anthropic.claude-sonnet-5 (all models used: eu.anthropic.claude-sonnet-5)
- Prompt version: `5fdfaaf2e87b` · SDK: pydantic-ai-slim 2.54.0
- Runtime: pydantic-ai · provider: bedrock · cost source: genai-prices 0.1.9 (regional.anthropic.claude-sonnet-5-v1:0)
- Core commit: `bb95f2f8c338e6433a1e6a48375cf634710ff421` · catalogue: 133 operations
- dtcc-sim: available (probe names simulations)
- Spent: $1.16 of a $10.00 cap

Each cell is median / max over the question's successful runs; failed runs are left out
and counted under OK and Problems. Run 1 is cold, later runs warm. Input tokens include
cache reads and writes. Cost is computed as the cost source above says.

| Question | OK | Cold latency (s) | Warm latency (s) | Input tok | Output tok | Cost ($) | Tools | Ops | Problems |
|---|---|---|---|---|---|---|---|---|---|
| q01-building-count | 3 / 3 | 17.1 / 17.1 | 9.2 / 11.5 | 24,701 / 34,213 | 262 / 271 | 0.010 / 0.036 | 2 / 2 | 2 / 2 |  |
| q02-cached-subarea | 3 / 3 | 9.6 / 9.6 | 9.3 / 9.9 | 25,871 / 34,277 | 501 / 549 | 0.017 / 0.039 | 2 / 2 | 2 / 2 |  |
| q03-terrain-slope | 3 / 3 | 33.4 / 33.4 | 27.7 / 31.9 | 67,927 / 95,943 | 1,820 / 1,953 | 0.046 / 0.074 | 7 / 8 | 7 / 7 |  |
| q04-render-buildings | 3 / 3 | 18.7 / 18.7 | 12.6 / 13.9 | 53,344 / 71,363 | 712 / 716 | 0.047 / 0.051 | 4 / 4 | 4 / 4 |  |
| q05-export-geojson | 3 / 3 | 9.8 / 9.8 | 10.1 / 10.5 | 33,406 / 33,562 | 420 / 486 | 0.014 / 0.015 | 3 / 3 | 3 / 3 |  |
| q06-discover-operations | 3 / 3 | 9.4 / 9.4 | 9.3 / 10.0 | 17,523 / 23,385 | 714 / 885 | 0.016 / 0.034 | 1 / 2 | 1 / 2 |  |
| q07-trees | 3 / 3 | 21.4 / 21.4 | 23.9 / 26.9 | 73,893 / 75,235 | 1,433 / 2,004 | 0.062 / 0.070 | 6 / 6 | 6 / 6 |  |
| q08-refused-path | 3 / 3 | 4.8 / 4.8 | 4.1 / 4.1 | 7,953 / 7,953 | 203 / 213 | 0.004 / 0.004 | 0 / 0 | 0 / 0 |  |
| q09-too-large | 3 / 3 | 34.4 / 34.4 | 24.6 / 29.4 | 44,512 / 54,858 | 1,767 / 1,821 | 0.035 / 0.038 | 4 / 5 | 4 / 5 |  |
| q10-heatwave | 3 / 3 | 119.1 / 119.1 | 101.4 / 121.7 | 144,967 / 157,208 | 3,672 / 4,038 | 0.102 / 0.108 | 13 / 14 | 13 / 14 |  |

**Totals:** 30 of 30 runs ok · $1.13 · 742 s of answering
