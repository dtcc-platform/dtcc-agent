# #88 same-day Sonnet 5.5, one run (2026-10-06)

The default model on image `develop-30b315c`, run right after `m3-sonnet-5.md` so the two differ only in model.

- Commit: `30b315c` · runs per question: 3 · questions: 10
- Model: eu.anthropic.claude-sonnet-5-5 (all models used: eu.anthropic.claude-sonnet-5-5)
- Prompt version: `5fdfaaf2e87b` · SDK: pydantic-ai-slim 2.54.0
- Runtime: pydantic-ai · provider: bedrock · cost source: genai-prices 0.1.9 (regional.anthropic.claude-sonnet-5-v1:0)
- Core commit: `bb95f2f8c338e6433a1e6a48375cf634710ff421` · catalogue: 133 operations
- dtcc-sim: available (probe names simulations)
- Spent: $0.72 of a $10.00 cap

Each cell is median / max over the question's successful runs; failed runs are left out
and counted under OK and Problems. Run 1 is cold, later runs warm. Input tokens include
cache reads and writes. Cost is computed as the cost source above says.

| Question | OK | Cold latency (s) | Warm latency (s) | Input tok | Output tok | Cost ($) | Tools | Ops | Problems |
|---|---|---|---|---|---|---|---|---|---|
| q01-building-count | 3 / 3 | 15.1 / 15.1 | 5.2 / 5.2 | 24,825 / 24,825 | 342 / 351 | 0.009 / 0.011 | 2 / 2 | 2 / 2 |  |
| q02-cached-subarea | 3 / 3 | 6.1 / 6.1 | 7.5 / 8.7 | 25,526 / 25,532 | 506 / 509 | 0.015 / 0.015 | 2 / 2 | 2 / 2 |  |
| q03-terrain-slope | 3 / 3 | 17.4 / 17.4 | 15.4 / 15.9 | 53,943 / 53,954 | 1,271 / 1,386 | 0.032 / 0.033 | 7 / 7 | 7 / 7 |  |
| q04-render-buildings | 3 / 3 | 12.6 / 12.6 | 9.7 / 9.7 | 52,802 / 52,804 | 758 / 764 | 0.047 / 0.047 | 4 / 4 | 4 / 4 |  |
| q05-export-geojson | 3 / 3 | 8.8 / 8.8 | 7.7 / 8.4 | 33,017 / 33,024 | 497 / 518 | 0.014 / 0.015 | 3 / 3 | 3 / 3 |  |
| q06-discover-operations | 3 / 3 | 7.0 / 7.0 | 8.3 / 9.5 | 17,879 / 17,879 | 958 / 1,012 | 0.015 / 0.020 | 2 / 2 | 2 / 2 |  |
| q07-trees | 3 / 3 | 14.5 / 14.5 | 11.2 / 11.6 | 39,634 / 39,689 | 1,041 / 1,094 | 0.031 / 0.031 | 4 / 4 | 4 / 4 |  |
| q08-refused-path | 2 / 3 | 4.9 / 4.9 | 4.5 / 4.5 | 7,887 / 7,887 | 420 / 424 | 0.006 / 0.007 | 0 / 0 | 0 / 0 | 1 error |
| q09-too-large | 3 / 3 | 11.7 / 11.7 | 16.3 / 19.2 | 24,939 / 34,260 | 1,109 / 1,301 | 0.020 / 0.026 | 2 / 3 | 2 / 3 |  |
| q10-heatwave | 3 / 3 | 57.6 / 57.6 | 60.4 / 63.8 | 68,661 / 68,761 | 1,499 / 1,548 | 0.039 / 0.045 | 7 / 7 | 7 / 7 |  |

**Totals:** 29 of 30 runs ok · $0.69 · 446 s of answering
