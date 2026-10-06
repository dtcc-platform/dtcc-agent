# #87 variant C: the summary plus the seven M2 schemas in the cached prefix, two runs pooled (2026-10-06)

`list_operations` plus `describe_operation` for the seven operations M2 pasted. Run interleaved
with `m3-control.md`. Not adopted: see the ADR-0006 update note.

- Pooled from 2 runs: `20261006T100434Z-ab53b91`, `20261006T103211Z-ab53b91`
- Commit: `ab53b91` · runs per question: 6 · questions: 10
- Model: eu.anthropic.claude-sonnet-5-5 (all models used: eu.anthropic.claude-sonnet-5-5)
- Prompt version: `05adbbfb569b` · SDK: pydantic-ai-slim 2.54.0
- Runtime: pydantic-ai · provider: bedrock · cost source: genai-prices 0.1.9 (regional.anthropic.claude-sonnet-5-v1:0)
- Core commit: `bb95f2f8c338e6433a1e6a48375cf634710ff421` · catalogue: 133 operations
- dtcc-sim: available
- Spent: $2.43 across the runs

Each cell is median / max over the question's successful runs; failed runs are left out
and counted under OK and Problems. Run 1 is cold, later runs warm. Input tokens include
cache reads and writes. Cost is computed as the cost source above says.

| Question | OK | Cold latency (s) | Warm latency (s) | Input tok | Output tok | Cost ($) | Tools | Ops | Problems |
|---|---|---|---|---|---|---|---|---|---|
| q01-building-count | 6 / 6 | 25.9 / 34.5 | 7.8 / 8.4 | 81,720 / 81,781 | 428 / 550 | 0.069 / 0.070 | 2 / 2 | 2 / 2 |  |
| q02-cached-subarea | 6 / 6 | 8.2 / 9.6 | 7.8 / 9.2 | 64,877 / 64,936 | 494 / 533 | 0.024 / 0.025 | 2 / 2 | 2 / 2 |  |
| q03-terrain-slope | 6 / 6 | 21.5 / 21.7 | 16.3 / 21.8 | 109,564 / 132,481 | 1,322 / 1,471 | 0.045 / 0.052 | 6 / 7 | 6 / 7 |  |
| q04-render-buildings | 6 / 6 | 44.5 / 48.1 | 43.9 / 62.1 | 127,739 / 137,913 | 876 / 901 | 0.058 / 0.069 | 5 / 5 | 5 / 5 |  |
| q05-export-geojson | 6 / 6 | 18.3 / 27.2 | 10.7 / 13.3 | 85,748 / 85,815 | 636 / 670 | 0.028 / 0.029 | 3 / 3 | 3 / 3 |  |
| q06-discover-operations | 6 / 6 | 8.2 / 8.8 | 7.7 / 9.0 | 21,003 / 21,003 | 1,070 / 1,117 | 0.017 / 0.017 | 0 / 0 | 0 / 0 |  |
| q07-trees | 6 / 6 | 17.8 / 18.5 | 22.2 / 28.4 | 105,210 / 143,078 | 982 / 1,177 | 0.055 / 0.062 | 5 / 8 | 5 / 8 |  |
| q08-refused-path | 5 / 6 | 7.1 / 7.7 | 4.9 / 5.8 | 21,007 / 42,474 | 462 / 634 | 0.010 / 0.018 | 0 / 1 | 0 / 1 | 1 error |
| q09-too-large | 6 / 6 | 58.8 / 86.8 | 34.4 / 66.9 | 110,550 / 157,723 | 1,530 / 1,907 | 0.049 / 0.064 | 5 / 8 | 5 / 8 |  |
| q10-heatwave | 6 / 6 | 143.5 / 174.9 | 115.6 / 149.3 | 122,662 / 152,924 | 1,634 / 2,230 | 0.057 / 0.077 | 6 / 9 | 6 / 9 |  |

**Totals:** 59 of 60 runs ok · $2.43 · 1,816 s of answering
