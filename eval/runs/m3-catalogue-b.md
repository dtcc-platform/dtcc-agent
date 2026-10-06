# #87 variant B: every operation's schema in the cached prefix, two runs pooled (2026-10-06)

Base prompt without the seven pasted schemas, plus every `describe_operation` (57,733 tokens).
Not adopted: see the ADR-0006 update note.

- Pooled from 2 runs: `20261006T071947Z-f86766d`, `20261006T073042Z-f86766d`
- Commit: `f86766d` · runs per question: 6 · questions: 10
- Model: eu.anthropic.claude-sonnet-5-5 (all models used: eu.anthropic.claude-sonnet-5-5)
- Prompt version: `a49c0a25bb84` · SDK: pydantic-ai-slim 2.54.0
- Runtime: pydantic-ai · provider: bedrock · cost source: genai-prices 0.1.9 (regional.anthropic.claude-sonnet-5-v1:0)
- Core commit: `bb95f2f8c338e6433a1e6a48375cf634710ff421` · catalogue: 133 operations
- dtcc-sim: available
- Spent: $4.66 across the runs

Each cell is median / max over the question's successful runs; failed runs are left out
and counted under OK and Problems. Run 1 is cold, later runs warm. Input tokens include
cache reads and writes. Cost is computed as the cost source above says.

| Question | OK | Cold latency (s) | Warm latency (s) | Input tok | Output tok | Cost ($) | Tools | Ops | Problems |
|---|---|---|---|---|---|---|---|---|---|
| q01-building-count | 6 / 6 | 17.1 / 17.8 | 6.7 / 6.8 | 222,517 / 230,975 | 512 / 595 | 0.068 / 0.102 | 2 / 2 | 2 / 2 |  |
| q02-cached-subarea | 6 / 6 | 6.7 / 6.9 | 6.1 / 6.8 | 214,062 / 214,080 | 500 / 577 | 0.056 / 0.058 | 2 / 2 | 2 / 2 |  |
| q03-terrain-slope | 6 / 6 | 17.3 / 17.4 | 14.0 / 16.2 | 430,870 / 431,292 | 1,310 / 1,360 | 0.115 / 0.117 | 7 / 7 | 7 / 7 |  |
| q04-render-buildings | 6 / 6 | 27.7 / 28.6 | 27.9 / 28.4 | 304,384 / 386,629 | 772 / 876 | 0.102 / 0.123 | 4 / 5 | 4 / 5 |  |
| q05-export-geojson | 6 / 6 | 18.9 / 23.2 | 19.1 / 23.5 | 356,226 / 428,240 | 750 / 944 | 0.089 / 0.107 | 4 / 5 | 4 / 5 |  |
| q06-discover-operations | 6 / 6 | 7.1 / 7.1 | 6.6 / 7.6 | 70,726 / 70,726 | 1,014 / 1,168 | 0.027 / 0.028 | 0 / 0 | 0 / 0 |  |
| q07-trees | 6 / 6 | 12.2 / 12.4 | 12.7 / 14.5 | 290,510 / 290,545 | 907 / 954 | 0.083 / 0.085 | 4 / 4 | 4 / 4 |  |
| q08-refused-path | 3 / 6 | 5.6 / 6.2 | 4.9 / 4.9 | 70,730 / 70,730 | 492 / 549 | 0.021 / 0.022 | 0 / 0 | 0 / 0 | 3 error |
| q09-too-large | 6 / 6 | 45.2 / 46.6 | 36.0 / 40.4 | 358,431 / 432,467 | 1,828 / 2,069 | 0.104 / 0.124 | 6 / 6 | 6 / 6 |  |
| q10-heatwave | 6 / 6 | 81.2 / 102.0 | 76.5 / 85.4 | 370,422 / 525,348 | 1,606 / 1,867 | 0.106 / 0.146 | 6 / 8 | 6 / 8 |  |

**Totals:** 57 of 60 runs ok · $4.66 · 1,287 s of answering
