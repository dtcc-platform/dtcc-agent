# #87 variant A: the summary catalogue in the cached prefix, two runs pooled (2026-10-06)

Base prompt without the seven pasted schemas, plus `list_operations` (8,759 tokens) as a static
instruction. Not adopted: see the ADR-0006 update note and `m3-control.md`.

- Pooled from 2 runs: `20261006T065712Z-f86766d`, `20261006T070855Z-f86766d`
- Commit: `f86766d` · runs per question: 6 · questions: 10
- Model: eu.anthropic.claude-sonnet-5-5 (all models used: eu.anthropic.claude-sonnet-5-5)
- Prompt version: `a114390ebe33` · SDK: pydantic-ai-slim 2.54.0
- Runtime: pydantic-ai · provider: bedrock · cost source: genai-prices 0.1.9 (regional.anthropic.claude-sonnet-5-v1:0)
- Core commit: `bb95f2f8c338e6433a1e6a48375cf634710ff421` · catalogue: 133 operations
- dtcc-sim: available
- Spent: $2.27 across the runs

Each cell is median / max over the question's successful runs; failed runs are left out
and counted under OK and Problems. Run 1 is cold, later runs warm. Input tokens include
cache reads and writes. Cost is computed as the cost source above says.

| Question | OK | Cold latency (s) | Warm latency (s) | Input tok | Output tok | Cost ($) | Tools | Ops | Problems |
|---|---|---|---|---|---|---|---|---|---|
| q01-building-count | 6 / 6 | 17.9 / 19.0 | 5.7 / 6.3 | 60,808 / 60,814 | 478 / 578 | 0.019 / 0.045 | 2 / 2 | 2 / 2 |  |
| q02-cached-subarea | 6 / 6 | 6.1 / 6.4 | 6.8 / 10.0 | 52,368 / 52,386 | 515 / 523 | 0.021 / 0.021 | 2 / 2 | 2 / 2 |  |
| q03-terrain-slope | 6 / 6 | 27.7 / 31.6 | 26.6 / 29.8 | 153,122 / 210,233 | 2,004 / 2,294 | 0.068 / 0.086 | 12 / 14 | 12 / 14 |  |
| q04-render-buildings | 6 / 6 | 18.3 / 19.8 | 16.5 / 26.6 | 88,850 / 117,364 | 824 / 984 | 0.056 / 0.064 | 4 / 5 | 4 / 5 |  |
| q05-export-geojson | 6 / 6 | 8.6 / 8.6 | 8.1 / 10.8 | 70,770 / 70,793 | 652 / 728 | 0.024 / 0.026 | 4 / 4 | 4 / 4 |  |
| q06-discover-operations | 6 / 6 | 7.3 / 7.6 | 7.3 / 7.6 | 16,832 / 16,832 | 1,084 / 1,129 | 0.016 / 0.016 | 0 / 0 | 0 / 0 |  |
| q07-trees | 6 / 6 | 17.0 / 19.3 | 14.2 / 14.5 | 117,776 / 145,369 | 1,043 / 1,257 | 0.046 / 0.075 | 6 / 7 | 6 / 7 |  |
| q08-refused-path | 5 / 6 | 5.3 / 5.3 | 5.8 / 7.5 | 16,836 / 34,202 | 519 / 658 | 0.010 / 0.016 | 0 / 1 | 0 / 1 | 1 error |
| q09-too-large | 6 / 6 | 41.5 / 42.7 | 18.9 / 32.9 | 74,094 / 74,123 | 1,871 / 2,026 | 0.047 / 0.049 | 7 / 7 | 7 / 7 |  |
| q10-heatwave | 6 / 6 | 78.5 / 80.6 | 85.1 / 87.7 | 102,998 / 126,536 | 1,690 / 2,124 | 0.053 / 0.062 | 8 / 8 | 8 / 8 |  |

**Totals:** 59 of 60 runs ok · $2.27 · 1,249 s of answering
