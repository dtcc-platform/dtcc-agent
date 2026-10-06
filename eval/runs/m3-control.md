# #87 same-day control: the T36 code, two runs pooled (2026-10-06)

**The code measured is `9378511` (image `dtcc-agent:ctl-9378511`), not the commit below:** the
harness records the checkout it ran from. Run interleaved with `m3-catalogue-c.md` to separate
Bedrock's day-to-day speed from the prompt. The same code was 18% slower than `m3-runtime.md`.

- Pooled from 2 runs: `20261006T095509Z-ab53b91`, `20261006T102010Z-ab53b91`
- Commit: `ab53b91` · runs per question: 6 · questions: 10
- Model: eu.anthropic.claude-sonnet-5-5 (all models used: eu.anthropic.claude-sonnet-5-5)
- Prompt version: `5fdfaaf2e87b` · SDK: pydantic-ai-slim 2.54.0
- Runtime: pydantic-ai · provider: bedrock · cost source: genai-prices 0.1.9 (regional.anthropic.claude-sonnet-5-v1:0)
- Core commit: `bb95f2f8c338e6433a1e6a48375cf634710ff421` · catalogue: 133 operations
- dtcc-sim: available
- Spent: $1.44 across the runs

Each cell is median / max over the question's successful runs; failed runs are left out
and counted under OK and Problems. Run 1 is cold, later runs warm. Input tokens include
cache reads and writes. Cost is computed as the cost source above says.

| Question | OK | Cold latency (s) | Warm latency (s) | Input tok | Output tok | Cost ($) | Tools | Ops | Problems |
|---|---|---|---|---|---|---|---|---|---|
| q01-building-count | 6 / 6 | 23.0 / 31.1 | 5.8 / 6.1 | 25,067 / 25,312 | 322 / 384 | 0.011 / 0.013 | 2 / 2 | 2 / 2 |  |
| q02-cached-subarea | 6 / 6 | 11.4 / 14.3 | 7.0 / 8.0 | 25,514 / 25,539 | 468 / 555 | 0.015 / 0.016 | 2 / 2 | 2 / 2 |  |
| q03-terrain-slope | 6 / 6 | 20.1 / 20.1 | 14.8 / 15.7 | 53,918 / 53,998 | 1,326 / 1,489 | 0.032 / 0.035 | 7 / 7 | 7 / 7 |  |
| q04-render-buildings | 6 / 6 | 10.3 / 10.4 | 11.9 / 12.6 | 52,862 / 52,893 | 836 / 892 | 0.048 / 0.048 | 4 / 4 | 4 / 4 |  |
| q05-export-geojson | 6 / 6 | 8.4 / 9.2 | 9.0 / 11.7 | 33,018 / 33,066 | 519 / 539 | 0.014 / 0.015 | 3 / 3 | 3 / 3 |  |
| q06-discover-operations | 6 / 6 | 7.3 / 7.3 | 8.4 / 9.0 | 17,879 / 18,183 | 946 / 1,051 | 0.020 / 0.022 | 2 / 3 | 2 / 3 |  |
| q07-trees | 6 / 6 | 13.9 / 15.0 | 12.8 / 18.5 | 39,637 / 39,750 | 977 / 1,096 | 0.030 / 0.031 | 4 / 4 | 4 / 4 |  |
| q08-refused-path | 5 / 6 | 5.2 / 5.8 | 5.8 / 6.4 | 7,887 / 7,887 | 485 / 516 | 0.007 / 0.007 | 0 / 0 | 0 / 0 | 1 error |
| q09-too-large | 6 / 6 | 44.0 / 66.8 | 17.5 / 25.0 | 35,101 / 66,200 | 1,236 / 1,985 | 0.026 / 0.045 | 4 / 6 | 4 / 6 |  |
| q10-heatwave | 6 / 6 | 76.1 / 87.0 | 77.6 / 112.7 | 54,112 / 68,655 | 1,432 / 1,592 | 0.037 / 0.040 | 6 / 7 | 6 / 7 |  |

**Totals:** 59 of 60 runs ok · $1.44 · 1,149 s of answering
