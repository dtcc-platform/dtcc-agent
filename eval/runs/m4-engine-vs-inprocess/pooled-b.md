# Measurement run 20261019T232727Z-0457229 … 20261019T233313Z-0457229

- Pooled from 2 runs: `20261019T232727Z-0457229`, `20261019T233313Z-0457229`
- Commit: `0457229` · runs per question: 6 · questions: 2
- Model: eu.anthropic.claude-sonnet-5-5 (all models used: eu.anthropic.claude-sonnet-5-5)
- Prompt version: `a9f7e88fc432` · SDK: pydantic-ai-slim 2.54.0
- Runtime: pydantic-ai · provider: bedrock · cost source: genai-prices 0.1.13 (regional.anthropic.claude-sonnet-5-5)
- Core commit: `a78df16e9c86b349ba3a67551d518223b7897443` · catalogue: 133 operations
- dtcc-sim: available
- Spent: $0.33 across the runs

Each cell is median / max over the question's successful runs; failed runs are left out
and counted under OK and Problems. Run 1 is cold, later runs warm. Input tokens include
cache reads and writes. Cost is computed as the cost source above says.

| Question | OK | Cold latency (s) | Warm latency (s) | Input tok | Output tok | Cost ($) | Tools | Ops | Problems |
|---|---|---|---|---|---|---|---|---|---|
| q03-terrain-slope | 6 / 6 | 20.5 / 24.7 | 17.1 / 18.6 | 44,472 / 44,582 | 1,414 / 1,535 | 0.026 / 0.027 | 6 / 6 | 6 / 6 |  |
| q09-too-large | 6 / 6 | 83.8 / 124.7 | 77.1 / 117.6 | 35,584 / 37,197 | 1,562 / 2,496 | 0.027 / 0.042 | 4 / 7 | 4 / 7 |  |

**Totals:** 12 of 12 runs ok · $0.33 · 560 s of answering
