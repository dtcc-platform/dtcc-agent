# Measurement run 20261019T232454Z-0457229 … 20261019T233726Z-0457229

- Pooled from 2 runs: `20261019T232454Z-0457229`, `20261019T233726Z-0457229`
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
| q03-terrain-slope | 6 / 6 | 18.3 / 20.6 | 14.1 / 15.6 | 54,278 / 54,396 | 1,346 / 1,379 | 0.027 / 0.047 | 6 / 6 | 6 / 6 |  |
| q09-too-large | 5 / 6 | 16.9 / 16.9 | 14.7 / 32.7 | 36,346 / 36,389 | 1,826 / 2,037 | 0.032 / 0.035 | 6 / 6 | 6 / 6 | 1 error |

**Totals:** 11 of 12 runs ok · $0.33 · 203 s of answering
