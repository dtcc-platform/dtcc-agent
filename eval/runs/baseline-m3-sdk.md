# M3 gate reference: the Agent SDK on Bedrock, two runs pooled (2026-10-04 22:04 and 22:22 UTC)

- Pooled from 2 runs: `20261004T220414Z-713b47a`, `20261004T222251Z-0e1c2fd`
- Commit: `713b47a, 0e1c2fd` · runs per question: 6 · questions: 10
- Model: claude-sonnet-5-5 (all models used: eu.anthropic.claude-sonnet-5-5)
- Prompt version: `5fdfaaf2e87b` · SDK: claude-agent-sdk 0.2.163
- Runtime: sdk · provider: bedrock · cost source: sdk total_cost_usd
- Core commit: `bb95f2f8c338e6433a1e6a48375cf634710ff421` · catalogue: 133 operations
- dtcc-sim: available
- Spent: $1.90 across the runs

Each cell is median / max over the question's successful runs; failed runs are left out
and counted under OK and Problems. Run 1 is cold, later runs warm. Input tokens include
cache reads and writes. Cost is computed as the cost source above says.

| Question | OK | Cold latency (s) | Warm latency (s) | Input tok | Output tok | Cost ($) | Tools | Ops | Problems |
|---|---|---|---|---|---|---|---|---|---|
| q01-building-count | 6 / 6 | 23.3 / 24.7 | 11.1 / 13.5 | 41,780 / 41,794 | 649 / 675 | 0.070 / 0.077 | 3 / 3 | 2 / 2 |  |
| q02-cached-subarea | 6 / 6 | 11.5 / 11.9 | 11.0 / 11.6 | 20,758 / 21,598 | 742 / 810 | 0.022 / 0.026 | 3 / 3 | 2 / 2 |  |
| q03-terrain-slope | 6 / 6 | 20.2 / 22.0 | 17.1 / 20.4 | 36,014 / 43,521 | 1,452 / 1,527 | 0.031 / 0.040 | 7 / 7 | 6 / 6 |  |
| q04-render-buildings | 6 / 6 | 14.2 / 15.1 | 12.8 / 13.8 | 32,748 / 51,050 | 942 / 1,046 | 0.030 / 0.057 | 5 / 6 | 4 / 5 |  |
| q05-export-geojson | 6 / 6 | 11.8 / 12.2 | 11.5 / 11.9 | 25,411 / 25,416 | 690 / 707 | 0.016 / 0.024 | 4 / 4 | 3 / 3 |  |
| q06-discover-operations | 6 / 6 | 11.8 / 12.1 | 11.1 / 11.6 | 14,218 / 14,218 | 1,058 / 1,134 | 0.016 / 0.027 | 3 / 3 | 2 / 2 |  |
| q07-trees | 6 / 6 | 17.1 / 18.4 | 15.8 / 17.5 | 45,689 / 53,921 | 1,317 / 1,637 | 0.053 / 0.067 | 7 / 7 | 6 / 6 |  |
| q08-refused-path | 6 / 6 | 6.6 / 6.8 | 7.0 / 7.8 | 3,456 / 3,456 | 330 / 403 | 0.007 / 0.010 | 0 / 0 | 0 / 0 |  |
| q09-too-large | 6 / 6 | 47.0 / 49.4 | 39.0 / 55.4 | 26,616 / 26,638 | 1,780 / 1,852 | 0.037 / 0.040 | 7 / 7 | 6 / 6 |  |
| q10-heatwave | 6 / 6 | 63.1 / 67.1 | 60.9 / 65.1 | 41,247 / 65,676 | 1,658 / 1,867 | 0.035 / 0.059 | 6 / 8 | 5 / 7 |  |

**Totals:** 60 of 60 runs ok · $1.90 · 1,238 s of answering
## Notes

This is the reference the M3 latency gate compares against (#29, #86). It changes one thing
from `baseline-m2.md`: the provider (Bedrock instead of Anthropic direct) and, with it, the
model. The runtime is still the Agent SDK. Measurement only: nothing here says whether an
answer was right.

**Why two runs, pooled.** Two identical runs disagreed by up to 21% on one question's warm
median (q01: 12.6 s vs 10.0 s) and 50% on another (q09: 48.9 s vs 24.3 s). In q09 the model's own time is steady at about 15 s;
the spread is dtcc-core operations (19–38 s per run), plus one run where the agent answered
in 4 requests without running an operation (12.8 s). Their aggregate barely moved (median of warm
medians 12.7 s vs 12.1 s). With 2 warm samples per question, a 25% per-question cap would
fail on luck. Pooled, each question has 4 warm samples (6 runs, of which 2 are cold). The
gate compares this pooled reference with a pooled pair of pydantic-ai runs. Regenerate with:

```sh
python -m eval.measure --pool eval/runs/baseline-m3-sdk/*.jsonl
```

**Setup.**
- **Image:** `dtcc-agent:t35`, built from branch `t35-sdk-on-bedrock` for `linux/amd64` under
  emulation, on an Apple Silicon Mac (Docker VM: 8 GB). The first run's report names
  `713b47a` because the branch's change was not yet committed; the image was the same for both.
- **Model:** `eu.anthropic.claude-sonnet-5-5` for answers and for the CLI's internal steps.
  Sonnet 4.5, M2's model, is refused on the account (404, Anthropic use-case form not
  submitted), so this reference does not share M2's model.
- **Bedrock:** `eu-north-1`, a Bedrock API key.
- **Cache:** a fresh data directory for each run, so each started with an empty download cache.
- **dtcc-sim:** the `dtcc-sim:local` image. Available for both runs.
- **Other containers:** a Supabase stack and a Postgres container (about 300 MB together)
  were running in the same Docker VM during both runs. The gate's pydantic-ai runs should
  use the same setup.
- **Clean runs:** neither container restarted, all 62 turns (including two probes) wrote
  provenance, none errored or retried, every answer used only `eu.anthropic.claude-sonnet-5-5`,
  and the agent used only `ToolSearch` and dtcc-agent tools.

**Against M2 (context, not a gate).** 60 of 60 ok. Per run, about $0.95 and 620 s of
answering, against $1.93 and 1,157 s on M2. Part of that is the model, part the provider,
and this run cannot separate them. Cost is the CLI's own `total_cost_usd`; whether it prices
Bedrock's `eu.` profile exactly is not verified here.
