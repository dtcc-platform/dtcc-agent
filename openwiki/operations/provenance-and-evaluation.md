---
type: operations
title: Provenance and evaluation
description: How every answer can be traced and measured. The chatbot writes one answers.jsonl line per turn (model, prompt version, tokens, cost, error and HTTP status), the MCP server one operations.jsonl line per tool call, joined on the turn id, and eval/measure.py runs a fixed question set through the chat to report latency, tokens and cost, compared only against a same-day control.
tags: [provenance, evaluation, measurement, cost, logging]
verified:
  - by: openwiki/0.5.2
    at: 2026-10-31T10:29:51.442Z
sources:
  - id: openwiki-source-d82fbc21a9f74516f7bfd0f8
    resource: repo://chatbot/app.py
  - id: openwiki-source-4b191c2a4922e158f63963c3
    resource: repo://chatbot/prices.py
  - id: openwiki-source-06f9816c33b0f5c6fdb9d8ed
    resource: repo://chatbot/provenance.py
  - id: openwiki-source-6ed7b95a502f0a6d7126df1d
    resource: repo://chatbot/runtime/pai.py
  - id: openwiki-source-6150372e327ba33b9fdace5c
    resource: repo://docs/adr/0008-evaluation-is-one-harness-with-two-layers.md
  - id: openwiki-source-5362e3b27c4a029e63907699
    resource: repo://docs/adr/0011-simulations-on-the-engine-downloads-stay-in-process.md
  - id: openwiki-source-00348018efcad3e081121b7b
    resource: repo://dtcc_agent/provenance.py
  - id: openwiki-source-4a2ecebedee07cd459d47e0a
    resource: repo://eval/measure.py
  - id: openwiki-source-015270975ffd802e33601bcf
    resource: repo://eval/questions.json
  - id: openwiki-source-3263a9523d9bc511ec36d404
    resource: repo://eval/runs/m4-engine-vs-inprocess/README.md
  - id: openwiki-source-b57c6287b311fd4ffadcbf36
    resource: repo://tests/test_eval_measure.py
  - id: openwiki-source-34f19409acce27bb03b9d119
    resource: repo://tests/test_limits.py
generated: { by: "claude-code", at: "2026-10-31T10:29:51.442Z" }
---

# Provenance and evaluation

Two pieces answer "what produced this answer, and how fast and costly was it": first-hand **provenance records** written by both services (T33, #77), and the **measurement harness** that runs a fixed question set through the chat (ADR-0008's measurement layer, T34, #78).

## Provenance: two records joined on the turn

Each service writes only what it knows itself, as JSON lines under `DTCC_AGENT_LOG_DIR` (Docker: `/data/logs`). The chatbot mints a turn id (`turn_…`) per user message and sends it to the MCP server as `X-DTCC-Turn` (stdio: `DTCC_AGENT_TURN`), which is how the two files join.

**`answers.jsonl`** (`chatbot/provenance.py`, one line per turn): the time, turn, session and subject; the model and every model used; the prompt version (first 12 hex characters of the system prompt's SHA-256); whether memory was added; the runtime, its package (in the field still named `sdk`, so older records compare) and the provider (`bedrock`); the tools called; summed totals and token usage; cost and its source; and the outcome: `is_error`, `retried_fresh`, `refused`, the error class and, since #106, `http_status`. Every field is present; unknowns are null. A turn a limit stopped (#132) is recorded like a failed one: `error` is `LimitReached` (turns per chat, or the monthly spend) or `TimeoutError` (the turn deadline). With a monthly spend limit set, the chat server also keeps `spend.json` next to these records: this month's spend and what running requests have set aside, not a record per turn; see [Turn and spend limits](turn-and-spend-limits.md). The chat app writes exactly one line per turn, including turns cut short, and also sends it to the page as a `provenance` frame before `done`. See [Chatbot (Lurkie)](../integrations/chatbot-lurkie.md).

**`operations.jsonl`** (`dtcc_agent/provenance.py`, one line per tool call, written by the `@tool` wrapper): the turn, Session, tool and operation, a hash of the arguments, duration, success, cache hit, the Object references returned and the catalogue it ran against. A `catalogue` line records each catalogue build with the Core commit and operation count.

Neither file holds parameter values, error texts or file contents: arguments are hashed and an error is reduced to its category. A record never fails the call it describes; a failed write only warns. The server writes nothing without `DTCC_AGENT_LOG_DIR`.

`python -m dtcc_agent.provenance join <log_dir> [--turn <id>]` prints one complete record per turn: the answer with its operations.

**Cost** (`chatbot/prices.py`): pydantic-ai's usage is priced with genai-prices on AWS's prices, with cache writes at the 5-minute rate the cache settings use; a model genai-prices does not know is recorded as `unpriced` rather than guessed.

## The measurement harness

`python -m eval.measure --url ws://localhost:8050/chat --runs 3 [--log-dir data/agent/logs]` asks every question of `eval/questions.json` through the real WebSocket, a fresh chat each time so no history leaks. The access code comes from `DTCC_AGENT_ACCESS_CODE`.

- **Questions.** Ten, with stable ids (`q01-building-count` to `q10-heatwave`): building counts, a cached sub-area, terrain slope, rendering, a GeoJSON export, operation discovery, trees, a refused path, a too-large area and the heatwave simulation. A question's text changes only with a new id, so before/after numbers compare like with like. `answer_key` is reserved for the correctness layer.
- **Order.** Runs are outermost, so run 1 is every question's cold run (empty download cache) and later runs are warm; cold and warm latency are reported apart, because a cached download changes latency several-fold.
- **Simulations.** A probe asks the chat to list simulations; if none are named, simulation questions are skipped. `--assume-sim` skips the probe.
- **Guards.** `--max-cost` (default $2) stops the run cleanly once spent; `--timeout` bounds each question; `--only` picks questions.
- **Output.** `eval/runs/<UTC time>-<git sha>.jsonl` (one line per question and run) and a Markdown report with median and max per question over successful runs only, failures counted apart. `--pool` combines several runs' JSONL into one report.

Baselines live in `eval/runs/`: `baseline-m2.md` (30/30 runs, $1.93), `baseline-m3-sdk.md` (M3's gate reference) and the M3 experiments (`m3-runtime`, `m3-catalogue-a/b/c`, `m3-control`, `m3-sonnet-5`, `m3-sonnet-5-5-same-day`). M4's `m4-engine-vs-inprocess/` holds T41's measurement (#110): point-cloud questions in-process against Engine jobs, arms alternating on one day, at one and at four Engine workers, with pooled reports per arm. It is the evidence behind ADR-0011.

## The same-day control rule

Bedrock's latency drifts from day to day: unchanged code ran 18% slower on 2026-10-06 than on 2026-10-05. So a speed comparison runs the reference and the candidate **on the same day, interleaved**, and compares the candidate with today's reference, never with a number recorded earlier (ADR-0008 update, #100; README, "Comparing two versions"). M4's in-process-versus-Engine measurement (#110) follows this rule.

## Tests

- `tests/test_provenance.py`: both record shapes with every field present, hashing instead of values, error categories, never failing the call, the join, and `http_status` in the answer record.
- `tests/test_eval_measure.py`: the committed question set is valid, malformed sets are refused, run order makes run 1 cold, the simulation probe, the cost cap stopping cleanly, and medians and maxes over successful runs only.

See [Model runtime](../architecture/model-runtime.md) for where usage and failures come from, and [Test suite](../testing/test-suite.md) for the wider suite.
