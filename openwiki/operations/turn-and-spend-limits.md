---
type: operations
title: Turn and spend limits
description: The four deployment limits on the chat server (monthly model spend, turns per chat, a deadline per turn, model requests at once), how one gate around every Bedrock request keeps the spend limit while requests are in flight, the spend ledger file and its restart and month rules, and what a user and the answer records see when a limit stops a turn.
tags: [limits, spend, bedrock, deployment, sessions]
verified:
  - by: openwiki/0.5.2
    at: 2026-10-31T10:29:51.442Z
sources:
  - id: openwiki-source-d82fbc21a9f74516f7bfd0f8
    resource: repo://chatbot/app.py
  - id: openwiki-source-df2b3567718802676143cbad
    resource: repo://chatbot/limits.py
  - id: openwiki-source-6ed7b95a502f0a6d7126df1d
    resource: repo://chatbot/runtime/pai.py
  - id: openwiki-source-f4e2e41b78ef4f0591d5eac2
    resource: repo://chatbot/sessions.py
  - id: openwiki-source-34f19409acce27bb03b9d119
    resource: repo://tests/test_limits.py
generated: { by: "claude-code", at: "2026-10-31T10:29:51.442Z" }
---

# Turn and spend limits

The first deployment is an invited beta with a fixed Bedrock budget ($100 a month), and the shared access code admits people without limiting what they spend. `chatbot/limits.py` (#132, merged in #139) adds four settings. Each is **off when unset**, so a local run behaves as before.

| Setting | What it limits | When it is reached | Suggested for the beta |
|---|---|---|---|
| `DTCC_AGENT_MONTHLY_SPEND_USD` | Model spend per calendar month (UTC), across all chats; never exceeded | New model requests are refused: "This service has reached its spending limit for the month, so it can't answer now. Please tell the administrator." | `90` |
| `DTCC_AGENT_TURNS_PER_CHAT` | Turns one chat may start | The turn is refused: "This chat has reached its limit of N questions." | `30` |
| `DTCC_AGENT_TURN_SECONDS` | How long one turn may run | The turn is stopped: "This answer took longer than N seconds and was stopped. Try a smaller question or area." | `180` |
| `DTCC_AGENT_MODEL_CONCURRENCY` | Model requests running at once, across all chats | Further requests wait for a slot; nothing is refused | `4` |

Settings are read by small helpers (`monthly_spend_usd()` and the others). A value that isn't a number, isn't finite (`nan` would switch the spend limit off, because it compares false with everything) or is below the minimum stops the server at startup. The chat app's lifespan logs the limits in force (`Limits: spend $90/month, 30 turns/chat, ...`) and builds the model, so a bad setting or an unreadable ledger fails there and not in a chat. Compose passes all four to the chat container, empty by default. See [Deployment and CI](deployment-and-ci.md).

## One gate for every model request

`model()` in `chatbot/runtime/pai.py` returns the Bedrock model wrapped in `LimitedModel`, a pydantic-ai `WrapperModel`. Every model request goes through it: each step of a turn, the fresh retry of a failed follow-up, and the startup probe. See [Model runtime](../architecture/model-runtime.md).

For each request the gate:

1. **Waits for a slot**, when `DTCC_AGENT_MODEL_CONCURRENCY` is set (an `asyncio.Semaphore`).
2. **Sets aside the request's worst case** in the spend ledger, or raises `LimitReached` if it doesn't fit. Reserving never awaits, so on the event loop no other request can get between the check and the amount it sets aside. At $99 recorded, two $2 turns cannot both start.
3. **Runs the request**, and when it finishes **replaces the worst case with what it actually cost**, priced with genai-prices as the answer records are.

The worst case is bounded by the request's size:

- **Input:** a token covers at least one byte of text, so the serialized request (its messages and tool definitions, plus a 4,096-token margin for what Bedrock adds) bounds its input tokens, capped at the model's context window. It is priced at the dearer of the cache-write and plain input rates.
- **Output:** while a spend limit is set, every request is capped at **8,192 output tokens** (`MAX_OUTPUT_TOKENS`). The longest of 369 recorded answers used 4,038 across all its requests. Without a spend limit, output is not capped (decided 2026-10-31).
- **One attempt:** with a spend limit, the Bedrock client is built with `AWS_MAX_ATTEMPTS=1`, and startup refuses to continue if botocore's own retries are still on. Otherwise a botocore retry after Bedrock accepted a request could be billed inside one reservation.

A model with no known price refuses to start with a spend limit set, since its spend can't be bounded.

### What a failed request costs

| Outcome | Charged |
|---|---|
| Completed | Its actual cost |
| Bedrock refused the request itself with a 4xx before answering (a `ModelHTTPError`, throttling included) | Nothing |
| Anything else: an error after the stream opened, a cancelled or deadline-stopped stream, an error that only *follows* a 4xx | The whole worst case, because Bedrock may have billed it |

A request that cost more than its worst case logs a warning, since that would mean the bound failed.

## The spend ledger

`SpendLedger` keeps `spend.json` in `DTCC_AGENT_LOG_DIR` (`/data/logs` in Compose): the month, what was spent, and what running requests have set aside. A spend limit refuses to start without an explicit `DTCC_AGENT_LOG_DIR`, because the default is under `/tmp`.

- **Writes** go to a fresh, exclusively created temporary file (`tempfile.mkstemp`), are fsynced and then renamed over the ledger, so a crash leaves the old file or the new one, never half of one.
- **Restart:** anything still set aside when the server stopped counts as spent, whatever month it was set aside in. The request may have run, and nothing settled it.
- **A new month** starts from zero, but what running requests have set aside still counts. A request that finishes after midnight is charged to the new month.
- **An unreadable ledger**, or one with negative, non-finite or missing amounts, stops the server: starting from zero could overspend. Deleting the file resets the month's count.
- **One chat server owns the file.** Two instances would each keep their own total, so the beta runs one.

## Turn limits

`answer()` checks `Session.turns_started` against `DTCC_AGENT_TURNS_PER_CHAT` before the turn and then counts it, so refused and failed turns count too. "New chat" in the page keeps the session, so it keeps the count. The turn then runs inside `asyncio.timeout(DTCC_AGENT_TURN_SECONDS)`. Only that timeout's own expiry is reported as the deadline: a `TimeoutError` from anything else in the turn is an ordinary error. An Engine job a stopped turn started keeps running.

A limit is never retried "fresh". The user gets one `text` frame with the limit's message, and the turn's `answers.jsonl` record has `error: "LimitReached"` or `"TimeoutError"`. See [Provenance and evaluation](provenance-and-evaluation.md).

## Not per Session

The monthly spend limit and the concurrency cap belong to the deployment: one chat can use up the month for everyone. ADR-0004 records this as an exception to per-Session budgets. The ceilings hold only amounts and counts, never another Session's objects, runs or memory. The turn limit is per Session. See [Sessions and isolation](../architecture/sessions-and-isolation.md).

The limits bound only requests this server makes. A Bedrock key used elsewhere is not counted, which is one reason the deployment uses an instance role and an AWS Budgets alert as well.

## Tests

`tests/test_limits.py` (34 tests, FunctionModel, no Bedrock):

- **The ledger:** two turns that together pass the limit aren't both admitted; settling replaces the worst case; a restart, including one in a new month, still counts what was set aside; a request settled after midnight counts in the new month; unreadable or implausible ledgers stop the server; no temporary files are left.
- **The gate:** output is capped only with a spend limit; the Bedrock client doesn't retry with one; concurrent turns near the limit can't pass it; requests over the concurrency limit wait; a 4xx before answering is free, but one after the stream opened, or an error that merely follows a 4xx, keeps the worst case; a request stopped midway keeps its worst case.
- **Turn limits and settings:** the turn limit refuses and survives "new chat"; the deadline stops a turn; another timeout is an error; bad settings, `nan` and `inf` included, stop the server; unset limits are off.
