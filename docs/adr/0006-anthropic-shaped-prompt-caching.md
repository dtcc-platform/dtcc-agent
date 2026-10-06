---
status: accepted
---

# Prompt caching targets Anthropic's shape, despite choosing vendor portability

Having just decided to leave an Anthropic-specific harness for a provider-agnostic one
(ADR-0003), we are nevertheless designing prompt caching around Anthropic's explicit
`cache_control` breakpoints and prefix-match invalidation rather than a lowest-common-denominator
approach that would work identically everywhere.

This looks contradictory and is not. Portability exists so OpenRouter can be used for *testing*
and Bedrock for *production*; cache economics only matter in production, where the provider is
known. Designing for the lowest common denominator would forfeit the largest cost and latency win
on the only provider that serves real traffic, in exchange for symmetry during test runs.

**Consequence:** keep the prompt prefix stable and ordered — tools, then system, then messages —
with volatile content after the last breakpoint. Caching is then present where it pays and merely
absent in testing. What is cached is the instruction-and-catalogue prefix, which is identical on
every request by construction; caching *answers* was considered and rejected earlier, correctly,
on the grounds that the same question rarely recurs. These are different mechanisms with
different economics and should not be conflated again.

**Accepted 2026-09-19, and there is already a hand-rolled version of it in the code.**

`chatbot/config.py:19-22` instructs the model: "Use the operation schemas below directly — do NOT
call `describe_operation()` for these common operations" — followed by seven operation schemas
pasted into the system prompt (`:36-57`). Somebody measured the round trips and hard-coded a fix
for the seven most frequent. That is a manual prompt cache with no invalidation and no version.

Implementing this ADR properly **deletes that hack**: with the catalogue inside a stable cached
prefix, there is no reason to special-case seven operations. It also removes a latent
inconsistency — `disk_cache.CACHE_ALLOWLIST` holds those same seven plus `get_buildings`, so two
packages independently encode "the operations that matter" with nothing linking them. One should
derive from the other, or both from measurement.

*Update 2026-09-29 (#39):* `get_buildings` left `CACHE_ALLOWLIST`; it now caches under
`datasets.buildings`. The allowlist holds only the seven Core operations.

**Why this matters more than it looks.** The model never sees 133 tools; it sees 22, three of
which are the dispatch tools (`list_operations`, `describe_operation`, `run_operation`). Discovery
is therefore round trips before any real work starts, and the catalogue is the thing being re-sent.
Putting it in the cached prefix is the single largest structural win available on the prompt side.

*Update 2026-10-06 (#87): the catalogue in the cached prefix was measured and not adopted; the
seven pasted schemas stay.* Three variants of the catalogue as a static instruction before the
system cache point, each against the M3 latency gate (Sonnet 5.5 on Bedrock, two runs pooled,
`eval/runs/m3-catalogue-{a,b,c}.md`), with a same-day control of the unchanged code
(`eval/runs/m3-control.md`, run interleaved with C):

| | Prefix added | Median of warm medians | Worst question vs control |
|---|---|---|---|
| Control (seven pasted schemas) | | 10.5 s | |
| A: `list_operations` | 8.8k tokens | 11.2 s | q03 +80% |
| B: every `describe_operation` | 57.7k tokens | 13.3 s | q04 +135% |
| C: A plus the seven schemas | 8.8k tokens + 7 schemas | 13.5 s | q04 +269% |

What the numbers say:

- **The pasted schemas do real work.** Without them (A) the model looks the common operations
  up again: q03 went from 0 to a median of 4.5 `describe_operation` calls.
- **A large prefix is not free on Bedrock.** B makes no lookups and the same number of requests,
  yet every request reads ~70k cached tokens and answers slower.
- **The catalogue changes what the model does.** Shown all 133 operations, it renders q04's
  buildings by downloading `datasets.city` and building a city surface mesh (5 of 6 runs, 30-62 s)
  where it otherwise downloads `datasets.buildings` (6 of 6, 10-12 s). The harness measures and
  does not score (ADR-0008), so whether that is a better answer is open.
- **Bedrock's speed drifts by day.** The unchanged code ran 18% slower than the day before
  (`m3-runtime.md`), so a gate against a reference measured on another day partly measures the day.

**So the hand-rolled cache stays, as a measured choice.** Its cost is the one this ADR named: the
seven schemas are a copy with no link to the catalogue. Revisit when a dtcc-core upgrade changes
any of those seven operations, or when answers can be scored, which would say whether C's heavier
choices are worth their time.

Two changes from #87 stand on their own and shipped: memory context is now a dynamic instruction
after the system cache point (as a plain string it was inside the cached prefix, so each
conversation's memory re-wrote the prefix), and each model request's Bedrock usage is logged at
debug. The catalogue measured 8,759 tokens for the summary and 57,733 for every schema, against
#87's estimates of 5.8k and 36k (characters ÷ 4).
