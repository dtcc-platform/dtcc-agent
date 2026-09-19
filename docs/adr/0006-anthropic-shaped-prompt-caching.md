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

**Why this matters more than it looks.** The model never sees 133 tools; it sees 22, three of
which are the dispatch tools (`list_operations`, `describe_operation`, `run_operation`). Discovery
is therefore round trips before any real work starts, and the catalogue is the thing being re-sent.
Putting it in the cached prefix is the single largest structural win available on the prompt side.
