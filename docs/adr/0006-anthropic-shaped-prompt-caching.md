---
status: proposed
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
