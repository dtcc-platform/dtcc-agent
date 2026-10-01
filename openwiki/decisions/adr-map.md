---
type: decision-index
title: Architecture decisions map
description: A digest of ADR-0001 to ADR-0010, covering what each decides and how far the current code follows it, with the rebuild milestones (M0 to M4) that are meant to land the rest.
tags: [adr, decisions, rebuild, roadmap]
sources:
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-ca6cb4b1a14fd7969dfae3ec
    resource: repo://CHANGELOG.md
  - id: openwiki-source-d82fbc21a9f74516f7bfd0f8
    resource: repo://chatbot/app.py
  - id: openwiki-source-359e586cbb7fe6a899c447e0
    resource: repo://docs/adr/0003-replace-agent-sdk-with-pydantic-ai.md
  - id: openwiki-source-bc18e4e9c7a7ba8901ca76a6
    resource: repo://docs/adr/0005-retrieval-as-a-separate-mcp-server.md
  - id: openwiki-source-78817427c9adde406d601311
    resource: repo://docs/adr/0006-anthropic-shaped-prompt-caching.md
  - id: openwiki-source-6150372e327ba33b9fdace5c
    resource: repo://docs/adr/0008-evaluation-is-one-harness-with-two-layers.md
  - id: openwiki-source-3c0d2c67e2eefb3fce1e123d
    resource: repo://docs/adr/0009-rebuild-on-a-branch-not-a-fresh-repository.md
  - id: openwiki-source-7931b878d950a1ff97af7eb8
    resource: repo://docs/adr/0010-references-are-typed-and-a-run-records-its-object.md
  - id: openwiki-source-e706cdf6ed71c3ed5f88e79f
    resource: repo://docs/agents/domain.md
  - id: openwiki-source-56e73ecb740406a34053d68a
    resource: repo://docs/plans/2026-09-19-rebuild-plan.md
  - id: openwiki-source-d8839a242913c8f59a48c041
    resource: repo://dtcc_agent/refs.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
generated: { by: "claude-code", at: "2026-10-01T20:29:16.810Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-10-01T20:29:16.810Z
---

# Architecture decisions map

`docs/adr/` is the source of truth for decisions and supersedes `docs/plans/` (see `docs/agents/domain.md`). `CONTEXT.md` is the glossary. Where code and glossary disagree, `CONTEXT.md` records the conflict rather than hiding it. All ten ADRs are `status: accepted`.

The rebuild (`docs/plans/2026-09-19-rebuild-plan.md`) lands these decisions in five milestones:

- **M0:** CI and characterisation tests, with zero behaviour change.
- **M1:** transport, state and references. Split into M1a and M1b.
- **M2:** auth, provenance and measurement.
- **M3:** pydantic-ai, caching and Lurkie.
- **M4:** deployment.

`CHANGELOG.md` tracks what has shipped.

| ADR | Decision | State in code |
|---|---|---|
| **0001** Conversational front door | Of the three products hiding in this repo (front door, developer MCP surface, research instrument), the chatbot is the product's front door. Accepted 2026-09-15 by authority (Vasilis), not by interview evidence. It is not yet written into `dtcc-twin/DESIGN.md`. | Governs scope. Every other ADR assumes it. |
| **0002** Converge on dtcc-twin contracts | Adopt Twin's vocabulary and lifecycle (Capability Catalog, Dataset Definition and Realization, provenance) even though Twin has no implementation, because Twin forbids a competing registry. | Glossary only. Descriptors still carry three of Twin's nine fields (rebuild item D9). |
| **0003** Replace Claude Agent SDK with pydantic-ai | Needed for vendor portability: OpenRouter for testing, Bedrock for production. Rejected alternatives: the Claude API Tool Runner, Managed Agents (not on Bedrock), LangChain/LangGraph and the Vercel AI SDK. | **Not landed.** `chatbot/app.py` still uses `ClaudeSDKClient` with a hardcoded model (M3). |
| **0004** The Session is the isolation unit | Objects, Runs, memory and budgets belong to one Session. Public bounds-keyed downloads stay shared, while derived data is session-local. A future user subject owns many Sessions and does not replace them. | **Landed in M1a, auth pending.** Per-Session stores, runs and artifact folders, session-filtered memory, and one memory budget shared across Sessions (T5, T7, T11). Only bounds-keyed downloads are cached, so nothing derived is shared (U2). Session ids stay unauthenticated until T14 (M2); the MCP port is loopback-only meanwhile (U11). See [Sessions and isolation](../architecture/sessions-and-isolation.md). |
| **0005** Retrieval is a separate MCP server | Document and catalogue search becomes a `dtcc-docs` MCP server alongside this one, for separation and independent deployment. The earlier chromadb/protobuf evidence was withdrawn. | **Not landed.** Only conversation memory (ChromaDB) exists. |
| **0006** Anthropic-shaped prompt caching | Design around `cache_control` breakpoints and a stable prefix (tools, then system, then messages), because caching only pays in production, where the provider is known. Answer caching is rejected. | **Not landed** (M3). The system prompt still embeds seven hardcoded operation schemas. |
| **0007** Keep generic dispatch | Keep `registry.py`, `dispatcher.py`, `runner.py` and `serializers.py` rather than becoming a DTCC Engine client. The duplication is accepted because the Engine has no code. | **Holds.** See [Operation catalogue](../concepts/operation-catalogue.md). |
| **0008** One evaluation harness, two layers | A measurement layer (latency, tokens, cost, model, prompt and catalogue revision) that needs no expert, and a correctness layer (the operations, order and parameters expected per Scenario) that needs a domain expert and drops in as an assertion pass. | **Not landed** (M2). Provenance is its prerequisite. |
| **0009** Rebuild on a branch | Feature branches off `develop`, merged by PR every milestone. This reverses a 2026-09-17 "fresh repository" reading of the same recording. | **Holds.** It is the working process. |
| **0010** Typed references; a Run records its Object | References carry their kind (`obj_…`, `run_…`) so a misrouted id fails loudly, and a Run records the Object reference it yielded. | **Landed (M1b/T9, #57).** `dtcc_agent/refs.py` mints `obj_…` and `run_…`; object tools, `get_run_summary` and `run_operation` refuse the wrong kind; tools take `object_ref` and `run_ref`. U6 decided 2026-09-30: the Object owns a Run's result and the Run keeps only its reference. See [Dispatch, object references and serialization](../concepts/dispatch-and-object-store.md). |

## Delivered so far (per `CHANGELOG.md`, 2026-10-01)

- **M0:** dtcc-core pinned, CI running, characterisation tests (#7, #8).
- **M1a, every task merged** (T4, T5, T6, T7, T8, T10, T11, T13). Tools off the event loop (#32); Session isolation over HTTP (#33); a bounded worker pool with a fair per-Session share and one download per tile (#36); the catalogue built once per process (#43); builder caching off, which made the cache split (T6) moot (#62); per-Session artifact folders, a download route and path refusal (#63); one memory budget counted from real sizes (#66); two services with the MCP server on loopback (#67). The M1a QA checklist remains before the epic (#16) closes.
- **M1b:** typed references with a Run linked to its Object (T9, #57) and a cache version stamp (T12, #56).
- **Decisions settled:** U1 (refuse paths), U2 (builder caching off), U4 (accurate accounting), U6 (the Object owns a Run's result), U9 (fix rendering, with matplotlib), U10 (the pin wins, asserted in the image, #61), U11 (loopback until T14).
- **Core pin** at `bb95f2f` with the three upstream fixes (#59).

## Deferred decisions

The rebuild plan keeps a register, D1 to D9, of questions deliberately left open. Examples:

- whether Lurkie renders geometry and with which engine (D1, D2);
- the UI stack (D3);
- the identity provider (D4);
- a move from 22 generic tools to Task tools (D5);
- whether the agent ever becomes an Engine client (D6);
- the nine-versus-three descriptor fields (D9).

Read the plan before treating any of these as settled.
