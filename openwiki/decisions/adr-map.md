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
  - id: openwiki-source-5f2b4e0b591e0b8e093c4c16
    resource: repo://chatbot/runtime/__init__.py
  - id: openwiki-source-6ed7b95a502f0a6d7126df1d
    resource: repo://chatbot/runtime/pai.py
  - id: openwiki-source-359e586cbb7fe6a899c447e0
    resource: repo://docs/adr/0003-replace-agent-sdk-with-pydantic-ai.md
  - id: openwiki-source-bc18e4e9c7a7ba8901ca76a6
    resource: repo://docs/adr/0005-retrieval-as-a-separate-mcp-server.md
  - id: openwiki-source-78817427c9adde406d601311
    resource: repo://docs/adr/0006-anthropic-shaped-prompt-caching.md
  - id: openwiki-source-69befafd44aed5a2ceb6450c
    resource: repo://docs/adr/0007-agent-keeps-its-own-generic-dispatch.md
  - id: openwiki-source-6150372e327ba33b9fdace5c
    resource: repo://docs/adr/0008-evaluation-is-one-harness-with-two-layers.md
  - id: openwiki-source-3c0d2c67e2eefb3fce1e123d
    resource: repo://docs/adr/0009-rebuild-on-a-branch-not-a-fresh-repository.md
  - id: openwiki-source-7931b878d950a1ff97af7eb8
    resource: repo://docs/adr/0010-references-are-typed-and-a-run-records-its-object.md
  - id: openwiki-source-5362e3b27c4a029e63907699
    resource: repo://docs/adr/0011-simulations-on-the-engine-downloads-stay-in-process.md
  - id: openwiki-source-e706cdf6ed71c3ed5f88e79f
    resource: repo://docs/agents/domain.md
  - id: openwiki-source-56e73ecb740406a34053d68a
    resource: repo://docs/plans/2026-09-19-rebuild-plan.md
  - id: openwiki-source-d8839a242913c8f59a48c041
    resource: repo://dtcc_agent/refs.py
  - id: openwiki-source-10801051a0be31ef9b711d8f
    resource: repo://dtcc_agent/server.py
generated: { by: "claude-code", at: "2026-10-31T10:29:51.442Z" }
verified:
  - by: openwiki/0.5.2
    at: 2026-10-31T10:29:51.442Z
---

# Architecture decisions map

`docs/adr/` is the source of truth for decisions and supersedes `docs/plans/` (see `docs/agents/domain.md`). `CONTEXT.md` is the glossary. Where code and glossary disagree, `CONTEXT.md` records the conflict rather than hiding it. There are eleven ADRs: ten are `status: accepted`, and ADR-0007 is `superseded` by ADR-0011.

The rebuild (`docs/plans/2026-09-19-rebuild-plan.md`) lands these decisions in five milestones:

- **M0:** CI and characterisation tests, with zero behaviour change.
- **M1:** transport, state and references. Split into M1a and M1b.
- **M2:** auth, provenance and measurement.
- **M3:** pydantic-ai, caching and Lurkie.
- **M4:** re-scoped 2026-10-13 (#30) to "the agent on the DTCC Engine, ready to deploy"; done, and the epic closed on 2026-10-30. The deployment itself is designed by the platform team in dtcc-infra, whose step A2 asked the agent for five changes, merged 2026-10-31.

`CHANGELOG.md` tracks what has shipped.

| ADR | Decision | State in code |
|---|---|---|
| **0001** Conversational front door | Of the three products hiding in this repo (front door, developer MCP surface, research instrument), the chatbot is the product's front door. Accepted 2026-09-15 by authority (Vasilis), not by interview evidence. It is not yet written into `dtcc-twin/DESIGN.md`. | Governs scope. Every other ADR assumes it. |
| **0002** Converge on dtcc-twin contracts | Adopt Twin's vocabulary and lifecycle (Capability Catalog, Dataset Definition and Realization, provenance) even though Twin has no implementation, because Twin forbids a competing registry. | Glossary only. Descriptors still carry three of Twin's nine fields (rebuild item D9). |
| **0003** Replace Claude Agent SDK with pydantic-ai | Originally for vendor portability (OpenRouter for testing, Bedrock for production). Rejected alternatives: the Claude API Tool Runner, Managed Agents (not on Bedrock), LangChain/LangGraph and the Vercel AI SDK. Updated 2026-10-06: Bedrock is the only provider; portability is shown by switching models on one code path. | **Landed (M3, #93, #97); rollback removed (#105).** `chatbot/runtime/pai.py` runs the agent loop in the chatbot process on Bedrock, with the model from `DTCC_AGENT_MODEL`; the Agent SDK runtime and its `sdk` extra are gone. See [Model runtime](../architecture/model-runtime.md). |
| **0004** The Session is the isolation unit | Objects, Runs, memory and budgets belong to one Session. Public bounds-keyed downloads stay shared, while derived data is session-local. A future user subject owns many Sessions and does not replace them. | **Landed (M1a); admission and subject in M2 (T14, #76); idle expiry (U7, #75).** Per-Session stores, runs and artifact folders, session-filtered memory, one shared memory budget, an access code for chats and a bearer secret for MCP calls. The subject is `"anonymous"` until central auth. An exception added in #132: the monthly spend limit and the model-concurrency cap are deployment-wide ceilings, not Session budgets; they hold only amounts and counts. See [Sessions and isolation](../architecture/sessions-and-isolation.md). |
| **0005** Retrieval is a separate MCP server | Document and catalogue search becomes a `dtcc-docs` MCP server alongside this one, for separation and independent deployment. The earlier chromadb/protobuf evidence was withdrawn. | **Not landed.** Only conversation memory (ChromaDB) exists. |
| **0006** Anthropic-shaped prompt caching | Design around cache breakpoints and a stable prefix (tools, then system, then messages), because caching only pays in production, where the provider is known. Answer caching is rejected. | **Landed (M3).** Bedrock caching of tool definitions, instructions and messages, with memory as a dynamic instruction after the cache point. Putting the full catalogue in the cached prefix was measured and not adopted (#95, #87). |
| **0007** Keep generic dispatch | Keep `registry.py`, `dispatcher.py`, `runner.py` and `serializers.py` rather than becoming a DTCC Engine client. The duplication is accepted because the Engine had no code. | **Superseded by ADR-0011 (2026-10-20).** Its "no Engine client" half no longer holds; its "keep our own dispatch" half is kept by ADR-0011. |
| **0008** One evaluation harness, two layers | A measurement layer (latency, tokens, cost, model, prompt and catalogue revision) that needs no expert, and a correctness layer (the operations, order and parameters expected per Scenario) that needs a domain expert and drops in as an assertion pass. Updated 2026-10-06: a speed comparison needs a same-day control. | **Measurement layer landed (M2, #77, #78).** `eval/measure.py` and provenance records; baselines in `eval/runs/`. The correctness layer is not built. See [Provenance and evaluation](../operations/provenance-and-evaluation.md). |
| **0009** Rebuild on a branch | Feature branches off `develop`, merged by PR every milestone. This reverses a 2026-09-17 "fresh repository" reading of the same recording. | **Holds.** It is the working process. |
| **0010** Typed references; a Run records its Object | References carry their kind (`obj_…`, `run_…`) so a misrouted id fails loudly, and a Run records the Object reference it yielded. | **Landed (M1b/T9, #57).** `dtcc_agent/refs.py` mints `obj_…` and `run_…`; object tools, `get_run_summary` and `run_operation` refuse the wrong kind; tools take `object_ref` and `run_ref`. U6 decided 2026-09-30: the Object owns a Run's result and the Run keeps only its reference. See [Dispatch, object references and serialization](../concepts/dispatch-and-object-store.md). |
| **0011** Simulations on the Engine, downloads in-process | The agent is a DTCC Engine client for simulations. Building and point-cloud downloads, and everything built from them, stay in-process behind the disk cache, and the agent keeps its own discovery and dispatch. Decided from T41's same-day measurement: point-cloud answers 30–38% slower through the Engine, against a 15% limit. | **Landed (M4, #113, #127, #128).** Simulations run as Engine jobs when `DTCC_ENGINE_URL` is set. An experimental `DTCC_ENGINE_CORE=1` switch stays so buildings can be measured once Core can return them (dtcc-core#160). In deployment the agent will reach the Engine through the Twin backend (dtcc-infra L1). See [DTCC Engine integration](../integrations/dtcc-engine.md). |

## Delivered so far (per `CHANGELOG.md`, 2026-10-31)

- **M0:** dtcc-core pinned, CI running, characterisation tests (#7, #8).
- **M1a and M1b:** tools off the event loop, Session isolation over HTTP, a bounded worker pool, the catalogue built once per process, builder caching off, artifact folders and path refusal, accurate memory accounting, the two-service deployment, typed references and the cache version stamp (#32–#69).
- **M2:** idle session expiry (#75), admission: access code, MCP secret and subject (#76), provenance on every answer (#77), the measurement harness and the M2 baseline (#78, #84), refused oversized downloads and chat recovery after a crash (#82), a smaller prompt with only DTCC tools (#83).
- **M3:** the chat on Bedrock with the model from config (#90), mcp 2.x (#92), pydantic-ai in process, 27% faster and 23% cheaper than M3's starting point (#93, #97), the catalogue-in-prefix experiment (#95), memory off the event loop (#98), refusals shown as refusals (#99), the same-day control rule (#100).
- **M4 (done):** heatwave answers with temperature statistics (#102), statistics-only answers (#104), the SDK runtime removed (#105), clear Bedrock key errors (#106), an exact building crop with Core `source_bounds` (#107, Core pin `a78df16`), simulations on the DTCC Engine (#113), `docs/deployment.md` (#122), QA fixes (#123, #124, #125), the Engine measurement and ADR-0011 (#127, #128).
- **Deployment (dtcc-infra A2):** no `.env` in the image (#135), cancelled Engine jobs (#136), Bedrock through an instance role (#137), serving under `/agent/` (#138), turn and spend limits (#139).
- **Decisions settled:** U1 (refuse paths), U2 (builder caching off), U4 (accurate accounting), U6 (the Object owns a Run's result), U7 (idle expiry), U9 (fix rendering, with matplotlib), U10 (the pin wins, asserted in the image), U11 (loopback until T14).

## Deferred decisions

The rebuild plan keeps a register, D1 to D9, of questions deliberately left open. Examples:

- whether Lurkie renders geometry and with which engine (D1, D2);
- the UI stack (D3);
- the identity provider (D4);
- a move from 22 generic tools to Task tools (D5);
- whether the agent becomes an Engine client (D6): answered by ADR-0011, simulations yes, downloads and dispatch no;
- the nine-versus-three descriptor fields (D9).

Read the plan before treating any of these as settled.
