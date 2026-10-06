---
status: accepted
---

# Replace the Claude Agent SDK with pydantic-ai

The chatbot runs on the Claude Agent SDK — Claude Code packaged as a library. It cannot front
non-Anthropic models, and `chatbot/app.py` currently hardcodes a superseded model with no
configuration at all. Vendor portability was asked for directly: OpenRouter for testing, Bedrock
for production. That is not reachable from this SDK, so the harness is being replaced with
**pydantic-ai**, whose provider list carries `openrouter`, `bedrock` and `bedrock-mantle` as
first-class entries, making the testing-to-production move a config change on one code path.

## Considered options

- **Claude API + Tool Runner** — loops over tools you define, but remains Anthropic-only, so it
  does not satisfy the requirement that prompted the change.
- **Managed Agents** — would delete the most of our own loop code, but is unavailable on Bedrock,
  which is the stated production target.
- **LangChain / LangGraph** — trades an Anthropic-shaped dependency for a framework-shaped one,
  and buys little when the tool protocol is already MCP.
- **Vercel AI SDK** — TypeScript; the backend is Python FastAPI.

## Consequences

The Agent SDK is also what ships the built-in Bash, Read, Write, Edit and WebFetch tools — the
exact surface the chatbot's remote-code-execution finding concerns. Leaving it removes that
attack surface by construction rather than by configuration, which is a second reason to do it.

What is genuinely lost is context management, which must be rebuilt. And a precise trap to avoid:
pydantic-ai's *native* `MCPServerTool`, where the provider connects to a remote MCP URL, is **not
supported on Bedrock**. It does not affect us — `dtcc-agent` is a stdio server, so the client-side
path is correct and is provider-agnostic — but reaching for the native path later would fail in
production only.

pydantic-ai moves quickly; pin it.

**Update 2026-09-14.** Core `develop` now takes `pydantic>=2.12.5` as a runtime dependency of its
own, after the dtcc-core#85 migration. Core, the chatbot extras and pydantic-ai's requirements
co-resolve cleanly on pydantic 2.13.5, verified in a clean venv. This change got slightly cheaper,
not more expensive: pydantic is no longer something we introduce to the platform.

**Accepted 2026-09-19.** "Standalone application" was settled as covering all four of process,
model independence, UI and deployment — and model independence is this ADR. It is therefore no
longer optional or deferred; it is the substance of rebuild milestone 3.

**Sequenced third, deliberately.** Milestone 1 is transport, session-scoped state and tests;
milestone 2 is auth and provenance; this lands in milestone 3. The reason is measurement: this is
the change most likely to produce a long debugging tail, and the evaluation harness (ADR-0008)
should be reading a stable system before the model runtime is swapped underneath it. Sequencing it
third is what gives the migration a genuine before-and-after number on latency and cost, which is
the comparison the platform asked for.

**A latency finding that belongs here.** `chatbot/app.py:228` constructs a new `ClaudeSDKClient`
per user message — process spawn plus MCP handshake, every turn. That is almost certainly the
largest single latency item in the system today, and it disappears by construction when the
subprocess does. This ADR is a performance change as much as a portability one.

**Update 2026-10-06 (M3, #29): done, on Bedrock only.** Bedrock is the only provider. OpenRouter
for testing and Anthropic direct were dropped when M3 was specced (2026-10-04, #29), replacing
the plan's "a second provider runs the same question set": portability, the requirement that
started this ADR, is shown by switching models on one code path. It is model choice on
Bedrock: `DTCC_AGENT_MODEL` names the model, and switching is that variable alone (Sonnet 5.5 to Sonnet 5, no code change, `eval/runs/m3-sonnet-5.md`). Every 4.x
model is refused on the company account until Anthropic's use-case form is filed.

The runtime result, same 10 questions, model and machine, two runs pooled per side
(`eval/runs/baseline-m3-sdk.md` against `eval/runs/m3-runtime.md`, #93): the median of
per-question warm medians went from 12.1 s to 8.9 s, no question got more than 3% slower,
and a run of the set costs $0.73 instead of $0.95. The latency finding below held: the
largest gains were on short questions, where the CLI spawn was most of the time.

The Agent SDK stays behind `DTCC_AGENT_RUNTIME=sdk` for one milestone as the rollback, checked
in Docker on Bedrock (#88). pydantic-ai is pinned at 2.54.0 and needed the MCP server on mcp
2.x first (#92). Context management, named above as the real loss, is still the conversation's
own message history; nothing summarises or trims it yet.
