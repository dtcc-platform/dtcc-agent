---
status: accepted
superseded_reasoning: none — the reasoning below still stands, the status changed
decided: 2026-09-15 by Vasilis, in a 1:1 (see ~/Projects/dtcc/meetings/2026-09-15-notes.md)
---

# The agent is a conversational front door

> **Status changed 2026-09-17, from proposed to accepted.** This ADR was deliberately written as
> a hypothesis awaiting municipality-interview evidence. On the 2026-09-15 call Vasilis answered
> it directly instead: the chatbot is part of the product despite DESIGN.md's silence, and on the
> scenario of people arriving with questions and being served the data directly — "είναι front
> door δηλαδή κανονικά αυτό" (16:14-16:17).
>
> **Decided by authority, not by evidence.** The interview evidence described below still does not
> exist, so the reasoning is unchanged and is now the *rationale* for a decision rather than the
> test that would settle it. Two consequences: the dependent ADRs no longer inherit provisional
> status, and the interview question is repurposed — it now shapes what the front door must be
> good at, and feeds the evaluation scenarios, rather than deciding whether to build it.
>
> Attribution on that call is inferred rather than diarized, and it is not yet written into
> `dtcc-twin/DESIGN.md`, which remains 643 lines that mention no agent or chatbot. Until it lands
> there the platform's own design still implies "not product" by silence.

Three different products hide behind this repo: a conversational front door to the platform, a
scriptable MCP surface for developers, and a research instrument that exists to produce a figure
and a transcript for a paper. Almost every other decision forks on which one it is, so we have
adopted **front door** as the working assumption — the argument being that Atlas and the Table
both require knowing what to click, and this is the only surface where someone can ask in their
own words.

It is recorded as a *hypothesis* deliberately. `dtcc-twin/DESIGN.md` is 643 lines and mentions no
agent, chatbot, assistant, conversational surface or LLM anywhere, so the platform's own design
does not yet agree. The evidence that should settle it is the municipality interviews: if people
arrive with questions rather than with datasets in mind, the hypothesis holds. If they arrive
knowing which dataset they want, the research-instrument reading is correct and much cheaper.

**Update 2026-09-14 — the silence got louder.** On 2026-09-13 the platform migrated its data
model across five repos at once (Core, Sim, Upload, Atlas, Tangible Twin) under dtcc-core#85.
dtcc-agent was not one of them, and no design document in the new Core mentions the agent, the
chatbot or Lurkie. The agent repo was last pushed on 2026-04-16, five months before everything
else moved. Nothing broke — the agent's 112 tests still pass against the new Core — so this is
evidence about standing, not about correctness. It is now a question in the questionnaire (§5a)
rather than an inference here.

**Consequence, as originally written:** every ADR that depends on this one inherits its
provisional status. Do not treat the set as settled architecture until the product question has an
owner and an answer. **That condition is now met** — the product question has both. What remains
is getting it written into DESIGN.md so the answer outlives the recording.

**Second consequence, from the same call.** The agent is not the only thing being built toward
this front door. Vasilis is building "DTCC Engine" — not a repo, part of DTCC Twin — a thin API
layer over dtcc-core and dtcc-sim so downstream never imports Python, with off-host workers,
Celery and a small queue database, explicitly no Kubernetes. His framing in the standup: "then
NAPO and Spiros can connect to it and even the MCP server that we might need plus the chatbot can
actually hook up there." On the 1:1 the division was stated as the MCP server being a native
module running *alongside* the Engine rather than through the Twin API. So the agent's path to
Core and Sim is a platform decision already in motion, not one this repo gets to make alone.
