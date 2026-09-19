---
status: accepted
---

# Retrieval lives in a separate dtcc-docs MCP server, not as a tool in dtcc-agent

Document and catalogue search will be a second MCP server rather than another tool inside this
one. The assistant sees tools from both servers in one flat list and decides between them; from
the user's perspective it is one assistant.

The reasoning is separation of concerns and independent deployment.

**Evidence withdrawn, 2026-09-14.** An earlier revision of this ADR claimed harder evidence: that
chromadb ships gencode requiring protobuf ≥6.33.5 against Core's `<6.0.0` ceiling, so a vector
store and the scientific stack could not share an environment. **That does not reproduce.** Tested
combinations, each built from source in a clean venv: Core `5cf56fa` and Core `develop` `464c58d`,
× chromadb installed before and after Core, × protobuf 5.29.6 and 7.36.1 forced. `import
dtcc_core` and `import chatbot.app` succeed in all of them, and chromadb 1.5.9 completes an
add-and-query round trip with the ONNX embedder alongside Core. chromadb 1.5.9 ships no `_pb2`
modules of its own; it touches protobuf only through opentelemetry-proto. The original failure was
real when observed but I cannot name the install path that produced it, so it cannot carry an
architectural decision.

**Re-verified 2026-09-18** against Core `develop` `18eb176` (`0.9.8.dev0`) in a clean 3.11 venv:
resolves to chromadb 1.5.9 + protobuf 5.29.6, both imports succeed, chroma round trip completes
alongside Core. The claim was withdrawn publicly the same day on `dtcc-agent#1`.

**Replacement evidence, weaker but durable.** Core `develop` now takes three exact runtime pins —
`linkml==1.11.1`, `linkml-runtime==1.11.1`, `h5py==3.16.0` — plus `polyforge==0.1.0a9` and a git
pin on dtcc-mesher. Exact `==` pins fail hard against any co-installed package wanting a different
version, and Core's dependency surface grew by nine packages in one commit. Everything resolves
today. The point is that it is Core's call, not ours, and a separate server does not have to care.

**Update 2026-09-15 — where the servers sit is now a platform decision.** Vasilis is building
"DTCC Engine", a thin API layer over dtcc-core and dtcc-sim so downstream never imports Python,
and on the 1:1 the division was stated as the MCP server being a **native module running
alongside the Engine**, not through the Twin API (14:16). In the standup he put it as "even the
MCP server that we might need plus the chatbot can actually hook up there." This ADR's
conclusion survives — a second server for retrieval is still the right shape, and the
environment-separation argument above applies to it identically — but the topology is no longer
this repo's to choose unilaterally. Confirm the split with Vasilis before implementing, and
expect `dtcc-docs` to sit beside the Engine the same way this server does.

**First corpus is the Capability Catalog**, not municipal plans. The descriptors are DTCC's own
machine-readable output with no third-party terms, they directly answer "what can you tell me
about this neighbourhood" without the asker naming an Operation, and they let the whole pipeline
be built before anyone has to read a licence. Citations are mandatory from the first commit;
retrofitting them is expensive and they are what makes an answer checkable.

**Accepted 2026-09-19, and the split is now precise.** `CONTEXT.md` already distinguishes the two:
**Corpus** is a body of material that retrieval searches; conversation memory is what was said
before. "It cannot look things up" and "it does not remember me" are different complaints.

`chatbot/memory.py` conflates them today — one ChromaDB collection serving both, and the file with
the missing `session_id` filter. The rebuild separates them along the line this ADR draws:
**conversation memory stays with the chat service and becomes session-scoped** (ADR-0004);
**corpus retrieval moves to the separate `dtcc-docs` server.**

That makes the separation a correctness fix as well as an architectural one, which is a stronger
reason than the original separation-of-concerns argument.
