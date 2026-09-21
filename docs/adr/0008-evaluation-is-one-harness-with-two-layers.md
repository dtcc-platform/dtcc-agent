---
status: accepted
decided: 2026-09-19 by Spiros
---

# Evaluation is one harness with two layers, and only one of them needs a domain expert

Two different things were collapsed into a single item called "the evaluation suite", and that
item then blocked everything behind it across two planning cycles. They are separated here.

**The measurement layer.** A fixed set of questions run repeatably through the system, recording
latency, token counts, cost per task, which model, which provider, which prompt version, which
catalogue revision. It answers *what did this cost and how long did it take*. It needs no domain
expertise — only questions and instrumentation.

**The correctness layer.** For a given question, the operations the assistant should have chosen,
in what order, with what parameters, and what a wrong answer looks like. It answers *did it do the
right thing*. This is a domain judgement and needs Anders or Nuri.

They share all their plumbing: both require running fixed inputs through the system and recording
what happened. So it is one harness. The correctness layer drops in later as an assertion pass
over runs the measurement layer is already producing.

## Consequences

- **The measurement layer starts immediately** and is not blocked on anyone. It is also the thing
  the platform actually asked for early — the ability to measure and compare time, cost and speed
  while building, so decisions are made on numbers rather than impressions.
- **Provenance is a prerequisite, not a nice-to-have.** `CONTEXT.md` defines Provenance; nothing in
  the code writes one. A benchmark result that cannot be attributed to a model, a prompt version
  and a catalogue revision is not a measurement. This moves provenance into the rebuild's early
  milestones (redesign item 6).
- **The answer key is sent as a draft to review, never as a request to author.** Ten to fifteen
  scenarios written from the meeting transcripts, the README examples, and the usage evidence
  already in the code — `geocode.py:22-39`'s fifteen hand-verified Gothenburg bounding boxes,
  `config.py:36-57`'s seven hardcoded schemas, `disk_cache.py:28-37`'s eight cached operations.
  Those are three independent recorded judgements about what gets asked. Authoring is homework and
  has been declined by silence twice; correcting a draft with two deliberately wrong entries in it
  takes fifteen minutes.
- **Architecture work is not gated on either layer.** Transport, session scoping, auth, provenance
  and tests are correct or incorrect on their own terms. Only tool-surface changes need scenarios,
  and the tool surface is frozen until scenarios exist (ADR-0007 keeps the dispatch shape; the 22
  MCP tools stay as they are until measurement says otherwise).
- **"Regression" needs care.** The rebuild replaces `server.py` and `chatbot/`; the 122 tests on
  `develop` are the baseline for the ~1,900 lines being kept, not for the parts being replaced.
  For those, the harness measures the new thing against the old one running side by side, which is
  possible only because the rebuild is a branch rather than a fresh repository (ADR-0009).

## What was rejected

**Hard-gating the rebuild on the answer key.** It had been the stated blocker across two
checkpoints without arriving, which is what a dependency looks like when nobody has accepted it.

**Dropping evaluation from the critical path.** Rejected for the reason the restructure design
already gave: replacing behaviour in a stochastic system with no way to measure regression means
comparing demos by eye.
