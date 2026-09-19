---
status: accepted
---

# Converge on dtcc-twin's contracts even though dtcc-twin has no implementation

`dtcc-twin` is a design repository — created 2026-08-18, no primary language, contents are
`DESIGN.md` and docs. Its Capability Catalog, provenance labelling and long-running-work
lifecycle are specified but not built. We are nevertheless adopting its vocabulary and its
lifecycle rather than inventing our own.

The alternative — stay parallel until Twin ships — was rejected because DESIGN.md explicitly
forbids a competing definition registry ("one semantic authority"), and this repo already
reflects over the Core registry that the Capability Catalog is itself a view of. We are one
adapter from convergence and a full rewrite from divergence.

**Consequence:** we implement contracts against a specification with no reference implementation,
so we will discover its gaps first. That is accepted, and arguably the point — this repo is the
only surface currently running long jobs in front of a person.

**Update 2026-09-14 — there is now a partial reference implementation, in a different repo.**
`dtcc-twin` still has no code. But `dtcc-tangible-twin` adopted Core's new native model on
2026-09-13 as part of the dtcc-core#85 migration, and its obligations as a consumer are written
down in Core's `docs/design/model-downstream-adoption.md`: it requests `dtcc`, verifies artifact
integrity before selecting previews, resolves a native summary through the manifest, and
explicitly does *not* decode or render arbitrary native models. That is a real contract to
converge on rather than a specification to guess at. It does not settle the vocabulary question
this ADR is about, because Tangible Twin is a catalogue and preview surface, not the Capability
Catalog. Read it as narrowing the gap, not closing it.

**Accepted 2026-09-19, with the gap measured rather than assumed.** The DTCC Engine — the piece
of Twin that would make the Capability Catalog real — **does not exist as code**. `dtcc-twin` at
`develop` is `DESIGN.md` (643 lines), `docs/dtcc-engine-backend-design-v1.md` (501 lines), two
`AGENTS.md`, a README, and an empty `dtcc-engine/` directory. No `.py`, no PRs ever opened, last
push 2026-09-06. The spec says so itself at `:466`: "No Engine implementation or runtime
validation was performed as part of writing it."

That strengthens this ADR rather than weakening it. Converging on a vocabulary costs nothing and
is reversible; the thing it protects against — two competing registries with different words for
the same concept — gets more expensive every month. So we adopt Twin's terms (`CONTEXT.md` already
does, marking them **(Twin)**) while keeping our own implementation (ADR-0007).

**The measured gap: nine fields against three.** `DESIGN.md:344-346` specifies that each descriptor
supplies identity, title, explanation, parameter schema, semantic result type, coverage, required
inputs, execution requirements and output semantics. `registry.py` yields name, category and
description. Closing that gap is real convergence work and belongs in the rebuild, sequenced with
catalogue-as-artifact (redesign item 9).
