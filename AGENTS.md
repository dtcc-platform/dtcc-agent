# dtcc-agent

Conversational front door to the DTCC platform. See `CONTEXT.md` for the domain glossary
and `docs/adr/` for the decisions that govern it.

## Agent skills

- **Issue tracker** — `docs/agents/issue-tracker.md`. GitHub Issues on
  `dtcc-platform/dtcc-agent`, via `gh`. Read it before filing or editing anything; it
  carries the account and permission rules this repo needs.
- **Domain docs** — `docs/agents/domain.md`. Single context: `CONTEXT.md` plus
  `docs/adr/`. ADRs are the source of truth and supersede `docs/plans/`.
- **Triage labels** — `docs/agents/triage-labels.md`. The five canonical roles.

## Working rules

Python >= 3.12; 3.11 fails to resolve. Set up with:

```sh
uv venv --python 3.12 && uv sync --locked --extra test --extra chatbot
```

Without the `chatbot` extra, `tests/test_chatbot_app.py` fails at collection (no
`fastapi`) and pytest aborts the whole run.

`dtcc-core` is a declared dependency, pinned to a commit in `pyproject.toml`; do not move
the pin by hand (`.github/workflows/dtcc-core-contract.yml` tests a candidate Core first).
A missing Core fails loudly at `import dtcc_agent` rather than serving an empty catalogue.
A Core that imports but fails to register a catalogue section raises `CatalogueError`
(`dtcc_agent/registry.py`): the HTTP server exits at startup naming the section, and stdio
fails the first call that reads the catalogue. Datasets from outside the pinned Core are optional.

<!-- OPENWIKI:START -->

## OpenWiki

This repository has a generated `openwiki/` evidence index. It is optional just-in-time context, not required startup reading.

- Treat source code and tests as authoritative. A brief's unknowns and review items are verification gaps, not automatic requirements.
- Prefer the narrowest quiet validation that proves the changed behavior. Preserve complete failure output.

The wiki is regenerated locally with `/openwiki` (update mode) every few pull requests; there is no scheduled workflow. Do not hand-edit generated OpenWiki pages unless explicitly asked; prefer updating source code/docs and letting OpenWiki regenerate.

<!-- OPENWIKI:END -->
