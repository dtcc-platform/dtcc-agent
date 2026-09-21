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
uv venv --python 3.12
uv pip install -e ../dtcc-core -e . pytest pytest-asyncio fastapi httpx
```

Without `fastapi` and `httpx`, `tests/test_chatbot_app.py` fails at collection and pytest
aborts the whole run.

`dtcc-core` is not yet a declared dependency and `registry.py` swallows the `ImportError`,
so a fresh install starts cleanly and serves an empty catalogue. M0 fixes this.
