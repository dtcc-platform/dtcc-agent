# Domain docs

Single-context repo. One context, rooted here.

- **`CONTEXT.md`** (repo root) — the domain glossary. Use its vocabulary in issue titles,
  ticket bodies, commit messages and code.
- **`docs/adr/`** — architecture decision records, `0001` through `0010`, all `accepted`.

## Consumer rules

Read `CONTEXT.md` before naming anything. Read the ADRs covering the area you are touching
before proposing a design; they are the source of truth and they supersede the planning
documents under `docs/plans/`, which are retained as history.

`docs/plans/2026-09-19-rebuild-plan.md` is the live implementation plan: five milestones
(M0-M4) and a fourteen-task register. `docs/plans/2026-09-14-restructure-design.md` is
superseded and annotated as such.

If an ADR and the code disagree, that is a finding worth raising, not a detail to smooth over.
