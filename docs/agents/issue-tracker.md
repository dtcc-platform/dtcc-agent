# Issue tracker

Issues for this repo live in **GitHub Issues** on `dtcc-platform/dtcc-agent`.

Use the `gh` CLI:

- Read: `gh issue list`, `gh issue view <n>`
- Create: `gh issue create --title ... --body ...`
- Label: `gh issue edit <n> --add-label <label>`

Triage labels are listed in `triage-labels.md`.

## Conventions

Title issues after the plan they implement, e.g. `M0/T2 — <what it does>`, so an issue
maps onto `docs/plans/2026-09-19-rebuild-plan.md` without a lookup.

State blocking edges in the body as `Blocked by #N` / `Blocks #N`. The task register is a
graph, not a sequence, and the frontier is whatever has no open blocker.

Use the vocabulary in `CONTEXT.md` for titles and bodies. See `domain.md`.

## Verifying a write landed

`gh issue create --label X` can silently drop the label rather than failing. After creating
or labelling, confirm with:

```sh
gh issue view <n> --json labels --jq '[.labels[].name]'
```

## PRs as a request surface

Off. Pull requests are not part of the triage queue.
