---
status: accepted
decided: 2026-09-19 by Spiros
supersedes: the 2026-09-17 fresh-repository decision recorded in docs/plans/2026-09-14-restructure-design.md:66 and questionnaire 5a-bis
---

# The rebuild is a branch off `develop`, not a fresh repository

The rebuild happens on feature branches off `develop` in `dtcc-platform/dtcc-agent`, merged back
through pull requests at every milestone. `develop` remains trunk throughout. Nothing is
archived, nothing is orphaned, and `develop` is never "replaced".

This reverses a decision recorded on 2026-09-17. Both readings come from one sentence at 16:48 in
the 2026-09-15 recording — *"Αν πάμε σε Clean Branch Implementation και τα λοιπά είναι ένα βήμα"* —
read once as "fresh repository" and once as "clean branch". It is two readings of one recording,
not one superseding the other by age, and it is recorded as a reversal rather than quietly
corrected.

## Why the reasons moved

- **The protobuf finding is withdrawn.** The claim that chromadb and dtcc-core cannot share an
  environment does not reproduce and was withdrawn publicly on `dtcc-agent#1`. Environment
  separation was a significant part of the fresh-repo case.
- **The ten follow-up issues are cancelled**, absorbed into the rebuild, so there is no backlog
  that a new repository would leave behind.
- **`develop` carries a green 122-test suite**, which is the only regression baseline the rebuild
  has. A fresh repository has nothing to regress against (ADR-0008).

## What this actually means, because "fresh start" was misleading

Combined with ADR-0007, roughly half the code survives untouched. The rebuild is not a retype.

| | Lines | Fate |
|---|---|---|
| `serializers.py`, `registry.py`, `dispatcher.py`, `analysis.py`, `crop.py`, `geocode.py`, `disk_cache.py`, `object_store.py`, `geojson_store.py`, `renderer.py` | ~1,900 | **Kept.** ADR-0007 keeps generic dispatch; rewriting these would undo it. |
| `server.py` | 1,190 | **Replaced.** 22 tool wrappers over module globals, zero direct test coverage. |
| `chatbot/` | 537 | **Replaced.** The Agent SDK wrapper (ADR-0003). |
| — | — | **New.** Session scoping, auth, provenance, tests for the tool surface. |

"Fresh start" meant *stop incrementally patching the old design* — new structure, new entry
points, new state model. It never meant rewriting code nobody has a complaint about.

## Consequences

- **Merge at every milestone, not once at the end.** This is the entire practical benefit over a
  fresh repository and it only pays off if it actually happens. A milestone that cannot merge was
  scoped wrong.
- **The old and new surfaces can run side by side** during cutover, which is what lets the harness
  compare them (ADR-0008).
- **The risk is parallel trees that never converge.** The milestone boundaries are the forcing
  function.
- **Several open questions simply cease to exist**: the new repository's name and owner, whether
  Lurkie moves across, whether the old public repository gets archived, and with it the argument
  that archiving is a security decision because the old history retains the findings.
  `dtcc-agent#1` keeps its assignment.
