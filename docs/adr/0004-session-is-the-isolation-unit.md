---
status: accepted
---

# The Session is the isolation unit

Objects, Runs, conversation memory and budgets all belong to exactly one Session and are never
visible from another. The alternative — a shared store with session-scoped views — was rejected
because it makes the isolation boundary a property of each query rather than of the object, and
therefore something a later feature can quietly break.

This also closes a live defect. Conversation memory currently queries without a session filter
and injects the result under a header claiming it is "past conversations with this user", so with
more than one user it is cross-user retrieval — into the system prompt of an agent that has a
shell. Making the Session the boundary in both subsystems means the two cannot drift apart.

**Consequence:** per-Session budgets come before per-user budgets, because there is no user
identity until authentication exists.

**Update 2026-09-18 — two corrections, netting out to "unchanged".**

On 2026-09-15 Vasilis said authentication is solved and will be "κεντρικό authentication για όλα
τα services" (14:57-15:15). I first wrote here that per-user identity was therefore arriving soon
and per-user budgets were a near-term extension. I then corrected that in the other direction, by
citing the Engine design's "single shared configurable token" and its deferral of per-consumer
tokens as proof that no user identity exists. **Both readings were wrong, for the same reason:
neither was evidence about this question.**

The Engine's shared HTTP token is *service-to-service* authentication between consumers and the
Engine API. What Vasilis described is *user* authentication across the platform's services. They
are different layers, and the MCP server runs alongside the Engine rather than through it
(ADR-0005), so the Engine's token model does not describe what the agent gets.

**Net position: the original consequence below stands as written, and is not yet evidenced either
way.** Per-Session budgets come first because this repo has no user identity *today*. Whether
central platform auth will hand the agent a user subject, and when, is unknown to me. The
outstanding ask is small and specific: **a pointer to where central authentication is specified**,
so Session identity maps onto the real subject instead of one invented here. Until that arrives,
the agent must not fabricate a user identity to fill the gap — a Session reference is the
strongest subject it can honestly record.

The same call asked for per-tool-call audit records and per-session budgets explicitly, which is
this ADR plus Phase 5, unchanged in shape.

**Accepted 2026-09-19, with two refinements and one question closed.**

**Refinement 1 — the boundary is hybrid, not total.** Taken literally, "never visible from
another" would destroy the disk cache. `dataset_lookup` (`disk_cache.py:156`) finds any cached
entry whose bounds *contain* the request, prefers the smallest such entry, and crops — so one
download of Gothenburg serves every neighbourhood inside it, for everyone. Per-session caches
would reduce that to near zero hit rate.

The split: **public upstream data stays shared; everything derived from user input is
session-keyed.**

> **Corrected 2026-09-20 after an outside review. The claim that followed here was wrong.**
> This ADR originally said the split was nearly free because `CACHE_ALLOWLIST`
> (`disk_cache.py:28-37`) already happened to be exactly the public set. **It is not.** Four of
> its eight entries — `builder.build_terrain_surface_mesh`, `builder.build_city_surface_mesh`,
> `builder.raster.slope_aspect`, `builder.pc_filter.classification_filter` — are builders over
> Objects the user supplied, not downloads derived from bounds. Worse, their cache key comes from
> `content_fingerprint` (`disk_cache.py:40-53`), which despite the name hashes only `type`,
> `source_op`, `nbytes` and `label` and **never reads the contents**. Two sessions whose inputs
> share those four attributes collide and can be served each other's derived results.
>
> **Sharing that cache across sessions would reintroduce the exact leak this ADR exists to
> close**, through the optimisation the ADR used to justify itself.
>
> **The corrected split:** only `datasets.point_cloud`, `datasets.buildings` and `get_buildings`
> stay shared — those are keyed on bounds and source alone. The other five are session-local.
> Restoring cross-session reuse of derived geometry requires a fingerprint that hashes contents,
> tracked as `TODOS.md` T-001.

So the isolation unit is the Session for *state*, and the bounds-and-parameters tuple for *public
derived data*. Both are properties of the object, which preserves this ADR's original reason for
rejecting session-scoped views over a shared store.

**Refinement 2 — no login, and the identifier is designed for substitution.** Sessions are
anonymous browser-scoped tokens. That closes the cross-session memory leak without requiring a
user directory or an identity-provider decision. What matters is that the session identifier is
carried end to end — browser, service, MCP server, stores — so adding an authenticated subject
later is an addition rather than a re-plumb.

> **Two corrections, 2026-09-20.**
>
> **A subject is not a substitute for a session.** Saying an authenticated user id "replaces" the
> anonymous token is wrong: one person has many conversations, and collapsing them onto one
> subject identifier would merge conversations this ADR isolates. **The session identifier and the
> subject identifier are separate fields from the start.** A subject owns many sessions; the
> isolation unit stays the session.
>
> **Anonymous tokens are not an admission boundary.** If anyone reachable can mint a session,
> refusing tokenless requests gates nothing — and a session costs model credit. Minting therefore
> requires a deployment-level shared secret. That secret cannot revoke or attribute an individual
> and is explicitly a placeholder for real authentication, but it makes the boundary real enough to
> gate a deployment.

**Question closed: there is no pointer to central authentication, because there is no
specification.** The outstanding ask above can be retired. The Engine design defers "per-consumer
tokens, roles, permissions, and quotas" from v1 (`docs/dtcc-engine-backend-design-v1.md:74`) and
states "all Engine HTTP services use the shared token; there are no per-consumer roles"
(`:344-347`). Nothing upstream will hand the agent a user subject in the timeframe of this
rebuild. The instruction not to fabricate a user identity therefore stands, and per-Session
budgets come first, as written.

**Scope note: this is new work, not a refactor.** `dtcc_agent/` contains zero occurrences of
`session`, `user_id` or `tenant`. The stores are module-level singletons (`server.py:32-38`)
referenced directly from fifteen tool bodies. There is no seam to thread an identifier through;
`dispatcher.py` is the only module already written to take `store` and `cache` as parameters, and
it is therefore where the rebuild starts.
