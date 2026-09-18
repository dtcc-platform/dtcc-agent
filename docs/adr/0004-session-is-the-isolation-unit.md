---
status: proposed
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
