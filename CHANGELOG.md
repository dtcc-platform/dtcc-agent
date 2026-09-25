# Changelog

Everything delivered on dtcc-agent since the rebuild started, newest first. Each entry says
what changed, why it matters, and how we know it works, so it can be walked through in a
team meeting without opening the code.

**Status:** ✅ merged to `develop` · 🔍 open pull request, in review · ⏳ decision or task still open

Last updated: 2026-09-25.

---

## At a glance

| When | What | Status |
|---|---|---|
| 2026-09-25 | dtcc-core pin moved to Core's latest `develop`, picking up the upstream fixes ([#35](https://github.com/dtcc-platform/dtcc-agent/pull/35)) | ✅ |
| 2026-09-25 | Session isolation over HTTP: each user's objects, runs and memory kept apart ([#33](https://github.com/dtcc-platform/dtcc-agent/pull/33)) | ✅ |
| 2026-09-25 | Every tool runs off the event loop, so Core downloads work under the web server ([#32](https://github.com/dtcc-platform/dtcc-agent/pull/32)) | ✅ |
| 2026-09-24 | Automated first-pass review on every pull request ([#31](https://github.com/dtcc-platform/dtcc-agent/pull/31)) | ✅ |
| 2026-09-24 | M1 reviewed a second time, split into M1a and M1b, and the whole programme put on GitHub ([#9](https://github.com/dtcc-platform/dtcc-agent/pull/9)) | ✅ |
| 2026-09-21 | **M0 done:** dtcc-core pinned, CI running, 61 tests describing today's tool surface ([#7](https://github.com/dtcc-platform/dtcc-agent/pull/7), [#8](https://github.com/dtcc-platform/dtcc-agent/pull/8)) | ✅ |
| 2026-09-21 | Working conventions for issues, triage and agents ([#6](https://github.com/dtcc-platform/dtcc-agent/pull/6)) | ✅ |
| 2026-09-21 | Glossary, ten architecture decisions (ADRs) and the rebuild plan ([#3](https://github.com/dtcc-platform/dtcc-agent/pull/3)) | ✅ |
| 2026-09-21 | The server starts on a fresh install again ([#2](https://github.com/dtcc-platform/dtcc-agent/pull/2)) | ✅ |
| 2026-09-18 | Four Core and Sim defects reported upstream; all four fixed by the Core team, and now in our build | ✅ |
| 2026-09-14 | Assessment of what works today ([#1](https://github.com/dtcc-platform/dtcc-agent/issues/1)) | ✅ |

**Tests:** 112 before the rebuild → 189 after M0 → 194 with #32 → 212 with #33, all passing on the new Core pin.

---

## ✅ Merged

### dtcc-core pin moved to Core's latest `develop` · 2026-09-25 · [#35](https://github.com/dtcc-platform/dtcc-agent/pull/35)

**Before:** we were pinned to Core `18eb176` from 18 September. Two problems:
- It was older than the fixes the Core team made for the defects we reported.
- Core's `develop` history was rewritten after we pinned, so that commit is no longer on any
  Core branch. A commit on no branch can be deleted by GitHub, and then a fresh install of this
  repo would fail.

**Now:** pinned to `9b4e9b9`, the head of Core's `develop` on 24 September. Nothing else in the
dependency lock moved.

**How we know it works:**
- The contract workflow, which tests a candidate Core before the pin moves, passes: the
  installed Core is the right commit, 194 tests pass, and the catalogue stays at 133
  operations.
- With T5 on top, all 212 tests pass on the new Core.
- One fix checked before and after, through the agent itself: reprojecting a mesh that
  carries a data field fails on the old pin ("Fields and semantic regions require an explicit
  reprojection rule") and works on the new one, keeping the field.

**Also fixed:** the contract workflow could never pass for any Core. It skipped installing the
chatbot's dependencies, so the test run stopped before testing anything. It now installs the
same things as the main CI.

### Session isolation over HTTP (M1a/T5) · 2026-09-25 · [#33](https://github.com/dtcc-platform/dtcc-agent/pull/33)

**Before:** every person using the chatbot shared one object store and one list of runs. One
user could see, use or delete what another user had built. Conversation memory also searched
everyone's past chats.

**Now:**
- The MCP server can run over HTTP (`DTCC_MCP_TRANSPORT=http`). stdio stays the default, so
  switching back is a configuration change.
- The chatbot sends its session id with every tool call (the `X-DTCC-Session` header). Each
  session gets its own objects and runs. A call without a session id is refused.
- Conversation memory only searches the current session's past messages.
- At most 8 sessions are kept in memory at once, each with a 256 MiB share, so the server
  stays under the same 2 GiB it had before. A session in the middle of a tool call is never
  dropped.

**Why a header and not the connection:** the chatbot opens a new connection for every
message. Anything tied to the connection would be forgotten after each message.

**How we know it works:**
- Two real clients over HTTP cannot see each other's objects, and a session still finds its
  objects on the next message.
- Tested end to end with the real Claude Code client: the right session sees its object, the
  other sees nothing.
- The review caught a leak before merge: the default HTTP mode kept 2 server tasks alive for
  every chat message, forever. Measured again after the fix: 0.
- 18 new tests, including one proving the default stdio mode still works end to end.

**Still open, each with an owner:**
- Only reachable through `localhost` until the deployment task sets the allowed host names
  (T13, [#24](https://github.com/dtcc-platform/dtcc-agent/issues/24)).
- The session id is not a password. Anyone who can reach the port and knows an id can read
  that session. Who may connect is decision U11 ([#15](https://github.com/dtcc-platform/dtcc-agent/issues/15)).
- Exported files still share one folder (T7, [#21](https://github.com/dtcc-platform/dtcc-agent/issues/21)),
  and the builder cache is still shared between sessions (T6, [#20](https://github.com/dtcc-platform/dtcc-agent/issues/20)).
- The 8-session cap is a stopgap. The real memory budget, with session expiry, is T11
  ([#23](https://github.com/dtcc-platform/dtcc-agent/issues/23)).

### Tools run off the event loop (M1a/T4) · 2026-09-25 · [#32](https://github.com/dtcc-platform/dtcc-agent/pull/32)

**Before:** dtcc-core starts its own event loop inside every lidar and GeoPackage download.
The server ran tools on its main loop and relied on a patch (`nest_asyncio`) to allow that.
The patch does not work with the faster loop the web server uses, so the first real download
after moving to HTTP would have failed.

**Now:**
- Every tool body runs on a worker thread, where Core can start its own loop safely. The
  patch is gone.
- Rendering (`render_object`) stays on the main thread. The graphics library needs that, and
  on macOS anything else crashes the whole server.
- Deleting an object is now one safe step. Two tools can run at the same time now, and the
  old check-then-delete could fail halfway.

**How we know it works:** with both download caches emptied, `develop` fails with
`asyncio.run() cannot be called from a running event loop`. This branch downloads the tiles
and returns the point cloud, also when served by the web server over HTTP.

**Found in review and handed on:**
- Two users downloading the same new tile at once can make one of the downloads fail. The bug
  is in Core's downloader, filed as [dtcc-core#126](https://github.com/dtcc-platform/dtcc-core/issues/126).
- Up to 40 tools can now run at once. Putting a limit on that is T8
  ([#19](https://github.com/dtcc-platform/dtcc-agent/issues/19)), in the same milestone.

### Automated first-pass review · 2026-09-24 · [#31](https://github.com/dtcc-platform/dtcc-agent/pull/31)

[PR-Agent](https://docs.pr-agent.ai) now reviews every pull request when it opens, using
Gemini. It posts a short review and code suggestions, and answers `/review`, `/improve` and
`/ask` comments.

- **An extra read, not a gate.** It never approves or blocks, and our own reviews are
  unchanged. It reads only the diff in one pass, so its suggestions are hints to check.
- **Only people with write access can trigger it by comment.** The repo is public, and
  otherwise anyone could spend the API key.
- **Installed from PyPI rather than as a GitHub Action,** because the organisation only
  allows actions it owns, GitHub's own, or Marketplace-verified ones.

Its first real suggestion, on #32, was valid. It was applied in a stricter form: only the one test that needs uvloop skips, rather than falling back to a loop that would let it pass untested.

### M1 reviewed and split; the programme tracked on GitHub · 2026-09-24 · [#9](https://github.com/dtcc-platform/dtcc-agent/pull/9)

A second engineering review of milestone M1, with Codex as an independent second reviewer.
It produced 31 findings.

- **M1 split into M1a** (transport, session state, execution) **and M1b** (typed
  references, cache versioning). Same ten tasks, nothing cut. They must not be built in
  parallel, because both rewrite `server.py`.
- **The concurrency limit (T8) moved into M1a,** next to the change that makes it necessary.
- **The work is now on GitHub:**
  - 6 milestones and 21 issues, with sub-issues and pull request links.
  - 6 decision issues that block M1a.
  - No task issues for M2 to M4 yet. They are not specified, and guessed issues would read
    as agreed scope.
- **Codex overturned one of our own conclusions.** FastMCP's startup hook runs once per
  session, not once per process. That changed the design of three tasks. Verified in the
  library source before accepting.

### M0: dtcc-core pinned, CI running, today's behaviour recorded · 2026-09-21 · [#7](https://github.com/dtcc-platform/dtcc-agent/pull/7), [#8](https://github.com/dtcc-platform/dtcc-agent/pull/8)

**The problem:** dtcc-core was not listed as a dependency. When it was missing, eight places
in the code quietly skipped their work. A fresh install started without errors and offered an
empty catalogue. It looked like a working server with nothing in it.

**What changed:**
- dtcc-core is declared and pinned to one exact commit (`18eb176`), so everyone runs the
  same Core.
- A missing Core now fails loudly at startup, and the error says how to fix it.
- A contract workflow tests a new Core version before the pin is moved.
- CI runs on every push and pull request.
- 61 characterisation tests record what the 22 tools do today, including what they do wrong.
  When M1 changes that behaviour on purpose, these tests fail on purpose and get updated. That
  way no change in behaviour goes unnoticed.

**Result:** 189 tests passing, catalogue of 133 operations, CI green.

### Working conventions · 2026-09-21 · [#6](https://github.com/dtcc-platform/dtcc-agent/pull/6)

Written conventions for GitHub issues, triage labels and the project glossary, read by both
Claude and Codex. `AGENTS.md` holds them; `CLAUDE.md` points to it.

### Glossary, decisions and the rebuild plan · 2026-09-21 · [#3](https://github.com/dtcc-platform/dtcc-agent/pull/3)

- **`CONTEXT.md`:** a glossary. Where Twin already defines a term, its definition wins.
- **Architecture decisions (ADRs):**

  | ADR | Decision |
  |---|---|
  | 0001 | The agent is the conversational front door |
  | 0002 | Converge on Twin's contracts |
  | 0003 | Replace the Claude Agent SDK with pydantic-ai |
  | 0004 | The Session is the isolation unit |
  | 0005 | Retrieval is a separate MCP server |
  | 0006 | Prompt caching shaped for Anthropic |
  | 0007 | The agent keeps its own generic dispatch |
  | 0008 | Evaluation is one harness with two layers |
  | 0009 | Rebuild on a branch, not a fresh repository |
  | 0010 | References are typed, and a run records its object |

- **The rebuild plan:** milestones M0 to M4 in `docs/plans/2026-09-19-rebuild-plan.md`.

### The server starts on a fresh install again · 2026-09-21 · [#2](https://github.com/dtcc-platform/dtcc-agent/pull/2)

A clean install picked up version 2 of the `mcp` library, which removed the module the server
imports, so `python -m dtcc_agent` crashed. The tests still passed, because none of them
loaded the server. Fixed by pinning `mcp` below version 2, and by adding a test that loads the
server, so this can't go unnoticed again.

### Upstream fixes in dtcc-core and dtcc-sim · reported 2026-09-18, fixed 2026-09-22/23, in our build 2026-09-25

Found while checking the agent against the current Core. All four were fixed by the Core team.

| Issue | Problem |
|---|---|
| [dtcc-core#110](https://github.com/dtcc-platform/dtcc-core/issues/110) | Building a terrain raster from ground points rejected every downloaded point cloud |
| [dtcc-core#111](https://github.com/dtcc-platform/dtcc-core/issues/111) | Buildings with no lidar points on the roof were silently deleted |
| [dtcc-core#112](https://github.com/dtcc-platform/dtcc-core/issues/112) | No rule for reprojecting geometry that carries data fields |
| [dtcc-sim#8](https://github.com/dtcc-platform/dtcc-sim/issues/8) | Two simulations could not write the native `dtcc` output format |

**In our build since #35.** The Core fixes arrived with the pin move to `9b4e9b9`. The Sim fix
lives in dtcc-sim and doesn't depend on our pin.

### Assessment of what works · 2026-09-14 · [#1](https://github.com/dtcc-platform/dtcc-agent/issues/1)

A checked write-up of the starting point. The component tests were green, but the server
could not start from a fresh install. The docs and dependencies had drifted, results from
the mini-service were incomplete, and three README examples could not be reproduced. Posted on
#1 and used as the input to the plan above.

---

## ⏳ Open decisions

These block tasks in M1a. Each issue carries the evidence needed to decide.

| Decision | Blocks | Status |
|---|---|---|
| U3: where once-per-process startup lives ([#12](https://github.com/dtcc-platform/dtcc-agent/issues/12)) | T8, T10, T11 | ✅ Decided 2026-09-24: at process startup, not in FastMCP's per-session hook |
| U1: how far the filesystem boundary goes ([#10](https://github.com/dtcc-platform/dtcc-agent/issues/10)) | T7 | ⏳ |
| U2: fix the cache keys, or turn builder caching off ([#11](https://github.com/dtcc-platform/dtcc-agent/issues/11)) | T6 | ⏳ |
| U4: how accurate the memory budget must be ([#13](https://github.com/dtcc-platform/dtcc-agent/issues/13)) | T11 | ⏳ |
| U10: which dtcc-core install wins in the container ([#14](https://github.com/dtcc-platform/dtcc-agent/issues/14)) | T13 | ⏳ |
| U11: which network interface the MCP server listens on, and who may connect ([#15](https://github.com/dtcc-platform/dtcc-agent/issues/15)) | T13 | ⏳ |

## What's next

1. **The rest of M1a:** limit concurrent Core work (T8), split the cache (T6), per-session
   file folders (T7), build the catalogue once per process (T10), the memory budget (T11),
   and the two-service container (T13).
2. **M1b:** typed references, where a run and its object stay linked (T9), and cache
   versioning (T12).

## For discussion with the team

- **Who may reach the MCP server** (U11, #15). This decides whether the session id alone is
  enough, and what host names the deployment allows.
- **PR-Agent.** It runs on every PR on one Gemini API key. Who owns that key, and is an
  extra automated read worth it for the team?
