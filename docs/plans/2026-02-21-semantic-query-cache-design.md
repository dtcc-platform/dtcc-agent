# Semantic Query Cache

## Problem

Users often ask similar or identical questions across sessions. Each time, the full LLM pipeline runs (RAG retrieval, Claude Agent SDK call, MCP tool execution). A semantic cache can return previous answers instantly for near-duplicate queries.

## Approach

Extend `ConversationMemory` with a second ChromaDB collection (`query_cache`) that indexes user queries and stores their answers. On each request, check the cache first; if a close semantic match exists within TTL, return it directly with a label — skipping the LLM call entirely.

## Data Model

Collection `query_cache` in the existing ChromaDB persistent client:

- **Document text:** user query only (not combined Q&A, to keep search signal clean)
- **Doc ID:** `cache-<timestamp>-<random6>`
- **Metadata:**
  - `session_id` — originating session
  - `timestamp` — ISO string, for TTL expiry checks
  - `answer` — the cached assistant response (up to ~30K chars)
  - `user_query` — first 500 chars, for logging/debugging

Constants:
- `CACHE_DISTANCE_THRESHOLD = 0.20` (cosine distance; configurable)
- `CACHE_TTL_HOURS = 24` (configurable)

## Integration Flow

```
User message arrives
  ├─ Length check
  ├─ cache_lookup(user_text) ← NEW
  │    ├─ Hit → send labeled response + "done", skip LLM
  │    └─ Miss → continue normal flow ↓
  ├─ RAG context retrieval
  ├─ LLM call + streaming
  ├─ memory.store() for RAG (unchanged)
  ├─ cache_store(session_id, user_text, assistant_text) ← NEW
  └─ send "done"
```

Key decisions:
- Cache check runs before RAG retrieval — a hit skips both RAG and LLM
- Only fresh LLM responses are cached (no re-caching of cache hits)
- Cache check applies to all sessions (fresh and resumed)
- Labeled response: `"*From a previous conversation:*\n\n{cached_answer}"`

## API Surface

Two new methods on `ConversationMemory` in `chatbot/memory.py`:

```python
def cache_lookup(self, query: str) -> str | None:
    """Return cached answer if a semantically similar question exists
    within distance threshold and TTL. None on miss."""

def cache_store(self, session_id: str, user_text: str, assistant_text: str) -> None:
    """Store a Q&A pair in the query cache."""
```

No new files, classes, or dependencies.

## Files Changed

- `chatbot/memory.py` — add `query_cache` collection, two new methods, two new constants
- `chatbot/app.py` — ~10 lines: cache check before LLM call, cache store after
