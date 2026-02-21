# Semantic Query Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a semantic query cache to the Lurkie chatbot that returns previous answers instantly for near-duplicate questions, skipping the LLM call.

**Architecture:** A second ChromaDB collection (`query_cache`) in the existing `ConversationMemory` class indexes user queries by semantic similarity. On each request, the cache is checked first; hits within distance threshold and TTL are returned with a label. Two new methods (`cache_lookup`, `cache_store`) and ~10 lines in `app.py`.

**Tech Stack:** ChromaDB (already a dependency), Python datetime for TTL

---

### Task 1: Add cache collection and `cache_store` method

**Files:**
- Modify: `chatbot/memory.py:1-96`
- Test: `tests/test_chatbot_memory.py` (create)

**Step 1: Write the failing test**

Create `tests/test_chatbot_memory.py`:

```python
"""Tests for ConversationMemory query cache."""

import tempfile
from pathlib import Path

from chatbot.memory import ConversationMemory


def test_cache_store_increases_count():
    with tempfile.TemporaryDirectory() as td:
        mem = ConversationMemory(persist_dir=Path(td))
        assert mem._cache.count() == 0
        mem.cache_store("sess-1", "What is Lindholmen?", "Lindholmen is a district in Gothenburg.")
        assert mem._cache.count() == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chatbot_memory.py::test_cache_store_increases_count -v`
Expected: FAIL with `AttributeError: 'ConversationMemory' object has no attribute '_cache'`

**Step 3: Write minimal implementation**

In `chatbot/memory.py`, add constants after `MAX_DISTANCE = 0.8`:

```python
# --- Query cache settings ---
CACHE_DISTANCE_THRESHOLD = 0.20
CACHE_TTL_HOURS = 24
```

In `ConversationMemory.__init__`, add after the `_collection` creation:

```python
self._cache = self._client.get_or_create_collection(
    name="query_cache",
    metadata={"hnsw:space": "cosine"},
)
logger.info(
    "Query cache initialized: %d cached answers",
    self._cache.count(),
)
```

Add the `cache_store` method after the existing `retrieve` method:

```python
def cache_store(
    self,
    session_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    """Store a Q&A pair in the query cache."""
    now = datetime.now()
    doc_id = f"cache-{now:%Y%m%d%H%M%S%f}-{uuid.uuid4().hex[:6]}"
    self._cache.add(
        documents=[user_text],
        ids=[doc_id],
        metadatas=[{
            "session_id": session_id,
            "timestamp": now.isoformat(),
            "answer": assistant_text[:30_000],
            "user_query": user_text[:500],
        }],
    )
    logger.debug("Cached answer %s (%d chars)", doc_id, len(assistant_text))
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_chatbot_memory.py::test_cache_store_increases_count -v`
Expected: PASS

**Step 5: Commit**

```bash
git add chatbot/memory.py tests/test_chatbot_memory.py
git commit -m "feat: add query cache collection and cache_store method"
```

---

### Task 2: Add `cache_lookup` method

**Files:**
- Modify: `chatbot/memory.py`
- Test: `tests/test_chatbot_memory.py`

**Step 1: Write the failing tests**

Append to `tests/test_chatbot_memory.py`:

```python
def test_cache_lookup_returns_none_when_empty():
    with tempfile.TemporaryDirectory() as td:
        mem = ConversationMemory(persist_dir=Path(td))
        assert mem.cache_lookup("What is Lindholmen?") is None


def test_cache_lookup_hits_on_similar_query():
    with tempfile.TemporaryDirectory() as td:
        mem = ConversationMemory(persist_dir=Path(td))
        mem.cache_store("sess-1", "What is Lindholmen?", "Lindholmen is a district.")
        # Exact same query should hit
        result = mem.cache_lookup("What is Lindholmen?")
        assert result is not None
        assert "Lindholmen is a district." in result


def test_cache_lookup_returns_none_for_unrelated_query():
    with tempfile.TemporaryDirectory() as td:
        mem = ConversationMemory(persist_dir=Path(td))
        mem.cache_store("sess-1", "What is Lindholmen?", "Lindholmen is a district.")
        # Completely different question should miss
        result = mem.cache_lookup("How do I install Python on Windows?")
        assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chatbot_memory.py -v -k "cache_lookup"`
Expected: FAIL with `AttributeError: 'ConversationMemory' object has no attribute 'cache_lookup'`

**Step 3: Write minimal implementation**

Add to `ConversationMemory` after `cache_store`:

```python
def cache_lookup(self, query: str) -> str | None:
    """Check if a semantically similar question was answered before.

    Returns the cached answer if found within distance threshold
    and TTL, or None on miss.
    """
    if self._cache.count() == 0:
        return None

    results = self._cache.query(
        query_texts=[query],
        n_results=1,
        include=["documents", "distances", "metadatas"],
    )

    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not distances or not metadatas:
        return None

    distance = distances[0]
    metadata = metadatas[0]

    if distance > CACHE_DISTANCE_THRESHOLD:
        logger.debug("Cache miss: closest distance %.3f > threshold %.2f",
                      distance, CACHE_DISTANCE_THRESHOLD)
        return None

    # TTL check
    cached_time = datetime.fromisoformat(metadata["timestamp"])
    age_hours = (datetime.now() - cached_time).total_seconds() / 3600
    if age_hours > CACHE_TTL_HOURS:
        logger.debug("Cache miss: entry expired (%.1f hours old)", age_hours)
        return None

    logger.info("Cache hit: distance=%.3f, age=%.1fh, query=%s",
                distance, age_hours, metadata.get("user_query", "")[:80])
    return metadata["answer"]
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chatbot_memory.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add chatbot/memory.py tests/test_chatbot_memory.py
git commit -m "feat: add cache_lookup method with distance threshold and TTL"
```

---

### Task 3: Add TTL expiry test

**Files:**
- Test: `tests/test_chatbot_memory.py`

**Step 1: Write the failing test**

Append to `tests/test_chatbot_memory.py`:

```python
from unittest.mock import patch
from datetime import datetime, timedelta


def test_cache_lookup_returns_none_when_expired():
    with tempfile.TemporaryDirectory() as td:
        mem = ConversationMemory(persist_dir=Path(td))
        mem.cache_store("sess-1", "What is Lindholmen?", "Lindholmen is a district.")

        # Simulate time passing beyond TTL
        future = datetime.now() + timedelta(hours=25)
        with patch("chatbot.memory.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.fromisoformat = datetime.fromisoformat
            result = mem.cache_lookup("What is Lindholmen?")
        assert result is None
```

**Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_chatbot_memory.py::test_cache_lookup_returns_none_when_expired -v`
Expected: PASS (implementation already handles TTL)

**Step 3: Commit**

```bash
git add tests/test_chatbot_memory.py
git commit -m "test: add TTL expiry test for query cache"
```

---

### Task 4: Integrate cache into chat flow

**Files:**
- Modify: `chatbot/app.py:210-276`
- Test: `tests/test_chatbot_app.py`

**Step 1: Write the failing test**

Append to `tests/test_chatbot_app.py`:

```python
from unittest.mock import patch


def test_cache_hit_returns_labeled_response():
    client = TestClient(app)
    with client.websocket_connect("/chat") as ws:
        ws.send_json({"session_id": None})
        session_msg = ws.receive_json()
        assert session_msg["type"] == "session"

        # Mock cache_lookup to return a cached answer
        with patch.object(
            app.state if hasattr(app, "state") else type(ws),
            "__class__",  # placeholder — see step 3 for actual patching
        ):
            pass
```

> **Note to implementer:** The exact test mock depends on how `memory` is referenced. The simplest approach is to patch `chatbot.app.memory.cache_lookup`. Write the integration in step 3 first, then finalize this test to match.

**Step 2: Integrate cache check in `app.py`**

In `chatbot/app.py`, after the length check (line 219) and before RAG retrieval (line 224), add:

```python
# Check semantic query cache before invoking LLM
cached_answer = memory.cache_lookup(user_text)
if cached_answer is not None:
    logger.info("[%s] Cache hit, returning cached answer", session_id)
    await ws.send_json({
        "type": "text",
        "content": f"*From a previous conversation:*\n\n{cached_answer}",
    })
    await ws.send_json({"type": "done"})
    continue
```

After the existing `memory.store()` call (line 274), add:

```python
memory.cache_store(session_id, user_text, assistant_text)
```

**Step 3: Write the proper test**

Replace the placeholder test with:

```python
def test_cache_hit_skips_llm():
    client = TestClient(app)
    with client.websocket_connect("/chat") as ws:
        ws.send_json({"session_id": None})
        session_msg = ws.receive_json()
        assert session_msg["type"] == "session"

        with patch("chatbot.app.memory") as mock_memory:
            mock_memory.cache_lookup.return_value = "Cached answer text"
            ws.send_json({"content": "test query"})

            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg.get("type") == "done":
                    break

            # Should have text + done, no LLM call
            text_msgs = [m for m in messages if m["type"] == "text"]
            assert len(text_msgs) == 1
            assert "From a previous conversation" in text_msgs[0]["content"]
            assert "Cached answer text" in text_msgs[0]["content"]
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chatbot_app.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add chatbot/app.py tests/test_chatbot_app.py
git commit -m "feat: integrate semantic query cache into chat flow"
```

---

### Task 5: Final verification

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 2: Manual smoke test (optional)**

Start the chatbot and ask the same question twice. The second time should return instantly with the "From a previous conversation" label.

**Step 3: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: cleanup after semantic query cache implementation"
```
