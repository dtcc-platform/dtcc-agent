"""RAG-based conversation memory using ChromaDB."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path

import chromadb

logger = logging.getLogger("lurkie.memory")

# Persist conversation memory across server restarts
_PERSIST_DIR = Path("/tmp/dtcc_lurkie_memory")

# How many past exchanges to retrieve per query
TOP_K = 5

# Cosine distance threshold — 0 = identical, 2 = opposite.
# Only return results closer than this.
MAX_DISTANCE = 0.8


class ConversationMemory:
    """Stores and retrieves past conversation exchanges using ChromaDB."""

    def __init__(self, persist_dir: Path = _PERSIST_DIR) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Memory initialized: %d stored exchanges in %s",
            self._collection.count(),
            persist_dir,
        )

    def store(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Store a user-assistant exchange."""
        now = datetime.now()
        doc = f"User: {user_text}\nAssistant: {assistant_text}"
        doc_id = f"{session_id}-{now:%Y%m%d%H%M%S%f}-{uuid.uuid4().hex[:6]}"
        self._collection.add(
            documents=[doc],
            ids=[doc_id],
            metadatas=[{
                "session_id": session_id,
                "timestamp": now.isoformat(),
                "user_query": user_text[:500],
            }],
        )
        logger.debug("Stored exchange %s (%d chars)", doc_id, len(doc))

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        """Retrieve relevant past exchanges for a query.

        Returns a formatted string to inject into the system prompt,
        or empty string if no relevant history found.
        """
        if self._collection.count() == 0:
            return ""

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        if not documents:
            return ""

        # Only keep results that are semantically close enough
        filtered = [
            doc for doc, dist in zip(documents, distances) if dist < MAX_DISTANCE
        ]
        if not filtered:
            logger.debug("No relevant past exchanges (all distances > %.1f)", MAX_DISTANCE)
            return ""

        context = "\n\n---\n\n".join(filtered)
        logger.info("Retrieved %d relevant past exchanges for query", len(filtered))
        return (
            "Here are relevant excerpts from past conversations with this user. "
            "Use them for context but don't repeat information unless asked:\n\n"
            f"{context}"
        )
