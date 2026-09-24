"""Conversation memory is scoped to one Session (ADR-0004, M1a/T5 #18)."""

import pytest

pytest.importorskip("chromadb")

from chromadb.api.types import EmbeddingFunction

from chatbot.memory import ConversationMemory


class _BagOfLetters(EmbeddingFunction):
    """Deterministic and offline, so the test never downloads a model."""

    def __init__(self):
        pass

    def __call__(self, input):
        return [
            [float(text.lower().count(c)) + 0.01 for c in "abcdefghijklmnopqrstuvwxyz"]
            for text in input
        ]

    @staticmethod
    def name():
        return "bag-of-letters"


@pytest.fixture
def memory(tmp_path, monkeypatch):
    mem = ConversationMemory(persist_dir=tmp_path)
    mem._collection = mem._client.create_collection(
        "test", embedding_function=_BagOfLetters(), metadata={"hnsw:space": "cosine"}
    )
    return mem


def test_retrieve_returns_this_sessions_exchanges(memory):
    memory.store("s1", "flood risk in Lindholmen", "Here is the flood map.")
    assert "flood map" in memory.retrieve("flood risk in Lindholmen", session_id="s1")


def test_retrieve_never_returns_another_sessions_exchanges(memory):
    memory.store("s1", "flood risk in Lindholmen", "Here is the flood map.")
    assert memory.retrieve("flood risk in Lindholmen", session_id="s2") == ""
