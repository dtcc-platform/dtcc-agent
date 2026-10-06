"""The chatbot's half of provenance (T33, #73): one answers.jsonl line per turn.

The chatbot knows first-hand which model answered, under which prompt, and
what it cost; the MCP server records the operations that ran
(dtcc_agent/provenance.py). The two join on the turn id.

SDK messages are read by attribute, so this module needs no SDK import.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from collections.abc import Sequence
from typing import Any

from dtcc_agent.provenance import now

from .config import SYSTEM_PROMPT

USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens",
              "cache_creation_input_tokens")
_SUMMED = ("num_turns", "duration_ms", "duration_api_ms", "total_cost_usd")


def prompt_version(prompt: str = SYSTEM_PROMPT) -> str:
    """The first 12 hex characters of the sha256 of the static system prompt."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def _sdk() -> str:
    try:
        return f"claude-agent-sdk {importlib.metadata.version('claude-agent-sdk')}"
    except importlib.metadata.PackageNotFoundError:
        return "claude-agent-sdk unknown"


PROMPT_VERSION = prompt_version()
SDK = _sdk()
# Where answers come from (#85). Bedrock is the only provider (#29).
PROVIDER = "bedrock"
SDK_COST_SOURCE = "sdk total_cost_usd"


class TurnRecord:
    """What one turn's agent calls report. A fresh retry after a failed
    resume adds to the same record: one turn, one line."""

    def __init__(self, turn_id: str, session_id: str, subject: str, memory_context: bool,
                 *, runtime: str = "sdk", package: str = SDK) -> None:
        self.turn_id = turn_id
        self.runtime = runtime
        self.package = package
        self.cost_source: str | None = None
        self.session_id = session_id
        self.subject = subject
        self.memory_context = memory_context
        self.model: str | None = None
        self.models_used: list[str] = []
        self.tools_called: list[str] = []
        self.totals: dict[str, Any] = {}  # empty until a result arrives
        self.usage: dict[str, int] | None = None
        self.result_error = False
        self.retried_fresh = False
        self.refused = False
        self.error: str | None = None

    def saw_model(self, model: str | None) -> None:
        if model:
            self.model = model

    def saw_tool(self, name: str) -> None:
        self.tools_called.append(name)

    def saw_result(self, result: Any) -> None:
        """Add a ResultMessage's figures to the turn's."""
        for key in _SUMMED:
            value = getattr(result, key, None)
            if value is not None:
                self.totals[key] = self.totals.get(key, 0) + value
        usage = getattr(result, "usage", None) or {}
        if self.usage is None:
            self.usage = dict.fromkeys(USAGE_KEYS, 0)
        for key in USAGE_KEYS:
            self.usage[key] += usage.get(key) or 0
        for model in getattr(result, "model_usage", None) or {}:
            if model not in self.models_used:
                self.models_used.append(model)
        self.result_error = self.result_error or bool(getattr(result, "is_error", False))
        self.cost_source = SDK_COST_SOURCE

    def saw_run(self, usage: Any, *, elapsed_ms: int, cost: float | None,
                cost_source: str, models: Sequence[str] = ()) -> None:
        """A pydantic-ai turn's figures, from its RunUsage (every attempt,
        retries included). pydantic-ai counts cache reads and writes inside
        input_tokens; here input_tokens means fresh input, as the SDK's did."""
        self.usage = {
            "input_tokens": usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_input_tokens": usage.cache_read_tokens,
            "cache_creation_input_tokens": usage.cache_write_tokens,
        }
        self.totals = {"num_turns": usage.requests, "duration_ms": elapsed_ms}
        if cost is not None:
            self.totals["total_cost_usd"] = cost
        self.cost_source = cost_source
        for model in models:
            if model not in self.models_used:
                self.models_used.append(model)

    def retry(self) -> None:
        self.retried_fresh = True

    def refuse(self) -> None:
        """The model declined the request (#96): an answer, not an error."""
        self.refused = True

    def failed(self, exc: BaseException) -> None:
        """The turn ended without an answer: record the class, never the text."""
        self.error = type(exc).__name__

    def record(self) -> dict[str, Any]:
        """The answers.jsonl line, every field present; unknowns are null."""
        return {
            "at": now(), "turn_id": self.turn_id, "session_id": self.session_id,
            "subject": self.subject, "model": self.model, "models_used": self.models_used,
            "prompt_version": PROMPT_VERSION, "memory_context": self.memory_context,
            "sdk": self.package, "tools_called": self.tools_called,
            **{key: self.totals.get(key) for key in _SUMMED},
            "usage": self.usage,
            "is_error": self.error is not None or self.result_error,
            "retried_fresh": self.retried_fresh, "refused": self.refused, "error": self.error,
            "runtime": self.runtime, "provider": PROVIDER, "cost_source": self.cost_source,
        }
