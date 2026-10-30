"""Turn and spend limits for a deployment (#132).

The first deployment is an invited beta with a fixed Bedrock budget, and the
access code admits people without limiting what they spend. Four settings,
each off when unset:

- DTCC_AGENT_MONTHLY_SPEND_USD: total model spend per calendar month (UTC).
  Never exceeded: before any model request starts, the most it can cost is
  set aside, and a request that doesn't fit is refused.
- DTCC_AGENT_TURNS_PER_CHAT: turns one chat may take.
- DTCC_AGENT_TURN_SECONDS: how long one turn may run before it is stopped.
- DTCC_AGENT_MODEL_CONCURRENCY: model requests running at once, across chats.

Every model request, a turn's, its fresh retry's and the startup probe's,
goes through LimitedModel. Cost is known only after a request, sessions run
turns concurrently and one turn makes several requests, so the spend limit
counts the worst case of every request still running. The worst case is
bounded by the request's size: a token covers at least one byte of text, so
the serialized request bounds its input tokens, and output is capped at
MAX_OUTPUT_TOKENS. Input is priced at the cache-write rate, the dearest.

The ledger is a file next to the answer records, so a restart doesn't reset
the month: an amount still set aside when the server stopped counts as spent.
One chat server process owns the file.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pydantic_core
from genai_prices import Usage, calc_price
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from chatbot.prices import price
from chatbot.provenance import http_status

logger = logging.getLogger("lurkie")

# Every model request is capped at this many output tokens. The longest
# recorded turn (369 pydantic-ai turns, 2026-10) used 4,038 across all its
# requests, so the cap bounds cost without cutting answers short.
MAX_OUTPUT_TOKENS = 8192
# Tokens Bedrock adds that the serialized request doesn't show: its tool-use
# system prompt and message framing.
INPUT_MARGIN_TOKENS = 4096

# Said instead of an answer when a limit stops a turn.
SPEND_REACHED = ("This service has reached its spending limit for the month, so it can't "
                 "answer now. Please tell the administrator.")
TURNS_REACHED = "This chat has reached its limit of {limit} questions."
TURN_TIMED_OUT = ("This answer took longer than {seconds:.0f} seconds and was stopped. "
                  "Try a smaller question or area.")


class LimitReached(Exception):
    """A limit stopped a turn: an answer for the user, never retried."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _setting(name: str, kind: type, minimum: float) -> Any:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = kind(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a number, not {raw!r}.") from None
    if value < minimum:
        raise SystemExit(f"{name} must be at least {minimum}.")
    return value


def monthly_spend_usd() -> float | None:
    return _setting("DTCC_AGENT_MONTHLY_SPEND_USD", float, 0.01)


def turns_per_chat() -> int | None:
    return _setting("DTCC_AGENT_TURNS_PER_CHAT", int, 1)


def turn_seconds() -> float | None:
    return _setting("DTCC_AGENT_TURN_SECONDS", float, 1)


def model_concurrency() -> int | None:
    return _setting("DTCC_AGENT_MODEL_CONCURRENCY", int, 1)


def _month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


class SpendLedger:
    """This month's spend and what running requests have set aside.

    reserve and settle never await, so on the event loop a check and the
    amount it sets aside can't be interleaved with another request's."""

    def __init__(self, path: Path, limit_usd: float, month: Callable[[], str] = _month):
        self.path = path
        self.limit = limit_usd
        self._month_of = month
        self._lock = threading.Lock()
        state = self._read()
        if state.get("month") == self._month_of():
            spent = float(state.get("spent_usd", 0.0))
            # Set aside when the server stopped: the request may have run.
            spent += sum(float(v) for v in state.get("held_usd", {}).values())
        else:
            spent = 0.0
        self.month = self._month_of()
        self.spent = spent
        self.held: dict[str, float] = {}
        self._write()

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            # Unreadable: starting at zero could overspend, so refuse.
            raise SystemExit(f"The spend ledger {self.path} can't be read ({exc}); "
                             "fix or remove it to start.") from exc

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"month": self.month, "spent_usd": self.spent, "held_usd": self.held}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def reserve(self, amount: float) -> str:
        """Set amount aside, or raise LimitReached if it doesn't fit."""
        with self._lock:
            month = self._month_of()
            if month != self.month:
                # A new month: what was spent is behind us; what is still set
                # aside is not, and still counts.
                self.month, self.spent = month, 0.0
            if self.spent + sum(self.held.values()) + amount > self.limit:
                raise LimitReached(SPEND_REACHED)
            key = secrets.token_hex(8)
            self.held[key] = amount
            self._write()
            return key

    def settle(self, key: str, cost: float | None) -> None:
        """Replace what key set aside with what was spent: None keeps all of it."""
        with self._lock:
            held = self.held.pop(key)
            if cost is not None and cost > held:
                logger.warning("A model request cost $%.4f, more than the $%.4f set aside for it.",
                               cost, held)
            self.spent += held if cost is None else cost
            self._write()

    def total(self) -> float:
        return self.spent + sum(self.held.values())


class LimitedModel(WrapperModel):
    """The Bedrock model, with every request capped, counted and bounded."""

    def __init__(self, wrapped: Any, model_id: str, ledger: SpendLedger | None = None,
                 concurrency: int | None = None):
        super().__init__(wrapped)
        self.price_id = model_id
        self.ledger = ledger
        self._slots = asyncio.Semaphore(concurrency) if concurrency else None
        self._context_window = None
        if ledger is not None:
            try:
                found = calc_price(Usage(output_tokens=1), model_ref=model_id, provider_id="aws")
            except LookupError:
                raise SystemExit(f"DTCC_AGENT_MONTHLY_SPEND_USD is set, but {model_id} has no "
                                 "known price, so its spend can't be bounded.") from None
            self._context_window = found.model.context_window

    def _bounded(self, settings: ModelSettings | None) -> ModelSettings:
        asked = (settings or {}).get("max_tokens") or MAX_OUTPUT_TOKENS
        return {**(settings or {}), "max_tokens": min(asked, MAX_OUTPUT_TOKENS)}

    def worst_case(self, messages: list[ModelMessage], settings: ModelSettings,
                   parameters: ModelRequestParameters) -> float:
        """The most this request can cost, in USD."""
        size = (len(pydantic_core.to_json(messages, fallback=repr))
                + len(pydantic_core.to_json(parameters, fallback=repr)))
        tokens = size + INPUT_MARGIN_TOKENS
        if self._context_window:
            tokens = min(tokens, self._context_window)
        output = settings["max_tokens"]
        return max(
            float(calc_price(Usage(input_tokens=tokens, cache_write_tokens=tokens, output_tokens=output),
                             model_ref=self.price_id, provider_id="aws").total_price),
            float(calc_price(Usage(input_tokens=tokens, output_tokens=output),
                             model_ref=self.price_id, provider_id="aws").total_price),
        )

    @asynccontextmanager
    async def _admitted(self, messages, settings, parameters) -> AsyncIterator[Callable[[Any], None]]:
        """Hold a slot and the request's worst case while it runs; yields a
        function that records what the request actually used."""
        async with self._slots or contextlib.nullcontext():
            key = (self.ledger.reserve(self.worst_case(messages, settings, parameters))
                   if self.ledger is not None else None)
            cost: list[float | None] = []

            def charge(usage: Any) -> None:
                cost.append(price(usage, self.price_id)[0])

            try:
                yield charge
            except BaseException as exc:
                # Refused before Bedrock ran it (a 4xx, throttling included):
                # nothing was spent. Anything else, a cancelled or broken
                # stream too, may have been billed, so it keeps its worst case.
                status = http_status(exc)
                if not cost and status is not None and 400 <= status < 500:
                    cost.append(0.0)
                raise
            finally:
                if key is not None:
                    self.ledger.settle(key, cost[0] if cost else None)

    async def request(self, messages: list[ModelMessage], model_settings: ModelSettings | None,
                      model_request_parameters: ModelRequestParameters) -> ModelResponse:
        settings = self._bounded(model_settings)
        async with self._admitted(messages, settings, model_request_parameters) as charge:
            response = await self.wrapped.request(messages, settings, model_request_parameters)
            charge(response.usage)
            return response

    @asynccontextmanager
    async def request_stream(self, messages: list[ModelMessage], model_settings: ModelSettings | None,
                             model_request_parameters: ModelRequestParameters,
                             run_context: Any = None) -> AsyncIterator[StreamedResponse]:
        settings = self._bounded(model_settings)
        async with self._admitted(messages, settings, model_request_parameters) as charge:
            async with self.wrapped.request_stream(messages, settings, model_request_parameters,
                                                   run_context) as stream:
                yield stream
            charge(stream.usage)


def ledger(log_dir: Path) -> SpendLedger | None:
    limit = monthly_spend_usd()
    return SpendLedger(log_dir / "spend.json", limit) if limit is not None else None


def describe() -> str:
    """The limits in force, for the startup log."""
    parts = [f"spend ${v:g}/month" if (v := monthly_spend_usd()) is not None else "no spend limit",
             f"{v} turns/chat" if (v := turns_per_chat()) is not None else "no turn limit",
             f"{v:g} s/turn" if (v := turn_seconds()) is not None else "no turn deadline",
             f"{v} model requests at once" if (v := model_concurrency()) is not None
             else "no model concurrency limit"]
    return ", ".join(parts)
