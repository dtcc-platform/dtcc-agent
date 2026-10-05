"""What a pydantic-ai turn cost (#86).

The Agent SDK reported its own total_cost_usd; pydantic-ai reports tokens.
genai-prices turns those into dollars on Bedrock's prices. Its rates for the
models M3 measures are pinned in tests/test_chatbot_prices.py, so a change in
its data fails a test instead of moving a gate run's cost column.
"""

from __future__ import annotations

import importlib.metadata

from genai_prices import Usage, calc_price
from pydantic_ai.usage import RunUsage

_VERSION = importlib.metadata.version("genai-prices")


def price(usage: RunUsage, model_id: str) -> tuple[float | None, str]:
    """(USD, where the price came from), or (None, "unpriced") for a model
    genai-prices does not know. Cache writes are priced at the 5-minute rate,
    the TTL our cache settings use."""
    try:
        result = calc_price(
            Usage(input_tokens=usage.input_tokens, cache_read_tokens=usage.cache_read_tokens,
                  cache_write_tokens=usage.cache_write_tokens, output_tokens=usage.output_tokens),
            model_ref=model_id, provider_id="aws",
        )
    except LookupError:
        return None, "unpriced"
    return float(result.total_price), f"genai-prices {_VERSION} ({result.model.id})"
