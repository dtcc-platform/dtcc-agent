"""Pricing pydantic-ai turns (#86). genai-prices is the source; these tests
pin the rates M3 is measured with, so a change in its data fails here instead
of quietly moving the cost column of a gate run."""

import pytest
from pydantic_ai.usage import RunUsage

from chatbot.prices import price

M = 1_000_000


@pytest.mark.parametrize("model, rates", [
    # Sonnet 5.5 resolves to the regional Sonnet 5 entry. Checked against the
    # CLI's own total_cost_usd over the 60 baseline turns: within 2% (#86).
    ("eu.anthropic.claude-sonnet-5-5", (2.20, 0.22, 2.75, 11.00)),
    # Sonnet 5, T38's model-switch run: the account refuses every 4.x model.
    ("eu.anthropic.claude-sonnet-5", (2.20, 0.22, 2.75, 11.00)),
    ("eu.anthropic.claude-sonnet-4-6", (3.30, 0.33, 4.125, 16.50)),
])
def test_each_model_is_priced_per_token_class(model, rates):
    fresh, read, write, out = rates
    for usage, expected in [
        (RunUsage(input_tokens=M), fresh),
        (RunUsage(input_tokens=M, cache_read_tokens=M), read),
        (RunUsage(input_tokens=M, cache_write_tokens=M), write),
        (RunUsage(output_tokens=M), out),
    ]:
        cost, source = price(usage, model)
        assert cost == pytest.approx(expected)
        assert source.startswith("genai-prices ")


def test_the_source_names_the_price_entry_it_used():
    _, source = price(RunUsage(input_tokens=10), "eu.anthropic.claude-sonnet-5-5")
    assert "regional.anthropic.claude-sonnet-5-v1:0" in source


def test_an_unknown_model_is_unpriced_not_free():
    assert price(RunUsage(input_tokens=10), "eu.example.not-a-model") == (None, "unpriced")
