from agentic_extractor.costs import (
    TokenUsage,
    aggregate_usage,
    calculate_usage_cost,
    rate_assumptions,
)
from agentic_extractor.models import UsageRecord


def test_cost_uses_supplied_uncached_cached_and_output_formula() -> None:
    cost = calculate_usage_cost(
        TokenUsage(
            input_tokens=100_000,
            cached_input_tokens=25_000,
            cache_write_input_tokens=10_000,
            output_tokens=10_000,
        )
    )
    assert cost.uncached_input_tokens == 65_000
    assert cost.input_cost_usd == 0.016
    assert cost.output_cost_usd == 0.012
    assert cost.total_cost_usd == 0.028
    assert cost.status == "exact"


def test_rate_assumptions_include_separate_cache_write_price() -> None:
    rates = rate_assumptions()["usd_per_million_tokens"]
    assert rates == {
        "uncached_input": 0.20,
        "cached_input": 0.02,
        "cache_write_input": 0.25,
        "output": 1.20,
    }


def test_long_prompt_multiplier_applies_to_whole_request() -> None:
    cost = calculate_usage_cost(
        TokenUsage(
            input_tokens=300_000,
            cached_input_tokens=0,
            cache_write_input_tokens=0,
            output_tokens=10_000,
        )
    )
    assert cost.input_multiplier == 2
    assert cost.output_multiplier == 1.5
    assert cost.total_cost_usd == 0.138


def test_missing_cached_usage_is_estimated_and_missing_totals_are_unavailable() -> None:
    estimate = calculate_usage_cost(TokenUsage(input_tokens=100, output_tokens=50))
    assert estimate.status == "estimate"
    assert estimate.total_cost_usd is not None
    unavailable = calculate_usage_cost(TokenUsage(input_tokens=None, output_tokens=50))
    assert unavailable.status == "unavailable"
    assert unavailable.total_cost_usd is None

    complete_breakdown = calculate_usage_cost(
        TokenUsage(
            input_tokens=100,
            cached_input_tokens=0,
            cache_write_input_tokens=0,
            output_tokens=50,
        )
    )
    assert complete_breakdown.status == "exact"


def test_invalid_usage_is_unavailable() -> None:
    cost = calculate_usage_cost(
        TokenUsage(input_tokens=100, cached_input_tokens=101, output_tokens=1)
    )
    assert cost.status == "unavailable"
    assert cost.total_cost_usd is None

    overlapping_categories = calculate_usage_cost(
        TokenUsage(
            input_tokens=100,
            cached_input_tokens=60,
            cache_write_input_tokens=41,
            output_tokens=1,
        )
    )
    assert overlapping_categories.status == "unavailable"


def test_usage_aggregation_preserves_unavailable_values() -> None:
    first = UsageRecord(
        call_count=1,
        input_tokens=100,
        cached_input_tokens=10,
        output_tokens=20,
        total_tokens=120,
        input_cost_usd=0.0000182,
        output_cost_usd=0.000024,
        total_cost_usd=0.0000422,
        cost_status="exact",
        calls=[{"pages": [1]}],
    )
    second = UsageRecord(
        call_count=1,
        input_tokens=None,
        cached_input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        cost_status="unavailable",
        calls=[{"pages": [2]}],
    )

    total = aggregate_usage([first, second])

    assert total.call_count == 2
    assert total.input_tokens is None
    assert total.total_cost_usd is None
    assert total.cost_status == "unavailable"
    assert [call["pages"] for call in total.calls] == [[1], [2]]
