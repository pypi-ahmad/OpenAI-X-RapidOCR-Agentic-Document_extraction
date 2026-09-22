"""Central gpt-6-sol rates and usage-based cost calculation.

Responsible for: converting provider-reported token usage into a cost
breakdown using the rates below, and aggregating per-call records for the
usage panel. Must not: fabricate a cost when usage is missing or internally
inconsistent (must return status "unavailable" instead), or treat these
rates as anything but application policy — bump them here (and in tests)
when pricing changes, not by patching values at call sites. Next:
artifacts.py, which folds `rate_assumptions()` into the exported manifest.
"""

from dataclasses import asdict, dataclass
from typing import Literal

from agentic_extractor.config import MODEL_NAME as MODEL_NAME
from agentic_extractor.config import REASONING_EFFORT as REASONING_EFFORT
from agentic_extractor.models import UsageRecord

LONG_PROMPT_THRESHOLD = 272_000
UNCACHED_INPUT_USD_PER_MILLION = 2.00
CACHED_INPUT_USD_PER_MILLION = 0.20
CACHE_WRITE_INPUT_USD_PER_MILLION = 2.50
OUTPUT_USD_PER_MILLION = 10.00


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    text_input_tokens: int | None = None
    image_input_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    uncached_input_tokens: int | None
    input_cost_usd: float | None
    output_cost_usd: float | None
    total_cost_usd: float | None
    input_multiplier: float
    output_multiplier: float
    status: Literal["exact", "estimate", "unavailable"]


def calculate_usage_cost(usage: TokenUsage) -> CostBreakdown:
    """Calculate cost and flag incomplete API usage instead of fabricating it."""
    # Strictly greater-than: a request at exactly LONG_PROMPT_THRESHOLD tokens
    # does NOT get the multiplier. Keep this boundary exclusive on the low
    # side if it is ever revisited; it mirrors the provider's own billing cutoff.
    long_prompt = usage.input_tokens is not None and usage.input_tokens > LONG_PROMPT_THRESHOLD
    input_multiplier = 2.0 if long_prompt else 1.0
    output_multiplier = 1.5 if long_prompt else 1.0
    # Any missing or internally inconsistent field (negative counts, cached/
    # cache-write tokens exceeding total input) makes the whole cost
    # "unavailable" rather than partially computed — a wrong displayed cost
    # is worse than an honest blank, since this feeds a user-facing dollar figure.
    if (
        usage.input_tokens is None
        or usage.output_tokens is None
        or usage.input_tokens < 0
        or usage.output_tokens < 0
        or (usage.cached_input_tokens is not None and usage.cached_input_tokens < 0)
        or (usage.cache_write_input_tokens is not None and usage.cache_write_input_tokens < 0)
        or (
            usage.cached_input_tokens is not None and usage.cached_input_tokens > usage.input_tokens
        )
        or (
            (usage.cached_input_tokens or 0) + (usage.cache_write_input_tokens or 0)
            > usage.input_tokens
        )
    ):
        return CostBreakdown(
            None, None, None, None, input_multiplier, output_multiplier, "unavailable"
        )
    cached = usage.cached_input_tokens or 0
    cache_write = usage.cache_write_input_tokens or 0
    uncached = usage.input_tokens - cached - cache_write
    input_cost = (
        uncached / 1_000_000 * UNCACHED_INPUT_USD_PER_MILLION
        + cached / 1_000_000 * CACHED_INPUT_USD_PER_MILLION
        + cache_write / 1_000_000 * CACHE_WRITE_INPUT_USD_PER_MILLION
    ) * input_multiplier
    output_cost = usage.output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION * output_multiplier
    return CostBreakdown(
        uncached,
        input_cost,
        output_cost,
        input_cost + output_cost,
        input_multiplier,
        output_multiplier,
        (
            # "estimate" (not "unavailable") when totals are valid but the
            # cached/cache-write breakdown wasn't reported. This estimate is
            # not an upper bound: unreported cache writes can cost more.
            "exact"
            if usage.cached_input_tokens is not None and usage.cache_write_input_tokens is not None
            else "estimate"
        ),
    )


def rate_assumptions() -> dict[str, object]:
    return {
        "model": MODEL_NAME,
        "reasoning_effort": REASONING_EFFORT,
        "usd_per_million_tokens": {
            "uncached_input": UNCACHED_INPUT_USD_PER_MILLION,
            "cached_input": CACHED_INPUT_USD_PER_MILLION,
            "cache_write_input": CACHE_WRITE_INPUT_USD_PER_MILLION,
            "output": OUTPUT_USD_PER_MILLION,
        },
        "cache_write_pricing": "priced separately when provider usage reports it",
        "long_prompt": {
            "threshold_input_tokens": LONG_PROMPT_THRESHOLD,
            "input_multiplier": 2.0,
            "output_multiplier": 1.5,
            "scope": "whole request",
        },
    }


def usage_and_cost_dict(usage: TokenUsage) -> dict[str, object]:
    return {"tokens": asdict(usage), "cost": asdict(calculate_usage_cost(usage))}


def aggregate_usage(records: list[UsageRecord]) -> UsageRecord:
    """Aggregate complete records without turning unavailable values into zeroes."""
    if not records:
        return UsageRecord()

    def total(field: str):
        values = [getattr(record, field) for record in records]
        return sum(values) if all(value is not None for value in values) else None

    statuses = {record.cost_status for record in records}
    status = (
        "unavailable"
        if "unavailable" in statuses
        else ("estimate" if "estimate" in statuses else "exact")
    )
    return UsageRecord(
        call_count=sum(record.call_count for record in records),
        input_tokens=total("input_tokens"),
        cached_input_tokens=total("cached_input_tokens"),
        cache_write_input_tokens=total("cache_write_input_tokens"),
        text_input_tokens=total("text_input_tokens"),
        image_input_tokens=total("image_input_tokens"),
        output_tokens=total("output_tokens"),
        reasoning_tokens=total("reasoning_tokens"),
        total_tokens=total("total_tokens"),
        cloud_seconds=sum(record.cloud_seconds for record in records),
        input_cost_usd=total("input_cost_usd"),
        output_cost_usd=total("output_cost_usd"),
        total_cost_usd=total("total_cost_usd"),
        cost_status=status,
        calls=[call for record in records for call in record.calls],
    )
