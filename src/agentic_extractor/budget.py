"""Fail-closed reservations for explicitly authorized, bounded live validation."""

import json
from decimal import Decimal
from pathlib import Path

from agentic_extractor.costs import LONG_PROMPT_THRESHOLD


class RequestBudget:
    def __init__(self, limit_usd: float = 5.0) -> None:
        if not 0 < limit_usd <= 5:
            raise ValueError("Live validation budget must be greater than zero and at most $5.")
        self.limit = Decimal(str(limit_usd))
        self.reservations: list[dict] = []
        self.uncertain = False
        self.ledger_path: Path | None = None

    def _persist(self) -> None:
        if self.ledger_path is not None:
            temporary = self.ledger_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "limit_usd": str(self.limit),
                        "reservations": self.reservations,
                        "billing_unknown": self.uncertain,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary.replace(self.ledger_path)

    def reserve(self, input_tokens: int, max_output_tokens: int) -> dict:
        if self.uncertain:
            raise RuntimeError("Prior request billing is unknown; no more paid calls allowed.")
        if input_tokens < 0 or max_output_tokens < 1:
            raise ValueError("Invalid token reservation.")
        long = input_tokens > LONG_PROMPT_THRESHOLD
        # Treat every input token as the most expensive cache-write token.
        amount = (
            Decimal(input_tokens) * Decimal("2.50") * (2 if long else 1)
            + Decimal(max_output_tokens) * 10 * (Decimal("1.5") if long else 1)
        ) / 1_000_000
        used = sum((Decimal(item["reserved_usd"]) for item in self.reservations), Decimal(0))
        if used + amount > self.limit:
            raise RuntimeError("Request exceeds the remaining authorized live-validation budget.")
        entry = {
            "input_tokens": input_tokens,
            "max_output_tokens": max_output_tokens,
            "reserved_usd": str(amount),
            "status": "pending",
            "reported_usd": None,
        }
        self.reservations.append(entry)
        self._persist()
        return entry

    def settle(self, entry: dict, usage) -> None:
        entry["reported_usd"] = usage.total_cost_usd
        entry["status"] = usage.cost_status
        if usage.cost_status == "exact" and usage.total_cost_usd is not None:
            entry["reserved_usd"] = str(usage.total_cost_usd)
        elif usage.total_cost_usd is None:
            self.uncertain = True
        # Incomplete cache detail keeps the original worst-case reservation.
        self._persist()
