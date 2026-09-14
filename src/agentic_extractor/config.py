"""Application configuration with conservative resource limits.

Responsible for: process-wide limits (upload size, pages, pixels), fixed
model/reasoning policy, and the one user-tunable knob (OCR worker cap) read
from the environment. Must not: read or expose the OpenAI credential value
anywhere but inside a request to the OpenAI client, or silently clamp an
invalid `ADE_OCR_MAX_WORKERS` instead of raising. Next: costs.py for the
pricing that pairs with `model`/`reasoning_effort`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    max_upload_bytes: int = 50 * 1024 * 1024
    max_pages: int = 200
    max_image_pixels: int = 25_000_000
    render_dpi: int = 150
    job_ttl_seconds: int = 3600
    model: str = "gpt-5.6-luna"
    reasoning_effort: str = "medium"
    cloud_batch_characters: int = 80_000

    @property
    def ocr_max_workers(self) -> int:
        # Fail loudly on a bad override rather than clamping: a silently
        # clamped value would let a misconfigured deployment run with a
        # different concurrency cap than the operator intended.
        raw = os.environ.get("ADE_OCR_MAX_WORKERS", "4")
        try:
            value = int(raw)
        except ValueError as exc:
            raise RuntimeError("ADE_OCR_MAX_WORKERS must be an integer from 1 through 4.") from exc
        if not 1 <= value <= 4:
            raise RuntimeError("ADE_OCR_MAX_WORKERS must be between 1 and 4.")
        return value

    @property
    def openai_configured(self) -> bool:
        """Report credential presence without exposing its value.

        Security: this is a presence check only, used for UI/diagnostics
        messaging. It must never be changed to return or log the key itself.
        """
        return bool(os.environ.get("OPENAI_API_KEY"))

    @property
    def openai_api_key(self) -> str:
        # Only place the key is read out of the process environment; callers
        # pass it straight to the OpenAI client and must not persist or log it.
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not available in this process.")
        return key

    @property
    def openai_base_url(self) -> str | None:
        return os.environ.get("OPENAI_BASE_URL") or None


SETTINGS = Settings()
