"""Application configuration with conservative resource limits."""

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
    def openai_configured(self) -> bool:
        """Report credential presence without exposing its value."""
        return bool(os.environ.get("OPENAI_API_KEY"))

    @property
    def openai_api_key(self) -> str:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not available in this process.")
        return key

    @property
    def openai_base_url(self) -> str | None:
        return os.environ.get("OPENAI_BASE_URL") or None


SETTINGS = Settings()
