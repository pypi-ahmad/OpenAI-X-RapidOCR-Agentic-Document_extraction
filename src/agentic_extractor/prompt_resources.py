"""Strict loading and rendering for versioned Markdown prompt resources."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from string import Template

PROMPT_PACKAGE = "agentic_extractor.prompts"
_VERSION = re.compile(r"<!--\s*prompt-version:\s*([^\s]+)\s*-->")


@dataclass(frozen=True, slots=True)
class PromptResource:
    name: str
    version: str
    text: str
    sha256: str


@lru_cache(maxsize=32)
def load_prompt(name: str) -> PromptResource:
    """Load a packaged Markdown prompt and fail closed when metadata is absent."""
    if not name.endswith(".md") or "/" in name or "\\" in name:
        raise ValueError("Prompt name must be a Markdown filename without a path.")
    resource = files(PROMPT_PACKAGE).joinpath(name)
    try:
        source = resource.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise RuntimeError(f"Required prompt resource {name!r} is unavailable.") from exc
    match = _VERSION.search(source)
    if match is None:
        raise RuntimeError(f"Prompt resource {name!r} has no prompt-version metadata.")
    return PromptResource(
        name=name,
        version=match.group(1),
        text=_VERSION.sub("", source, count=1).strip(),
        sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def render_prompt(name: str, **values: object) -> tuple[str, PromptResource]:
    """Render one prompt using strict placeholders and return its audit metadata."""
    resource = load_prompt(name)
    try:
        rendered = Template(resource.text).substitute(
            {key: str(value) for key, value in values.items()}
        )
    except KeyError as exc:
        raise RuntimeError(f"Prompt resource {name!r} is missing value {exc.args[0]!r}.") from exc
    return rendered, resource
