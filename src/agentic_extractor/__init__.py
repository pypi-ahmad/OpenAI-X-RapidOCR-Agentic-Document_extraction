"""Agentic document extraction with local OCR and cloud refinement.

`__all__` is the package's public surface for external/programmatic callers.
It intentionally re-exports only the stable entry points and their result
types (`process_document`/`process_hybrid_document`, `DocumentRequest`/
`DocumentResult`, `LocalParseResult`) — it must not grow into a dumping
ground for every internal helper as new modules are added. Next:
`pipeline.py`, which implements the functions re-exported here."""

from agentic_extractor.models import DocumentRequest, DocumentResult
from agentic_extractor.ocr import LocalParseResult
from agentic_extractor.pipeline import (
    process_document,
    process_hybrid_document,
)

__all__ = [
    "DocumentRequest",
    "DocumentResult",
    "LocalParseResult",
    "process_document",
    "process_hybrid_document",
]
