"""Portable ZIP export for complete and diagnostic results."""

from __future__ import annotations

import io
import json
import zipfile

from pypdf import PdfReader, PdfWriter

from agentic_extractor.models import DocumentResult


def build_result_zip(result: DocumentResult, original_pdf: bytes | None = None) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("document.md", result.markdown)
        archive.writestr("result.json", result.model_dump_json(indent=2))
        archive.writestr(
            "blocks.jsonl", "\n".join(block.model_dump_json() for block in result.blocks)
        )
        archive.writestr("metadata.json", json.dumps(result.metadata, indent=2, default=str))
        if result.extraction:
            archive.writestr("extraction.json", result.extraction.model_dump_json(indent=2))
        if original_pdf and result.splits:
            reader = PdfReader(io.BytesIO(original_pdf))
            for index, split in enumerate(result.splits, 1):
                writer = PdfWriter()
                for page_index in range(split.page_start - 1, split.page_end):
                    writer.add_page(reader.pages[page_index])
                output = io.BytesIO()
                writer.write(output)
                archive.writestr(
                    f"splits/{index:02d}-{_safe_name(split.name)}.pdf", output.getvalue()
                )
    return target.getvalue()


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-" for character in value
    )
    return cleaned.strip("-")[:80] or "document"
