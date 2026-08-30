"""Local Parse artifact generation from source images and grounded OCR evidence."""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from markdown_it import MarkdownIt
from PIL import Image, ImageDraw

from agentic_extractor.costs import rate_assumptions
from agentic_extractor.ocr import LocalParseResult
from agentic_extractor.parse import build_layout_chunks

_PAGE_MARKER_PATTERN = re.compile(r"<!--\s*page\s*:?\s*(\d+)\s*-->", re.IGNORECASE)
_HTML_TABLE_PATTERN = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class LocalArtifacts:
    markdown: bytes
    parse_result: bytes
    annotated_pdf: bytes
    html: bytes
    manifest: dict[str, Any]
    checkbox_crops: dict[str, bytes]


def build_local_artifacts(result: LocalParseResult) -> LocalArtifacts:
    """Generate selected-page-only Markdown, annotated PDF, HTML, and manifest."""
    markdown = result.markdown.encode("utf-8")
    parse_result = json.dumps(_structured_parse(result), indent=2).encode("utf-8")
    annotated_pdf = _annotated_pdf(result)
    standalone_html = _standalone_html(result).encode("utf-8")
    checkbox_crops = _checkbox_crops(result)
    artifacts = {
        "markdown": _artifact_entry("document.md", markdown),
        "parse_result": _artifact_entry("parse-result.json", parse_result),
        "annotated_pdf": _artifact_entry("annotated.pdf", annotated_pdf),
        "html": _artifact_entry("document.html", standalone_html),
        "checkbox_crops": {
            name: _artifact_entry(name, data) for name, data in checkbox_crops.items()
        },
    }
    processing: dict[str, Any] = {
        "requested_mode": result.requested_mode.value,
        "effective_mode": result.effective_mode.value,
        "routing": result.routing,
    }
    usage_and_cost: dict[str, Any] = {
        "rapidocr": {
            "processing_seconds": result.timings.get("ocr_seconds"),
            "api_cost_usd": 0.0,
            "hardware_cost_usd": None,
        }
    }
    has_gpt = result.cloud_output is not None or result.usage.call_count > 0
    if has_gpt:
        processing.update(
            {
                "gpt_context_pages": result.cloud_pages,
                "gpt_image_pages": result.cloud_image_pages,
                "gpt_model": "gpt-5.6-luna",
                "reasoning_effort": "medium",
            }
        )
        usage_and_cost["gpt"] = result.usage.model_dump(mode="json")
    manifest = {
        "manifest_version": 3,
        "source": result.document_metadata,
        "selected_pages": result.selected_pages,
        "selected_page_range": {
            "start": min(result.selected_pages),
            "end": max(result.selected_pages),
        },
        "engine": {
            "name": result.engine.name,
            "version": result.engine.version,
            "device": result.engine.device,
            "confidence_calibrated": result.engine.confidence_calibrated,
            "model_metadata": result.engine.model_metadata,
        },
        "timings": result.timings,
        "quality_diagnostics": result.quality_diagnostics,
        "adaptive_processing": result.adaptive_processing,
        "processing": processing,
        "usage_and_cost": usage_and_cost,
        "agent_workflow": result.workflow_manifest,
        "ocr_attempts": result.ocr_attempts,
        "gpt_attempts": result.cloud_attempts,
        "refinement_layer": [item.model_dump(mode="json") for item in result.refinements],
        "checkboxes": [item.model_dump(mode="json") for item in result.checkboxes],
        "checkbox_corrections": [
            item.model_dump(mode="json") for item in result.checkbox_corrections
        ],
        "warnings": result.warnings,
        "failed_pages": result.failed_pages,
        "pages": _page_manifest(result),
        "artifacts": artifacts,
    }
    if has_gpt:
        manifest["gpt_rate_assumptions"] = rate_assumptions()
    return LocalArtifacts(
        markdown, parse_result, annotated_pdf, standalone_html, manifest, checkbox_crops
    )


def build_local_bundle(result: LocalParseResult) -> bytes:
    artifacts = build_local_artifacts(result)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("document.md", artifacts.markdown)
        archive.writestr("parse-result.json", artifacts.parse_result)
        archive.writestr("annotated.pdf", artifacts.annotated_pdf)
        archive.writestr("document.html", artifacts.html)
        for name, data in artifacts.checkbox_crops.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", json.dumps(artifacts.manifest, indent=2))
    return output.getvalue()


def _structured_parse(result: LocalParseResult) -> dict[str, Any]:
    blocks = [
        _canonical_block(block, order, result.engine.confidence_calibrated)
        for page in result.pages
        for order, block in enumerate(page.blocks, 1)
    ]
    chunks = [
        _canonical_chunk(chunk)
        for page in result.pages
        for chunk in (page.chunks or build_layout_chunks(page.blocks))
    ]
    return {
        "contract_version": 3,
        "document_metadata": result.document_metadata,
        "selected_pages": result.selected_pages,
        "markdown": result.markdown,
        "engine": {
            "name": result.engine.name,
            "version": result.engine.version,
            "device": result.engine.device,
            "confidence_calibrated": result.engine.confidence_calibrated,
            "model_metadata": result.engine.model_metadata,
        },
        "timings": result.timings,
        "warnings": result.warnings,
        "failed_pages": result.failed_pages,
        "ocr_attempts": result.ocr_attempts,
        "gpt_attempts": result.cloud_attempts,
        "refinement_layer": [item.model_dump(mode="json") for item in result.refinements],
        "checkboxes": [item.model_dump(mode="json") for item in result.checkboxes],
        "checkbox_corrections": [
            item.model_dump(mode="json") for item in result.checkbox_corrections
        ],
        "pages": [
            {
                "source_page": page.page,
                "width": page.width,
                "height": page.height,
                "status": page.status,
                "blocks": [
                    _canonical_block(block, order, result.engine.confidence_calibrated)
                    for order, block in enumerate(page.blocks, 1)
                ],
                "raw_evidence": page.raw_evidence,
                "layout_signals": page.layout_signals,
                "timings": {
                    "ocr_seconds": page.ocr_seconds,
                    "engine_seconds": page.engine_elapsed_seconds,
                    "stages": page.stage_timings,
                },
                "warnings": page.warnings,
            }
            for page in result.pages
        ],
        "chunks": chunks,
        "coordinate_spaces": {
            "bbox": "normalized_page_xyxy",
            "polygon": "source_page_pixels",
        },
        "grounding": {
            item["source_id"]: {
                "page": item["page"],
                "bbox": item["bbox"],
                "polygon": item.get("polygon"),
            }
            for item in [*blocks, *chunks]
        },
    }


def _canonical_block(block: Any, reading_order: int, calibrated: bool) -> dict[str, Any]:
    """Serialize a block without losing its raw RapidOCR score or grounding."""
    value = block.model_dump(mode="json")
    value.update(
        {
            "source_id": block.id,
            "reading_order": reading_order,
            "confidence": {
                "value": block.ocr_score,
                "engine": block.source,
                "calibrated": calibrated,
            },
        }
    )
    return value


def _canonical_chunk(chunk: Any) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "source_id": chunk.id,
        "page": chunk.page,
        "type": chunk.type,
        "reading_order": chunk.reading_order,
        "text": chunk.text,
        "markdown": chunk.markdown,
        "source_block_ids": chunk.source_block_ids,
        "bbox": chunk.bbox,
        "polygon": None,
        "raw_scores": chunk.raw_scores,
    }


def _annotated_pdf(result: LocalParseResult) -> bytes:
    pages: list[Image.Image] = []
    for page in result.pages:
        image = Image.open(io.BytesIO(page.original_image_bytes or page.image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        for region in _grounded_regions(page):
            points = region["polygon"]
            draw.line(points + [points[0]], fill="#D72D7A", width=max(2, image.width // 600))
            score = "?" if region["confidence"] is None else f"{region['confidence']:.2f}"
            draw.text(
                points[0],
                f"{region['id']} · {score} · {region['engine']}",
                fill="#00A8B8",
            )
        for checkbox in (item for item in result.checkboxes if item.page == page.page):
            left, top, right, bottom = checkbox.control_bbox
            points = [
                (left * image.width, top * image.height),
                (right * image.width, top * image.height),
                (right * image.width, bottom * image.height),
                (left * image.width, bottom * image.height),
            ]
            draw.line(points + [points[0]], fill="#00A8B8", width=max(3, image.width // 500))
            draw.text(
                points[0],
                f"{checkbox.id} · {checkbox.state.value} · {checkbox.decision_status}",
                fill="#D72D7A",
            )
        pages.append(image)
    if not pages:
        raise ValueError("Cannot generate an annotated PDF without successful pages.")
    output = io.BytesIO()
    pages[0].save(output, "PDF", save_all=True, append_images=pages[1:], resolution=150)
    return output.getvalue()


def _standalone_html(result: LocalParseResult) -> str:
    renderer = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")
    markdown_pages = _markdown_by_page(result.markdown)
    page_html: list[str] = []
    for page in result.pages:
        evidence: list[str] = []
        for order, region in enumerate(_grounded_regions(page), 1):
            confidence = (
                "unknown" if region["confidence"] is None else f"{region['confidence']:.4f}"
            )
            bbox = ", ".join(f"{coordinate:.4f}" for coordinate in region["bbox"])
            evidence.append(
                f'<li data-grounding-id="{html.escape(region["id"])}" '
                f'data-block-type="{html.escape(region["type"])}" data-reading-order="{order}" '
                f'data-bbox="{bbox}"><code>{html.escape(region["id"])}</code> · '
                f"{html.escape(region['type'])} · confidence {confidence} · "
                f"bbox [{bbox}]</li>"
            )
        for checkbox in (item for item in result.checkboxes if item.page == page.page):
            bbox = ", ".join(f"{coordinate:.4f}" for coordinate in checkbox.control_bbox)
            evidence.append(
                f'<li class="checkbox" '
                f'data-checkbox-id="{html.escape(checkbox.id)}" '
                f'data-checkbox-state="{html.escape(checkbox.state.value)}" '
                f'data-bbox="{bbox}"><code>{html.escape(checkbox.id)}</code> · '
                f"{html.escape(checkbox.label)} · {html.escape(checkbox.state.value)} · "
                f"bbox [{bbox}]</li>"
            )
        fallback = result.markdown if len(result.pages) == 1 else ""
        body = renderer.render(markdown_for_display(markdown_pages.get(page.page, fallback)))
        evidence_html = "".join(evidence) or "<li>No grounded regions available.</li>"
        page_html.append(
            f'<section class="page" data-source-page="{page.page}" '
            f'data-page-width="{page.width}" data-page-height="{page.height}" '
            f'aria-label="Document page {page.page}"><header>Page {page.page}</header>'
            f'<article class="markdown">{body}</article>'
            f'<details class="grounding"><summary>Grounding context</summary>'
            f"<ol>{evidence_html}</ol></details></section>"
        )
    return (
        """<!doctype html><html><head><meta charset="utf-8"><title>Document</title>
<style>
body { margin:0; padding:24px; background:#111318; color:#171922;
       font:16px/1.5 system-ui,sans-serif; }
main { max-width:960px; margin:auto; }
.page { box-sizing:border-box; min-height:900px; margin:0 auto 28px; padding:48px 56px;
        background:#fff; box-shadow:0 2px 8px #0005; }
.page>header { color:#687083; font-size:12px; border-bottom:1px solid #d8dbe3;
               margin-bottom:24px; padding-bottom:8px; }
.markdown { overflow-wrap:anywhere; }
.markdown table { border-collapse:collapse; width:100%; }
.markdown th,.markdown td { border:1px solid #aeb4c0; padding:6px 8px; text-align:left; }
.markdown pre { overflow:auto; padding:12px; background:#f2f3f6; }
.grounding { margin-top:36px; border-top:1px solid #d8dbe3; padding-top:12px; color:#4e5668; }
.grounding li { margin:4px 0; }
@media print { body { padding:0; background:#fff; }
               .page { box-shadow:none; page-break-after:always; }
               .grounding { display:none; } }
</style></head><body><main>"""
        + "".join(page_html)
        + "</main></body></html>"
    )


def _markdown_by_page(markdown: str) -> dict[int, str]:
    markers = list(_PAGE_MARKER_PATTERN.finditer(markdown))
    if not markers:
        return {1: markdown}
    return {
        int(match.group(1)): markdown[match.end() : markers[index + 1].start()].strip()
        for index, match in enumerate(markers)
        if index + 1 < len(markers)
    } | {int(markers[-1].group(1)): markdown[markers[-1].end() :].strip()}


def markdown_for_display(markdown: str) -> str:
    """Normalize layout Markdown for safe rendering without changing raw evidence."""

    def replace_table(match: re.Match[str]) -> str:
        parser = _TableParser()
        parser.feed(match.group(0))
        parser.close()
        return parser.as_markdown()

    rendered = _HTML_TABLE_PATTERN.sub(replace_table, markdown)
    rendered = _PAGE_MARKER_PATTERN.sub(r"\n\n---\n\n#### Page \1\n", rendered)
    return rendered.strip()


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None
        self._colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell_parts = []
            try:
                self._colspan = max(1, int(dict(attrs).get("colspan", "1") or "1"))
            except ValueError:
                self._colspan = 1
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._row is not None and self._cell_parts is not None:
            cell = " ".join("".join(self._cell_parts).split()).replace("|", r"\|")
            self._row.extend([cell, *([""] * (self._colspan - 1))])
            self._cell_parts = None
            self._colspan = 1
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def as_markdown(self) -> str:
        if not self.rows:
            return ""
        column_count = max(len(row) for row in self.rows)
        rows = [row + [""] * (column_count - len(row)) for row in self.rows]
        lines = [_markdown_row(rows[0]), _markdown_row(["---"] * column_count)]
        lines.extend(_markdown_row(row) for row in rows[1:])
        return "\n\n" + "\n".join(lines) + "\n\n"


def _markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _grounded_regions(page: Any) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for block in page.blocks:
        if not block.bbox:
            continue
        left, top, right, bottom = block.bbox
        polygon = (
            [(float(point[0]), float(point[1])) for point in block.polygon]
            if block.polygon
            else [
                (left * page.width, top * page.height),
                (right * page.width, top * page.height),
                (right * page.width, bottom * page.height),
                (left * page.width, bottom * page.height),
            ]
        )
        regions.append(
            {
                "id": block.id,
                "text": block.text,
                "type": block.type,
                "bbox": block.bbox,
                "polygon": polygon,
                "confidence": block.ocr_score,
                "engine": block.source,
            }
        )
    return regions


def _artifact_entry(name: str, data: bytes) -> dict[str, object]:
    return {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _checkbox_crops(result: LocalParseResult) -> dict[str, bytes]:
    pages = {page.page: page for page in result.pages}
    crops: dict[str, bytes] = {}
    for checkbox in result.checkboxes:
        page = pages.get(checkbox.page)
        if page is None or not checkbox.crop_ref:
            continue
        image = Image.open(io.BytesIO(page.original_image_bytes or page.image_bytes)).convert("RGB")
        left, top, right, bottom = checkbox.control_bbox
        width = max(right - left, 0.01)
        height = max(bottom - top, 0.01)
        pad_x, pad_y = max(width * 2, 0.02), max(height * 2, 0.02)
        crop = image.crop(
            (
                max(0, int((left - pad_x) * image.width)),
                max(0, int((top - pad_y) * image.height)),
                min(image.width, int((right + pad_x) * image.width)),
                min(image.height, int((bottom + pad_y) * image.height)),
            )
        )
        output = io.BytesIO()
        crop.save(output, "JPEG", quality=95, optimize=True)
        crops[checkbox.crop_ref] = output.getvalue()
    return crops


def _page_manifest(result: LocalParseResult) -> list[dict[str, object]]:
    parsed = {page.page: page for page in result.pages}
    pages: list[dict[str, object]] = []
    for number in result.selected_pages:
        page = parsed.get(number)
        pages.append(
            {
                "source_page": number,
                "status": result.page_statuses.get(number, "completed" if page else "failed"),
                "width": page.width if page else None,
                "height": page.height if page else None,
                "block_count": len(page.blocks) if page else 0,
                "timings": {
                    "wall_seconds": page.ocr_seconds if page else None,
                    "engine_seconds": page.engine_elapsed_seconds if page else None,
                    "stages": page.stage_timings if page else {},
                },
                "layout_signals": page.layout_signals if page else {},
                "warnings": page.warnings if page else [],
            }
        )
    return pages
