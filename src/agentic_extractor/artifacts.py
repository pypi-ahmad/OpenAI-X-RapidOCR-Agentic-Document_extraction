"""Local Parse artifact generation from source images and grounded OCR evidence.

Responsible for: building cheap artifacts eagerly (Markdown, Parse JSON,
manifest, checkbox crops) in `build_local_artifacts`, and providing thread-safe,
generate-once lazy access to expensive artifacts (annotated PDF, coordinate
HTML, and ZIP bundle) via `LocalArtifacts`.

Must not: mutate `LocalParseResult` or any raw OCR block, and must not generate
a lazy artifact merely because its metadata was inspected.

Next: `landing_contract.py` for the public Parse JSON projection, and
`layout_html.py` for the embedded coordinate HTML viewer.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import threading
import time
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, TypedDict

from PIL import Image, ImageDraw

from agentic_extractor.costs import rate_assumptions
from agentic_extractor.landing_contract import build_landing_parse
from agentic_extractor.layout_html import build_coordinate_html, coordinate_html_page_images
from agentic_extractor.ocr import LocalParseResult
from agentic_extractor.parse import (
    document_markdown,
    document_markdown_with_checkboxes,
    is_publishable_checkbox,
)
from agentic_extractor.timing import record_stage_timing, stage_timing_summary

_PAGE_MARKER_PATTERN = re.compile(r"<!--\s*page\s*:?\s*(\d+)\s*-->", re.IGNORECASE)
_HTML_TABLE_PATTERN = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)


class ArtifactMetadataValue(TypedDict):
    name: str
    bytes: int | None
    sha256: str | None
    generated: bool


@dataclass(slots=True)
class LocalArtifacts:
    markdown: bytes
    parse_result: bytes
    manifest: dict[str, Any]
    checkbox_crops: dict[str, bytes]
    _result: LocalParseResult = field(repr=False)
    _annotated_pdf_bytes: bytes | None = field(default=None, init=False, repr=False)
    _html_bytes: bytes | None = field(default=None, init=False, repr=False)
    _bundle_bytes: bytes | None = field(default=None, init=False, repr=False)
    _lock: Any = field(default_factory=threading.RLock, init=False, repr=False)

    # The three properties below generate their bytes exactly once, on first access, under
    # `_lock`. This is the concurrency invariant for this class: two callers racing to read
    # `annotated_pdf`/`html`/`bundle` for the same result must not double-generate or observe
    # a half-written manifest entry. Once `_*_bytes` is set it is never recomputed, so the
    # generated artifact is stable for the lifetime of this instance even if the underlying
    # `LocalParseResult` were (incorrectly) mutated afterward.
    @property
    def annotated_pdf(self) -> bytes:
        with self._lock:
            if self._annotated_pdf_bytes is None:
                started = time.perf_counter()
                self._annotated_pdf_bytes = _annotated_pdf(self._result)
                record_stage_timing(
                    self._result.timings,
                    "annotated_pdf_generation_seconds",
                    time.perf_counter() - started,
                )
                self.manifest["artifacts"]["annotated_pdf"] = _artifact_entry(
                    "annotated.pdf", self._annotated_pdf_bytes
                )
                self._refresh_timing_analysis()
            return self._annotated_pdf_bytes

    @property
    def html(self) -> bytes:
        with self._lock:
            if self._html_bytes is None:
                started = time.perf_counter()
                self._html_bytes = build_coordinate_html(self._result).encode("utf-8")
                self.manifest["html_layout"]["page_images"] = coordinate_html_page_images(
                    self._result
                )
                record_stage_timing(
                    self._result.timings,
                    "html_generation_seconds",
                    time.perf_counter() - started,
                )
                self.manifest["artifacts"]["html"] = _artifact_entry(
                    "document.html", self._html_bytes
                )
                self._refresh_timing_analysis()
            return self._html_bytes

    @property
    def bundle(self) -> bytes:
        with self._lock:
            if self._bundle_bytes is None:
                annotated_pdf = self.annotated_pdf
                html = self.html
                started = time.perf_counter()
                output = io.BytesIO()
                with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr("document.md", self.markdown)
                    archive.writestr("parse-result.json", self.parse_result)
                    archive.writestr("annotated.pdf", annotated_pdf)
                    archive.writestr("document.html", html)
                    for name, data in self.checkbox_crops.items():
                        archive.writestr(name, data)
                    record_stage_timing(
                        self._result.timings,
                        "zip_packaging_seconds",
                        time.perf_counter() - started,
                    )
                    self._refresh_timing_analysis()
                    archive.writestr("manifest.json", json.dumps(self.manifest, indent=2))
                self._bundle_bytes = output.getvalue()
            return self._bundle_bytes

    def _refresh_timing_analysis(self) -> None:
        self.manifest["timing_analysis"] = stage_timing_summary(self._result.timings)

    def artifact_metadata(self, name: str) -> ArtifactMetadataValue:
        """Return metadata without forcing a lazy artifact to be generated."""
        values = {
            "document.md": self.markdown,
            "parse-result.json": self.parse_result,
            "annotated.pdf": self._annotated_pdf_bytes,
            "document.html": self._html_bytes,
            "bundle.zip": self._bundle_bytes,
        }
        if name not in values:
            raise KeyError(name)
        data = values[name]
        return _artifact_entry(name, data) if data is not None else _pending_artifact_entry(name)

    def get(self, name: str) -> bytes:
        """Generate and return one allowlisted artifact."""
        if name == "document.md":
            return self.markdown
        if name == "parse-result.json":
            return self.parse_result
        if name == "annotated.pdf":
            return self.annotated_pdf
        if name == "document.html":
            return self.html
        if name == "bundle.zip":
            return self.bundle
        raise KeyError(name)


def build_local_artifacts(
    result: LocalParseResult, *, include_atomic_grounding: bool = True
) -> LocalArtifacts:
    """Generate selected-page-only Markdown, annotated PDF, HTML, and manifest."""
    started = time.perf_counter()
    export_result = _synchronize_accepted_tables(result)
    structured_parse = _structured_parse(
        export_result, include_atomic_grounding=include_atomic_grounding
    )
    expected_tables = sum(
        table.status == "valid" and not table.review_required and bool(table.markdown)
        for page in export_result.pages
        for table in page.table_structures
    )
    exported_tables = sum(
        child.get("type") == "table"
        for page in structured_parse["structure"]["children"]
        for child in page.get("children", [])
    )
    if exported_tables != expected_tables:
        raise ValueError(
            "Artifact table integrity failed: "
            f"{expected_tables} accepted tables but {exported_tables} serialized tables."
        )
    markdown = structured_parse["markdown"].encode("utf-8")
    parse_result = json.dumps(structured_parse, indent=2).encode("utf-8")
    checkbox_crops = _checkbox_crops(result)
    artifacts = {
        "markdown": _artifact_entry("document.md", markdown),
        "parse_result": _artifact_entry("parse-result.json", parse_result),
        "annotated_pdf": _pending_artifact_entry("annotated.pdf"),
        "html": _pending_artifact_entry("document.html"),
        "checkbox_crops": {
            name: _artifact_entry(name, data) for name, data in checkbox_crops.items()
        },
    }
    processing: dict[str, Any] = {
        "requested_mode": result.requested_mode.value,
        "effective_mode": result.effective_mode.value,
        "routing": result.routing,
        "atomic_grounding_included": include_atomic_grounding,
    }
    usage_and_cost: dict[str, Any] = {
        "rapidocr": {
            "processing_seconds": result.timings.get("ocr_seconds"),
            "api_cost_usd": 0.0,
            "hardware_cost_usd": None,
        }
    }
    if result.layout_engine is not None:
        usage_and_cost["pp_doclayout_v3"] = {
            "processing_seconds": result.timings.get("layout_seconds"),
            "api_cost_usd": 0.0,
            "hardware_cost_usd": None,
        }
        usage_and_cost["table_structure"] = {
            "processing_seconds": result.timings.get("table_structure_seconds"),
            "api_cost_usd": 0.0,
            "hardware_cost_usd": None,
        }
    has_gpt = result.cloud_output is not None or result.usage.call_count > 0
    if has_gpt:
        processing.update(
            {
                "gpt_context_pages": result.cloud_pages,
                "gpt_image_pages": result.cloud_image_pages,
                "gpt_model": "gpt-6-sol",
                "reasoning_effort": "medium",
            }
        )
        usage_and_cost["gpt"] = result.usage.model_dump(mode="json")
    manifest = {
        "manifest_version": 7,
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
        "layout_engine": (
            {
                "name": result.layout_engine.name,
                "version": result.layout_engine.version,
                "device": result.layout_engine.device,
                "model_metadata": result.layout_engine.model_metadata,
            }
            if result.layout_engine
            else None
        ),
        "timings": result.timings,
        "quality_diagnostics": result.quality_diagnostics,
        "adaptive_processing": result.adaptive_processing,
        "visual_routing": result.visual_routing,
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
        "html_layout": {
            "renderer_version": 1,
            "coordinate_space": "normalized_page_xyxy",
            "page_background": "ocr_aligned_lossless_png",
            "selected_pages": result.selected_pages,
            "self_contained": True,
            "text_layers": ["refined", "raw"],
            "page_images": [],
        },
        "artifacts": artifacts,
    }
    if has_gpt:
        manifest["gpt_rate_assumptions"] = rate_assumptions()
    local_artifacts = LocalArtifacts(markdown, parse_result, manifest, checkbox_crops, result)
    record_stage_timing(
        result.timings, "artifact_finalization_seconds", time.perf_counter() - started
    )
    local_artifacts._refresh_timing_analysis()
    return local_artifacts


def _synchronize_accepted_tables(result: LocalParseResult) -> LocalParseResult:
    # Table review (Sol audit) can accept or reject a table's structure after
    # `result.markdown` was first assembled, so the stored Markdown can under-count how many
    # times a currently-accepted table's HTML actually appears. When that happens, rebuild
    # Markdown from the pages (which reflect the current accepted/rejected state) instead of
    # exporting a canonical result whose accepted tables and rendered Markdown have drifted
    # apart. `result` itself is left untouched; only the copy returned here changes.
    accepted_markup = [
        table.markdown
        for page in result.pages
        for table in page.table_structures
        if table.status == "valid" and not table.review_required and table.markdown
    ]
    missing = any(
        result.markdown.count(markup) < accepted_markup.count(markup)
        for markup in set(accepted_markup)
    )
    if not missing:
        return result
    synchronized = copy.copy(result)
    synchronized.markdown = (
        document_markdown_with_checkboxes(result.pages, result.checkboxes)
        if result.checkboxes
        else document_markdown(result.pages)
    )
    return synchronized


def build_local_bundle(result: LocalParseResult, *, include_atomic_grounding: bool = True) -> bytes:
    return build_local_artifacts(result, include_atomic_grounding=include_atomic_grounding).bundle


def _structured_parse(
    result: LocalParseResult, *, include_atomic_grounding: bool = True
) -> dict[str, Any]:
    return build_landing_parse(result, include_atomic_grounding=include_atomic_grounding)


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
        "provenance": chunk.provenance,
        "verification": chunk.verification,
    }


def _annotated_pdf(result: LocalParseResult) -> bytes:
    pages: list[Image.Image] = []
    table_reviews = result.document_metadata.get("table_reviews")
    accepted_table_ids = (
        {
            str(item["table_id"])
            for item in table_reviews
            if isinstance(item, dict)
            and item.get("status") == "accepted"
            and item.get("table_id") is not None
        }
        if isinstance(table_reviews, list)
        else None
    )
    for page in result.pages:
        image = Image.open(io.BytesIO(page.original_image_bytes or page.image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        for chunk in page.chunks:
            if chunk.provenance != "gpt-visual" or not chunk.bbox:
                continue
            left, top, right, bottom = chunk.bbox
            box = (
                left * image.width,
                top * image.height,
                right * image.width,
                bottom * image.height,
            )
            color = (
                "#008855"
                if chunk.verification in {"model_verified", "human_approved"}
                else "#D06000"
            )
            draw.rectangle(box, outline=color, width=max(2, image.width // 700))
            draw.text(box[:2], f"{chunk.type}: {chunk.verification}", fill=color)
        for region in page.layout_regions:
            points = (
                [(point[0], point[1]) for point in region.polygon]
                if region.polygon
                else [
                    (region.bbox[0] * image.width, region.bbox[1] * image.height),
                    (region.bbox[2] * image.width, region.bbox[1] * image.height),
                    (region.bbox[2] * image.width, region.bbox[3] * image.height),
                    (region.bbox[0] * image.width, region.bbox[3] * image.height),
                ]
            )
            draw.line(points + [points[0]], fill="#00A8B8", width=max(2, image.width // 700))
            draw.text(
                points[0],
                f"{region.id} · {region.label} · {region.score:.2f}",
                fill="#34285F",
            )
        for table in page.table_structures:
            if (
                table.status != "valid"
                or table.review_required
                or (accepted_table_ids is not None and table.id not in accepted_table_ids)
            ):
                continue
            for cell in table.cells:
                left, top, right, bottom = cell.bbox
                points = [
                    (left * image.width, top * image.height),
                    (right * image.width, top * image.height),
                    (right * image.width, bottom * image.height),
                    (left * image.width, bottom * image.height),
                ]
                draw.line(points + [points[0]], fill="#F0A020", width=max(2, image.width // 800))
        for checkbox in (
            item
            for item in result.checkboxes
            if item.page == page.page and is_publishable_checkbox(item)
        ):
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


def _artifact_entry(name: str, data: bytes) -> ArtifactMetadataValue:
    return {
        "name": name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "generated": True,
    }


def _pending_artifact_entry(name: str) -> ArtifactMetadataValue:
    return {"name": name, "bytes": None, "sha256": None, "generated": False}


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
                "layout_regions": (
                    [item.model_dump(mode="json") for item in page.layout_regions] if page else []
                ),
                "layout_block_links": (
                    [item.model_dump(mode="json") for item in page.layout_block_links]
                    if page
                    else []
                ),
                "reading_order_evidence": (
                    page.reading_order_evidence.model_dump(mode="json")
                    if page and page.reading_order_evidence
                    else None
                ),
                "table_structures": (
                    [item.model_dump(mode="json") for item in page.table_structures] if page else []
                ),
                "redaction_candidates": (
                    [item.model_dump(mode="json") for item in page.local_redaction_candidates]
                    if page
                    else []
                ),
                "warnings": page.warnings if page else [],
            }
        )
    return pages
