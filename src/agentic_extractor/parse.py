"""Canonical OCR parsing and cost-aware image routing.

Responsible for: intermediate layout chunking, reading order sorting,
Markdown synthesis from OCR blocks, low-confidence threshold tracking
(`LOW_CONFIDENCE_THRESHOLD = 0.85`), and checkbox representation formatting.

Must not: execute external OCR or model calls directly; operates strictly on
in-memory data models (`PageParse`, `Block`, `ParseChunk`).

Next: `pipeline.py`, which integrates `PageParse` into the full extraction workflow.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from statistics import fmean

from agentic_extractor.models import (
    Block,
    CheckboxRecord,
    CheckboxState,
    LayoutBlockLink,
    LayoutRegion,
    LocalCheckboxCandidate,
    LocalRedactionCandidate,
    ProcessingMode,
    ReadingOrderEvidence,
    TableStructureEvidence,
    VisualReviewRegion,
)

LOW_CONFIDENCE_THRESHOLD = 0.85


@dataclass(slots=True)
class ParseChunk:
    id: str
    page: int
    type: str
    reading_order: int
    text: str
    markdown: str
    source_block_ids: list[str]
    bbox: list[float] | None
    raw_scores: list[float | None]


@dataclass(slots=True)
class PageParse:
    page: int
    width: int
    height: int
    blocks: list[Block] = field(default_factory=list)
    chunks: list[ParseChunk] = field(default_factory=list)
    image_bytes: bytes = b""
    layout_image_bytes: bytes = b""
    original_image_bytes: bytes | None = None
    ocr_seconds: float = 0
    engine_elapsed_seconds: float | None = None
    stage_timings: dict[str, float | None] = field(default_factory=dict)
    raw_evidence: dict[str, object] = field(default_factory=dict)
    layout_raw_evidence: dict[str, object] = field(default_factory=dict)
    layout_signals: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    status: str = "completed"
    ocr_cache_hit: bool = False
    local_checkbox_candidates: list[LocalCheckboxCandidate] = field(default_factory=list)
    local_redaction_candidates: list[LocalRedactionCandidate] = field(default_factory=list)
    visual_review_regions: list[VisualReviewRegion] = field(default_factory=list)
    layout_regions: list[LayoutRegion] = field(default_factory=list)
    layout_block_links: list[LayoutBlockLink] = field(default_factory=list)
    reading_order_evidence: ReadingOrderEvidence | None = None
    table_structures: list[TableStructureEvidence] = field(default_factory=list)
    table_raw_evidence: list[dict[str, object]] = field(default_factory=list)

    @property
    def markdown(self) -> str:
        chunks = self.chunks or build_layout_chunks(self.blocks)
        return "\n\n".join(chunk.markdown for chunk in chunks if chunk.markdown)


def should_send_image(page: PageParse, mode: ProcessingMode, forced_pages: set[int]) -> bool:
    """Apply explicit, intentionally uncalibrated Balanced routing thresholds."""
    return bool(routing_reasons(page, mode, forced_pages))


def routing_reasons(page: PageParse, mode: ProcessingMode, forced_pages: set[int]) -> list[str]:
    """Return evidence-backed reasons for sending page context to GPT."""
    if mode is ProcessingMode.HIGH_ACCURACY:
        return ["high_accuracy_review"]
    reasons: list[str] = []
    if page.page in forced_pages:
        reasons.append("user_requested")
    if page.status == "failed" or not page.blocks:
        reasons.append("failed_or_empty_ocr")
        return reasons
    scores = [block.ocr_score for block in page.blocks if block.ocr_score is not None]
    if not scores or fmean(scores) < LOW_CONFIDENCE_THRESHOLD:
        reasons.append("low_confidence")
    if scores and sum(score < 0.70 for score in scores) / len(scores) > 0.20:
        reasons.append("conflicting_confidence")
    if len(page.blocks) >= 80:
        reasons.append("dense_layout")
    if any(block.type in {"table_row", "key_value"} for block in page.blocks):
        reasons.append("table_or_form")
    if page.layout_signals.get("columns_detected") == 2:
        reasons.append("multi_column")
    if page.layout_signals.get("ambiguous"):
        reasons.append("ambiguous_reading_order")
    grounded_area = sum(
        max(0.0, (_box(block)[2] - _box(block)[0]) * (_box(block)[3] - _box(block)[1]))
        for block in page.blocks
        if block.bbox
    )
    if grounded_area < 0.005:
        reasons.append("low_coverage")
    if _looks_like_complex_layout(page.blocks) and "multi_column" not in reasons:
        reasons.append("complex_layout")
    return reasons


def _looks_like_complex_layout(blocks: list[Block]) -> bool:
    grounded = [block for block in blocks if block.bbox]
    left = sum(_box(block)[0] < 0.45 and _box(block)[2] < 0.60 for block in grounded)
    right = sum(_box(block)[0] > 0.40 for block in grounded)
    pipe_rows = sum("|" in block.text or "\t" in block.text for block in blocks)
    return (left >= 4 and right >= 4) or pipe_rows >= 3


def document_markdown(pages: list[PageParse]) -> str:
    return "\n\n".join(f"<!-- page: {p.page} -->\n\n{p.markdown}" for p in pages)


def is_publishable_checkbox(checkbox: CheckboxRecord) -> bool:
    """Return whether a derived checkbox decision is safe as document content."""
    return (
        checkbox.decision_status in {"automated", "user_verified"}
        and checkbox.state is not CheckboxState.NOT_DETERMINABLE
        and bool(checkbox.label.strip())
    )


def document_markdown_with_checkboxes(
    pages: list[PageParse], checkboxes: list[CheckboxRecord]
) -> str:
    """Render derived checkbox markers in geometric order without changing OCR blocks."""
    markers = {
        CheckboxState.CHECKED: "[x]",
        CheckboxState.UNCHECKED: "[ ]",
        CheckboxState.INDETERMINATE: "[-]",
        CheckboxState.CROSSED_OUT: "[~]",
        CheckboxState.NOT_DETERMINABLE: "[?]",
    }
    rendered: list[str] = []
    for page in pages:
        page_checkboxes = [
            item for item in checkboxes if item.page == page.page and is_publishable_checkbox(item)
        ]
        if not page_checkboxes:
            rendered.append(f"<!-- page: {page.page} -->\n\n{page.markdown}")
            continue
        items: list[tuple[float, float, str]] = []
        for chunk in page.chunks or build_layout_chunks(page.blocks):
            box = chunk.bbox or [0, 1, 0, 1]
            items.append((box[1], box[0], chunk.markdown))
        for checkbox in page_checkboxes:
            left, top, _, _ = checkbox.control_bbox
            label = checkbox.label.strip()
            items.append((top, left, f"- {markers[checkbox.state]} {label}"))
        body = "\n\n".join(value for _, _, value in sorted(items, key=lambda item: item[:2]))
        rendered.append(f"<!-- page: {page.page} -->\n\n{body}")
    return "\n\n".join(rendered)


def reconstruct_layout(blocks: list[Block]) -> tuple[list[Block], dict[str, object]]:
    """Order and type OCR regions using only their text and geometry."""
    grounded = [block for block in blocks if block.bbox]
    ungrounded = [block for block in blocks if not block.bbox]
    heights = sorted(_box(block)[3] - _box(block)[1] for block in grounded)
    median_height = heights[len(heights) // 2] if heights else 0
    for block in blocks:
        text = block.text.strip()
        height = block.bbox[3] - block.bbox[1] if block.bbox else 0
        if text.startswith(("•", "- ", "* ")) or _numbered_list(text):
            block.type = "list_item"
        elif "\t" in text or text.count("|") >= 2:
            block.type = "table_row"
        elif ":" in text and len(text.split(":", 1)[0]) <= 40:
            block.type = "key_value"
        elif median_height and height >= median_height * 1.35 and len(text) <= 100:
            block.type = "heading"
        else:
            block.type = "paragraph"

    centers = sorted((_box(block)[0] + _box(block)[2]) / 2 for block in grounded)
    gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
    largest_gap, gap_index = max(gaps, default=(0.0, 0))
    two_columns = largest_gap > 0.25 and gap_index >= 1 and len(centers) - gap_index >= 3
    split = (centers[gap_index] + centers[gap_index + 1]) / 2 if two_columns else 0
    if two_columns:
        ordered = sorted(
            grounded,
            key=lambda block: (
                0 if (_box(block)[0] + _box(block)[2]) / 2 < split else 1,
                _box(block)[1],
                _box(block)[0],
            ),
        )
        order = "column-major"
    else:
        ordered = sorted(grounded, key=lambda block: (_box(block)[1], _box(block)[0]))
        order = "top-to-bottom-left-to-right"
    ambiguous = bool(ungrounded) or (0.12 < largest_gap <= 0.25)
    return ordered + ungrounded, {
        "reading_order": order,
        "columns_detected": 2 if two_columns else 1,
        "ambiguous": ambiguous,
        "orientation_degrees": None,
        "source": "derived_text_geometry",
        "native_layout": False,
    }


def layout_markdown(blocks: list[Block]) -> str:
    return "\n\n".join(chunk.markdown for chunk in build_layout_chunks(blocks))


def build_layout_chunks(blocks: list[Block]) -> list[ParseChunk]:
    """Group ordered OCR blocks only when text and geometry provide strong evidence."""
    chunks: list[ParseChunk] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        group = [block]
        chunk_type = block.type
        if block.type in {"list_item", "key_value", "table_row"}:
            while index + len(group) < len(blocks):
                candidate = blocks[index + len(group)]
                if candidate.type != block.type or candidate.page != block.page:
                    break
                group.append(candidate)
        if block.type == "table_row" and not _consistent_table(group):
            group = [block]
            chunk_type = "paragraph"
        elif block.type == "table_row":
            chunk_type = "table"
        elif block.type == "list_item":
            chunk_type = "list"
        elif block.type == "key_value" and len(group) > 1:
            chunk_type = "form"
        markdown = _chunk_markdown(group, chunk_type)
        order = len(chunks) + 1
        chunks.append(
            ParseChunk(
                id=f"p{block.page}-c{order}",
                page=block.page,
                type=chunk_type,
                reading_order=order,
                text="\n".join(item.text.strip() for item in group if item.text.strip()),
                markdown=markdown,
                source_block_ids=[item.id for item in group],
                bbox=_union_bbox(group),
                raw_scores=[item.ocr_score for item in group],
            )
        )
        index += len(group)
    return chunks


def _chunk_markdown(blocks: list[Block], chunk_type: str) -> str:
    texts = [block.text.strip() for block in blocks if block.text.strip()]
    if chunk_type == "heading":
        return f"## {texts[0]}" if texts else ""
    if chunk_type == "list":
        return "\n".join(
            text if text.startswith(("- ", "* ")) else f"- {text.lstrip('• ')}" for text in texts
        )
    if chunk_type == "table":
        rows = [_table_cells(text) for text in texts]
        parts = ["<table><tr>"]
        parts.extend(f"<th>{html.escape(cell)}</th>" for cell in rows[0])
        parts.append("</tr>")
        for row in rows[1:]:
            parts.append("<tr>")
            parts.extend(f"<td>{html.escape(cell)}</td>" for cell in row)
            parts.append("</tr>")
        parts.append("</table>")
        return "".join(parts)
    return "\n".join(texts)


def _table_cells(text: str) -> list[str]:
    return [cell.strip() for cell in text.replace("\t", "|").strip("|").split("|")]


def _consistent_table(blocks: list[Block]) -> bool:
    counts = [_table_cells(block.text) for block in blocks]
    return len(counts) >= 2 and len({len(row) for row in counts}) == 1 and len(counts[0]) >= 2


def _union_bbox(blocks: list[Block]) -> list[float] | None:
    boxes = [block.bbox for block in blocks if block.bbox]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _numbered_list(text: str) -> bool:
    head, separator, _ = text.partition(".")
    return bool(separator and head.isdigit() and len(head) <= 3)


def _box(block: Block) -> list[float]:
    if block.bbox is None:
        raise ValueError("Block has no bounding box.")
    return block.bbox
