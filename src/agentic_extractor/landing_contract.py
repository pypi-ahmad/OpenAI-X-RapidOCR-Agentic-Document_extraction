"""LandingAI-shaped public Parse projection derived from canonical local evidence.

Responsible for: projecting canonical local Parse output (`LocalParseResult`,
Markdown, and grounded chunks) into LandingAI ADE-compatible Parse JSON schema
(`markdown`, `metadata`, `structure` tree, and atomic grounding spans).

Must not: re-run OCR or modify source Markdown text or block coordinates.
Offsets are measured in 0-based Unicode codepoint ranges (`range_units: unicode_codepoints`),
and bounding boxes are normalized to [0, 1] relative to each page.

Next: `artifacts.py`, which saves this projection as `parse-result.json`.
"""

from __future__ import annotations

import hashlib
import html
import re
from collections import defaultdict
from typing import Any

from agentic_extractor.ocr import LocalParseResult
from agentic_extractor.parse import PageParse
from agentic_extractor.table_structure import build_chunks_with_tables

_PAGE_MARKER = re.compile(r"<!--\s*page\s*:\s*(\d+)\s*-->", re.IGNORECASE)
_SEMANTIC_TYPES = {"text", "marginalia", "logo", "attestation", "figure", "scan_code"}


def build_landing_parse(
    result: LocalParseResult, *, include_atomic_grounding: bool = True
) -> dict[str, Any]:
    """Project accepted Markdown and grounding into the public ADE-like schema."""
    markdown, page_ranges = _public_markdown(result)
    counters: defaultdict[str, int] = defaultdict(int)
    pages = [
        _page_node(page, markdown, page_ranges.get(page.page, (0, 0)), counters)
        for page in result.pages
    ]
    if not include_atomic_grounding:
        _remove_atomic_grounding(pages)
    digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()[:20]
    duration = result.timings.get("total_seconds")
    if duration is None:
        duration = sum(
            value for value in result.timings.values() if isinstance(value, (int, float))
        )
    page_count = result.document_metadata.get("page_count")
    if not isinstance(page_count, int):
        page_count = len(result.selected_pages)
    return {
        "markdown": markdown,
        "metadata": {
            "job_id": f"local-{digest}",
            "model_version": "rapidocr-gpt-5.6-luna-pp-doclayout-v3",
            "page_count": page_count,
            "output_markdown_chars": len(markdown),
            "range_units": "unicode_codepoints",
            "openapi_spec": "/openapi.json",
            "failed_pages": result.failed_pages,
            "duration_ms": round(float(duration) * 1000),
            "billing": {"service_tier": "local", "total_credits": None},
        },
        "structure": {"type": "document", "children": pages},
    }


def _public_markdown(result: LocalParseResult) -> tuple[str, dict[int, tuple[int, int]]]:
    matches = list(_PAGE_MARKER.finditer(result.markdown))
    bodies: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(result.markdown)
        bodies[int(match.group(1))] = result.markdown[match.end() : end].strip()
    if not matches:
        for page in result.pages:
            bodies[page.page] = page.markdown.strip()

    parts: list[str] = []
    ranges: dict[int, tuple[int, int]] = {}
    cursor = 0
    for index, page in enumerate(result.pages):
        if index:
            separator = "\n\n<!-- PAGE BREAK -->\n\n"
            parts.append(separator)
            cursor += len(separator)
        body = bodies.get(page.page, page.markdown.strip())
        start = cursor
        parts.append(body)
        cursor += len(body)
        ranges[page.page] = (start, cursor)
    document_id = hashlib.sha256(result.markdown.encode("utf-8")).hexdigest()[:20]
    suffix = f"\n\n<!-- document_id: local-{document_id} -->"
    parts.append(suffix)
    return "".join(parts), ranges


def _page_node(
    page: PageParse,
    markdown: str,
    page_range: tuple[int, int],
    counters: defaultdict[str, int],
) -> dict[str, Any]:
    start, end = page_range
    chunks = page.chunks or build_chunks_with_tables(page)
    table_by_markdown = {
        table.markdown: table
        for table in page.table_structures
        if table.status == "valid" and not table.review_required and table.markdown
    }
    children: list[tuple[int, dict[str, Any]]] = []
    cursor = start
    for chunk in chunks:
        value = chunk.markdown.strip()
        located = markdown.find(value, cursor, end) if value else -1
        if located < 0:
            located = markdown.find(value, start, end) if value else -1
        if located < 0:
            continue
        chunk_range = (located, located + len(value))
        cursor = chunk_range[1]
        table = table_by_markdown.get(value)
        if table is not None:
            node = _table_node(page, table, markdown, chunk_range, counters)
        else:
            node_type = _semantic_type(page, chunk.source_block_ids, chunk.type)
            region_id = _primary_region_id(page, chunk.source_block_ids)
            # Layout region coalescing: merge adjacent chunks from the same detector
            # region and semantic type into a unified structural node.
            if (
                region_id is not None
                and children
                and children[-1][1].get("_region_id") == region_id
                and children[-1][1].get("type") == node_type
                and node_type not in {"logo", "attestation", "figure", "scan_code"}
            ):
                prior_start, prior = children.pop()
                prior_ids = list(prior.get("_block_ids", []))
                combined_ids = [*prior_ids, *chunk.source_block_ids]
                node = _semantic_node(
                    page,
                    node_type,
                    combined_ids,
                    markdown,
                    (prior_start, chunk_range[1]),
                    counters,
                    _union_boxes([block.bbox for block in page.blocks if block.id in combined_ids]),
                    node_id=str(prior["id"]),
                )
                node["_region_id"] = region_id
                node["_block_ids"] = combined_ids
                children.append((prior_start, node))
                continue
            node = _semantic_node(
                page,
                node_type,
                chunk.source_block_ids,
                markdown,
                chunk_range,
                counters,
                chunk.bbox,
            )
            node["_region_id"] = region_id
            node["_block_ids"] = list(chunk.source_block_ids)
        children.append((located, node))
    if not children and end > start:
        children.append(
            (
                start,
                _semantic_node(page, "text", [], markdown, page_range, counters, [0, 0, 1, 1]),
            )
        )
    return {
        "type": "page",
        "grounding": _grounding(page.page, page_range, [0, 0, 1, 1]),
        "status": "ok" if page.status == "completed" else page.status,
        "children": [_public_node(node) for _, node in sorted(children, key=lambda item: item[0])],
    }


def _semantic_type(page: PageParse, block_ids: list[str], preferred: str = "text") -> str:
    if preferred in _SEMANTIC_TYPES - {"text"}:
        return preferred
    blocks = {block.id: block for block in page.blocks}
    if any(
        marker in blocks[block_id].text.upper()
        for block_id in block_ids
        if block_id in blocks
        for marker in ("[E-SIGNED]", "[SIGNED]", "[ILLEGIBLE_SIGNATURE]")
    ):
        return "attestation"
    region_by_id = {region.id: region for region in page.layout_regions}
    links = {link.block_id: link for link in page.layout_block_links}
    labels = {
        region_by_id[links[block_id].primary_region_id].label.lower()
        for block_id in block_ids
        if block_id in links and links[block_id].primary_region_id in region_by_id
    }
    if labels & {"header", "footer", "page_number", "footnote", "marginalia"}:
        return "marginalia"
    if any("logo" in label for label in labels):
        return "logo"
    if any(token in label for label in labels for token in ("signature", "stamp", "seal")):
        return "attestation"
    return "text"


def _primary_region_id(page: PageParse, block_ids: list[str]) -> str | None:
    ids = set(block_ids)
    primary = [link.primary_region_id for link in page.layout_block_links if link.block_id in ids]
    return max(set(primary), key=primary.count) if primary else None


def _semantic_node(
    page: PageParse,
    node_type: str,
    block_ids: list[str],
    markdown: str,
    span: tuple[int, int],
    counters: defaultdict[str, int],
    bbox: list[float] | None,
    node_id: str | None = None,
) -> dict[str, Any]:
    node_type = node_type if node_type in _SEMANTIC_TYPES else "text"
    if node_id is None:
        node_id = f"{node_type}-{counters[node_type]}"
        counters[node_type] += 1
    blocks = {block.id: block for block in page.blocks}
    atomic = []
    search_from = span[0]
    for block_id in block_ids:
        block = blocks.get(block_id)
        if block is None or not block.bbox:
            continue
        block_span = _find_text_span(markdown, block.text, search_from, span[1]) or span
        search_from = block_span[1]
        atomic.append(_grounding(page.page, block_span, block.bbox))
    return {
        "type": node_type,
        "id": node_id,
        "grounding": _grounding(
            page.page, span, bbox or _union_boxes([b.bbox for b in blocks.values()])
        ),
        "atomic_grounding": atomic or [_grounding(page.page, span, bbox or [0, 0, 1, 1])],
    }


def _table_node(
    page: PageParse,
    table: Any,
    markdown: str,
    span: tuple[int, int],
    counters: defaultdict[str, int],
) -> dict[str, Any]:
    table_id = f"table-{counters['table']}"
    counters["table"] += 1
    children = []
    cursor = span[0]
    blocks = {block.id: block for block in page.blocks}
    for cell in sorted(table.cells, key=lambda item: (item.row, item.column)):
        cell_text = html.escape(cell.source_text.strip())
        cell_span = _find_text_span(markdown, cell_text, cursor, span[1]) or span
        cursor = cell_span[1]
        cell_id = f"table_cell-{counters['table_cell']}"
        counters["table_cell"] += 1
        atomic = [
            _grounding(page.page, cell_span, blocks[block_id].bbox)
            for block_id in cell.source_block_ids
            if block_id in blocks and blocks[block_id].bbox
        ]
        children.append(
            {
                "type": "table_cell",
                "id": cell_id,
                "grounding": _grounding(page.page, cell_span, cell.bbox),
                "atomic_grounding": atomic or [_grounding(page.page, cell_span, cell.bbox)],
                "row": cell.row - 1,
                "col": cell.column - 1,
                "colspan": cell.column_span,
                "rowspan": cell.row_span,
            }
        )
    return {
        "type": "table",
        "id": table_id,
        "grounding": _grounding(page.page, span, table.bbox),
        "children": children,
    }


def _find_text_span(markdown: str, text: str, start: int, end: int) -> tuple[int, int] | None:
    if not text:
        return None
    located = markdown.find(text, start, end)
    return (located, located + len(text)) if located >= 0 else None


def _grounding(page: int, span: tuple[int, int], bbox: list[float] | None) -> dict[str, Any]:
    box = bbox or [0, 0, 1, 1]
    return {
        "page": page,
        "range": {"start": span[0], "end": span[1]},
        "box": {"xmin": box[0], "ymin": box[1], "xmax": box[2], "ymax": box[3]},
    }


def _union_boxes(boxes: list[list[float] | None]) -> list[float]:
    valid = [box for box in boxes if box]
    if not valid:
        return [0, 0, 1, 1]
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def _public_node(node: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in node.items() if not key.startswith("_")}


def _remove_atomic_grounding(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("atomic_grounding", None)
        for child in value.values():
            _remove_atomic_grounding(child)
    elif isinstance(value, list):
        for child in value:
            _remove_atomic_grounding(child)
