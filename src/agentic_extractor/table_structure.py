"""Ground table-structure model output in immutable RapidOCR evidence.

Responsible for: taking raw table HTML structure and bbox coordinates from
the PaddleX worker, re-parsing grid topology (`tr`/`td`/`th` with row/col
spans), spatially matching cell rectangles against immutable RapidOCR
word/block boxes, and generating cell-grounded HTML Markdown.

Must not: trust external worker text over RapidOCR evidence, or drop raw OCR
words falling inside table bounding boxes without attribution.

Next: `layout.py` which interfaces with the PaddleX subprocess worker, and
`pipeline.py` which calls `apply_table_reviews`.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Literal

from agentic_extractor.models import TableCellEvidence, TableStructureEvidence
from agentic_extractor.parse import PageParse, ParseChunk, build_layout_chunks


@dataclass(slots=True)
class _CellSpec:
    row: int
    column: int
    row_span: int
    column_span: int
    tag: Literal["th", "td"]


class _StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[_CellSpec] = []
        self.row = 0
        self._occupied: set[tuple[int, int]] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self.row += 1
            return
        if tag not in {"td", "th"} or self.row < 1:
            return
        values = dict(attrs)
        row_span = _positive_int(values.get("rowspan"))
        column_span = _positive_int(values.get("colspan"))
        column = 1
        while (self.row, column) in self._occupied:
            column += 1
        cell_tag: Literal["th", "td"] = "th" if tag == "th" else "td"
        self.cells.append(_CellSpec(self.row, column, row_span, column_span, cell_tag))
        for row in range(self.row, self.row + row_span):
            for col in range(column, column + column_span):
                self._occupied.add((row, col))


def normalize_table_result(
    layout_region_id: str, value: dict[str, Any], page: PageParse
) -> TableStructureEvidence:
    """Validate one supported Paddle table result and attach RapidOCR text."""
    region = next(
        (
            item
            for item in page.layout_regions
            if item.id == layout_region_id and item.label == "table"
        ),
        None,
    )
    if region is None:
        raise ValueError(f"Unknown table layout region: {layout_region_id}")
    classifier_value = value.get("classifier")
    classifier: dict[str, Any] = classifier_value if isinstance(classifier_value, dict) else {}
    structure_value_raw = value.get("structure")
    structure: dict[str, Any] = structure_value_raw if isinstance(structure_value_raw, dict) else {}
    labels = classifier.get("label_names") or []
    scores = classifier.get("scores") or []
    label = str(labels[0]).lower() if labels else ""
    declared_model = value.get("structure_model")
    structure_model = (
        declared_model
        if declared_model in {"SLANeXt_wired", "SLANet_plus"}
        else ("SLANeXt_wired" if "wired" in label and "wireless" not in label else "SLANet_plus")
    )
    style = "wired" if structure_model == "SLANeXt_wired" else "wireless"
    target_label = "wired_table" if style == "wired" else "wireless_table"
    score_index = next(
        (index for index, item in enumerate(labels) if str(item).lower() == target_label), 0
    )
    classifier_score = _score(scores[score_index] if score_index < len(scores) else 0)
    structure_score = _score(structure.get("structure_score", 0))
    structure_value = structure.get("structure") or []
    markup = "".join(str(item) for item in structure_value)
    parser = _StructureParser()
    parser.feed(markup)
    parser.close()
    boxes = structure.get("bbox") or []
    warnings: list[str] = []
    if value.get("structure_geometry_valid") is False:
        warnings.append("Table structure geometry does not fit the source table crop.")
    if len(boxes) != len(parser.cells):
        warnings.append("Table structure cell count does not match predicted cell geometry.")
    coordinate_mode: Literal["crop", "page"] = (
        "crop"
        if declared_model in {"SLANeXt_wired", "SLANet_plus"}
        else _table_coordinate_mode(boxes, region.coordinate, region.bbox, page)
    )
    cells: list[TableCellEvidence] = []
    for index, (spec, box) in enumerate(zip(parser.cells, boxes, strict=False), 1):
        page_box = _page_bbox(
            box,
            region.coordinate,
            page.width,
            page.height,
            mode=coordinate_mode,
            clip_crop=declared_model in {"SLANeXt_wired", "SLANet_plus"},
        )
        cells.append(
            TableCellEvidence(
                id=f"{layout_region_id}-cell-{index}",
                row=spec.row,
                column=spec.column,
                row_span=spec.row_span,
                column_span=spec.column_span,
                tag=spec.tag,
                bbox=page_box,
                source_block_ids=[],
                source_text="",
                raw_scores=[],
            )
        )
    linked_ids = {
        link.block_id
        for link in page.layout_block_links
        if link.primary_region_id == layout_region_id
    }
    table_block_values = sorted(
        (
            block
            for block in page.blocks
            if block.bbox
            and (block.id in linked_ids if linked_ids else _center_in(block.bbox, region.bbox))
        ),
        key=_block_position,
    )
    unmatched: list[str] = []
    for block in table_block_values:
        assert block.bbox is not None
        ranked = sorted(
            (
                (_block_cell_coverage(block.bbox, cell.bbox), index, cell)
                for index, cell in enumerate(cells)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if not ranked or ranked[0][0] <= 0:
            unmatched.append(block.id)
            continue
        cell = ranked[0][2]
        cell.source_block_ids.append(block.id)
        cell.raw_scores.append(block.ocr_score)
    blocks_by_id = {block.id: block for block in table_block_values}
    for cell in cells:
        cell.source_text = " ".join(
            blocks_by_id[block_id].text.strip()
            for block_id in cell.source_block_ids
            if blocks_by_id[block_id].text.strip()
        )
    if unmatched:
        warnings.append(
            "RapidOCR table blocks were not assigned to a predicted cell: " + ", ".join(unmatched)
        )
    outside = [cell.id for cell in cells if not _inside(cell.bbox, region.bbox, page)]
    if outside:
        warnings.append(
            "Predicted table cell geometry falls outside its parent table region: "
            + ", ".join(outside)
        )
    valid = bool(cells) and len(boxes) == len(parser.cells) and not warnings
    table = TableStructureEvidence(
        id=f"{layout_region_id}-table",
        page=page.page,
        layout_region_id=layout_region_id,
        bbox=region.bbox,
        style=style,
        classifier_score=classifier_score,
        structure_score=structure_score,
        structure_model=structure_model,
        # Keep rejected geometry as non-published review evidence. Only valid,
        # reviewed tables are rendered by build_chunks_with_tables().
        cells=cells,
        markdown=_render_table(cells) if valid else "",
        status="valid" if valid else "invalid",
        review_required=not valid,
        warnings=warnings,
    )
    return table


def enrich_page_tables(page: PageParse, tables: list[TableStructureEvidence]) -> None:
    page.table_structures = tables
    page.warnings.extend(warning for table in tables for warning in table.warnings)
    page.chunks = build_chunks_with_tables(page)


def apply_table_reviews(
    pages: list[PageParse], reviews: list[Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply only grounded Luna table decisions to derived table evidence."""
    blocks = {block.id: block for page in pages for block in page.blocks}
    words = {
        item["id"]: item
        for page in pages
        for item in table_word_evidence(page, {block.id for block in page.blocks})
    }
    by_id = {table.id: table for page in pages for table in page.table_structures}
    reviews_by_id: dict[str, Any] = {}
    duplicate_ids: set[str] = set()
    for review in reviews:
        if review.table_id in reviews_by_id:
            duplicate_ids.add(review.table_id)
        reviews_by_id[review.table_id] = review
    audits: list[dict[str, Any]] = []
    warnings: list[str] = []
    for table_id, table in by_id.items():
        review = reviews_by_id.get(table_id)
        status = "unresolved"
        reason = "Luna did not return a table review."
        if table_id in duplicate_ids:
            reason = "Luna returned duplicate reviews for this table."
        elif review is not None and review.page == table.page:
            visible_cell_count = getattr(review, "visible_cell_count", None)
            if (
                review.outcome == "confirmed"
                and table.status == "valid"
                and not review.cells
                and visible_cell_count == len(table.cells)
            ):
                _refresh_table_text(table, blocks)
                table.review_required = False
                status = "accepted"
                reason = None
            elif review.outcome == "not_table" and visible_cell_count == 0 and not review.cells:
                table.cells = []
                table.markdown = ""
                table.status = "invalid"
                table.review_required = False
                status = "rejected"
                reason = review.warning or "Luna rejected this layout false positive as non-table."
            elif review.outcome == "corrected" and _matches_applied_correction(table, review):
                status = "accepted"
                reason = None
            elif review.outcome == "corrected":
                page = next(item for item in pages if item.page == table.page)
                locally_grounded_block_ids = {
                    block_id for cell in table.cells for block_id in cell.source_block_ids
                }
                cells: list[TableCellEvidence] = []
                explicit_source_block_ids: list[str] = []
                invalid = False
                occupied: set[tuple[int, int]] = set()
                for index, proposal in enumerate(review.cells, 1):
                    slots = {
                        (row, column)
                        for row in range(proposal.row, proposal.row + proposal.row_span)
                        for column in range(proposal.column, proposal.column + proposal.column_span)
                    }
                    if occupied & slots:
                        invalid = True
                        break
                    occupied.update(slots)
                    proposal_word_ids = list(getattr(proposal, "source_word_ids", []))
                    if proposal_word_ids:
                        source_words = [words.get(word_id) for word_id in proposal_word_ids]
                        if any(word is None or word["page"] != table.page for word in source_words):
                            invalid = True
                            break
                        valid_words = [word for word in source_words if word is not None]
                        source_block_ids = list(
                            dict.fromkeys(str(word["block_id"]) for word in valid_words)
                        )
                        bbox = _union_boxes([word["bbox"] for word in valid_words])
                        source_text = " ".join(str(word["text"]) for word in valid_words)
                        raw_scores = [word["score"] for word in valid_words]
                    elif proposal.source_block_ids:
                        explicit_source_block_ids.extend(proposal.source_block_ids)
                        source_blocks = [
                            blocks.get(block_id) for block_id in proposal.source_block_ids
                        ]
                        if not source_blocks or any(
                            block is None or block.page != table.page for block in source_blocks
                        ):
                            invalid = True
                            break
                        valid_blocks = [block for block in source_blocks if block is not None]
                        source_block_ids = list(proposal.source_block_ids)
                        bbox = _union_boxes([block.bbox for block in valid_blocks if block.bbox])
                        source_text = " ".join(
                            block.text.strip() for block in valid_blocks if block.text.strip()
                        )
                        raw_scores = [block.ocr_score for block in valid_blocks]
                    else:
                        proposal_bbox = _normalized_bbox(getattr(proposal, "bbox", None))
                        review_bbox = table_review_bbox(page, table)
                        if (
                            proposal.text
                            or proposal_bbox is None
                            or not _center_in(proposal_bbox, review_bbox)
                        ):
                            invalid = True
                            break
                        source_block_ids = []
                        bbox = _clip_bbox(proposal_bbox, review_bbox)
                        source_text = ""
                        raw_scores = []
                    if bbox is None or not _center_in(bbox, table_review_bbox(page, table)):
                        invalid = True
                        break
                    cells.append(
                        TableCellEvidence(
                            id=f"{table.layout_region_id}-gpt-cell-{index}",
                            row=proposal.row,
                            column=proposal.column,
                            row_span=proposal.row_span,
                            column_span=proposal.column_span,
                            tag=proposal.tag,
                            bbox=bbox,
                            source_block_ids=source_block_ids,
                            source_word_ids=proposal_word_ids,
                            source_text=source_text,
                            raw_scores=raw_scores,
                        )
                    )
                expected_ids = table_review_block_ids(page, table)
                supplied = [block_id for cell in cells for block_id in cell.source_block_ids]
                supplied_ids = set(supplied)
                explicit_supplied_ids = set(explicit_source_block_ids)
                excluded = list(getattr(review, "excluded_source_block_ids", []))
                excluded_ids = set(excluded)
                expected_words = table_word_evidence(page, expected_ids)
                expected_word_ids = {item["id"] for item in expected_words}
                word_block_ids = {str(item["id"]): str(item["block_id"]) for item in expected_words}
                if table.status == "valid" and locally_grounded_block_ids:
                    required_word_ids = {
                        item["id"]
                        for item in expected_words
                        if item["block_id"] in locally_grounded_block_ids
                    }
                else:
                    required_word_ids = {
                        item["id"]
                        for item in expected_words
                        if _center_in(item["bbox"], table.bbox)
                    }
                supplied_words = [word_id for cell in cells for word_id in cell.source_word_ids]
                supplied_word_ids = set(supplied_words)
                excluded_words = list(getattr(review, "excluded_source_word_ids", []))
                excluded_word_ids = set(excluded_words)
                block_referenced_word_ids = {
                    word_id
                    for word_id, block_id in word_block_ids.items()
                    if block_id in explicit_supplied_ids | excluded_ids
                }
                direct_word_block_ids = {
                    word_block_ids[word_id]
                    for word_id in supplied_word_ids | excluded_word_ids
                    if word_id in word_block_ids
                }
                word_grounding_valid = (
                    bool(expected_word_ids)
                    and len(supplied_words) == len(supplied_word_ids)
                    and len(excluded_words) == len(excluded_word_ids)
                    and len(explicit_source_block_ids) == len(explicit_supplied_ids)
                    and len(excluded) == len(excluded_ids)
                    and not supplied_word_ids & excluded_word_ids
                    and not explicit_supplied_ids & excluded_ids
                    and supplied_word_ids | excluded_word_ids <= expected_word_ids
                    and explicit_supplied_ids | excluded_ids <= expected_ids
                    and not direct_word_block_ids & (explicit_supplied_ids | excluded_ids)
                    and required_word_ids
                    <= supplied_word_ids | excluded_word_ids | block_referenced_word_ids
                )
                required_block_ids: set[str] = set()
                for block_id in expected_ids:
                    block = blocks.get(block_id)
                    required_by_local_table = (
                        table.status == "valid"
                        and bool(locally_grounded_block_ids)
                        and block_id in locally_grounded_block_ids
                    )
                    required_by_detected_region = (
                        table.status != "valid"
                        and block is not None
                        and block.bbox
                        and _center_in(block.bbox, table.bbox)
                    )
                    if required_by_local_table or required_by_detected_region:
                        required_block_ids.add(block_id)
                block_grounding_valid = (
                    not expected_word_ids
                    and len(supplied) == len(supplied_ids)
                    and len(excluded) == len(excluded_ids)
                    and not supplied_ids & excluded_ids
                    and supplied_ids | excluded_ids <= expected_ids
                    and required_block_ids <= supplied_ids | excluded_ids
                )
                implausible_promotion = bool(
                    cells
                    and table.status != "valid"
                    and len({cell.row for cell in cells}) == 1
                    and len(cells) <= 2
                    and len({block_id for cell in cells for block_id in cell.source_block_ids}) >= 6
                )
                if implausible_promotion:
                    table.cells = []
                    table.markdown = ""
                    table.status = "invalid"
                    table.review_required = False
                    status = "rejected"
                    reason = (
                        "A locally invalid, dense form region cannot be promoted from a "
                        "one-row, two-cell correction alone."
                    )
                elif cells and not invalid and (word_grounding_valid or block_grounding_valid):
                    table.cells = cells
                    table.bbox = _union_boxes([cell.bbox for cell in cells]) or table.bbox
                    table.markdown = _render_table(cells)
                    table.status = "valid"
                    table.review_required = False
                    status = "accepted"
                    reason = None
                else:
                    reason = "Luna table correction lacked valid RapidOCR grounding."
            elif review.outcome == "abstained":
                reason = review.warning or "Luna abstained from table review."
            elif review.outcome == "confirmed" and review.cells:
                reason = "Luna confirmed outcome also supplied replacement cells."
            elif review.outcome == "confirmed":
                reason = "Luna visible cell count did not match the confirmed local grid."
            elif review.outcome == "not_table":
                reason = "Luna non-table outcome supplied an inconsistent cell count or grid."
            else:
                reason = "Luna confirmed a locally invalid table structure."
        if status == "unresolved":
            table.review_required = True
            warnings.append(f"Table {table_id} requires review: {reason}")
        audits.append(
            {
                "table_id": table_id,
                "page": table.page,
                "status": status,
                "outcome": review.outcome if review is not None else "missing",
                "reason": reason,
                "excluded_source_block_ids": (
                    list(getattr(review, "excluded_source_block_ids", []))
                    if review is not None
                    else []
                ),
                "excluded_source_word_ids": (
                    list(getattr(review, "excluded_source_word_ids", []))
                    if review is not None
                    else []
                ),
            }
        )
    for page in pages:
        page.chunks = build_chunks_with_tables(page)
    return audits, warnings


def _matches_applied_correction(table: TableStructureEvidence, review: Any) -> bool:
    """Recognize an already-validated correction without weakening grounding checks."""
    if table.status != "valid" or table.review_required or len(table.cells) != len(review.cells):
        return False
    cells = {
        (cell.row, cell.column, cell.row_span, cell.column_span, cell.tag): cell
        for cell in table.cells
    }
    if len(cells) != len(table.cells):
        return False
    for proposal in review.cells:
        cell = cells.get(
            (
                proposal.row,
                proposal.column,
                proposal.row_span,
                proposal.column_span,
                proposal.tag,
            )
        )
        if cell is None:
            return False
        proposal_words = list(getattr(proposal, "source_word_ids", []))
        proposal_blocks = list(getattr(proposal, "source_block_ids", []))
        if proposal_words:
            if cell.source_word_ids != proposal_words:
                return False
        elif proposal_blocks:
            if cell.source_block_ids != proposal_blocks:
                return False
        elif _normalized_bbox(getattr(proposal, "bbox", None)) != cell.bbox:
            return False
    return True


def _refresh_table_text(table: TableStructureEvidence, blocks: dict[str, Any]) -> None:
    for cell in table.cells:
        sources = [blocks.get(block_id) for block_id in cell.source_block_ids]
        cell.source_text = " ".join(
            block.text.strip() for block in sources if block is not None and block.text.strip()
        )
    table.markdown = _render_table(table.cells)


def table_review_bbox(page: PageParse, table: TableStructureEvidence) -> list[float]:
    """Return a bounded review window that can recover detector-clipped edge rows."""
    left, top, right, bottom = table.bbox
    height = max(0.0, bottom - top)
    pad_y = min(0.06, max(0.035, height * 0.5))
    review = [
        max(0.0, left - 0.01),
        max(0.0, top - pad_y),
        min(1.0, right + 0.01),
        min(1.0, bottom + pad_y),
    ]
    neighbors = [
        item
        for item in page.table_structures
        if item.id != table.id and _horizontal_overlap(item.bbox, table.bbox) >= 0.5
    ]
    previous_bottoms = [item.bbox[3] for item in neighbors if item.bbox[3] <= top]
    next_tops = [item.bbox[1] for item in neighbors if item.bbox[1] >= bottom]
    if previous_bottoms:
        previous_bottom = max(previous_bottoms)
        review[1] = top if previous_bottom >= top - pad_y else review[1]
    if next_tops:
        review[3] = min(review[3], min(next_tops))
    return review


def table_review_block_ids(page: PageParse, table: TableStructureEvidence) -> set[str]:
    """Return linked evidence plus adjacent OCR blocks from the bounded review window."""
    linked_ids = {
        link.block_id
        for link in page.layout_block_links
        if link.primary_region_id == table.layout_region_id
    }
    if not linked_ids:
        linked_ids = {
            block.id for block in page.blocks if block.bbox and _center_in(block.bbox, table.bbox)
        }
    review_bbox = table_review_bbox(page, table)
    adjacent_ids = {
        block.id
        for block in page.blocks
        if block.bbox
        and _center_in(block.bbox, review_bbox)
        and not _center_in(block.bbox, table.bbox)
    }
    return linked_ids | adjacent_ids


def build_chunks_with_tables(page: PageParse) -> list[ParseChunk]:
    block_order = {block.id: index for index, block in enumerate(page.blocks)}
    valid = [
        table
        for table in page.table_structures
        if table.status == "valid" and not table.review_required
    ]
    table_block_ids = {
        block_id for table in valid for cell in table.cells for block_id in cell.source_block_ids
    }
    chunks = build_layout_chunks(
        [block for block in page.blocks if block.id not in table_block_ids]
    )
    for table in valid:
        source_ids = [block_id for cell in table.cells for block_id in cell.source_block_ids]
        scores = [score for cell in table.cells for score in cell.raw_scores]
        chunks.append(
            ParseChunk(
                id=f"p{page.page}-c-table-{len(chunks) + 1}",
                page=page.page,
                type="table",
                reading_order=0,
                text="\n".join(cell.source_text for cell in table.cells if cell.source_text),
                markdown=table.markdown,
                source_block_ids=source_ids,
                bbox=table.bbox,
                raw_scores=scores,
            )
        )
    chunks.sort(
        key=lambda chunk: min(
            (block_order[block_id] for block_id in chunk.source_block_ids),
            default=len(block_order),
        )
    )
    for order, chunk in enumerate(chunks, 1):
        chunk.reading_order = order
        chunk.id = f"p{page.page}-c{order}"
    return chunks


def _render_table(cells: list[TableCellEvidence]) -> str:
    rows = sorted({cell.row for cell in cells})
    parts = ["<table>"]
    for row in rows:
        parts.append("<tr>")
        for cell in sorted(
            (item for item in cells if item.row == row), key=lambda item: item.column
        ):
            attrs = ""
            if cell.row_span > 1:
                attrs += f' rowspan="{cell.row_span}"'
            if cell.column_span > 1:
                attrs += f' colspan="{cell.column_span}"'
            parts.append(f"<{cell.tag}{attrs}>{html.escape(cell.source_text)}</{cell.tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _raw_bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Table cell geometry must be an xyxy box or quadrilateral.")
    if len(value) == 4 and all(isinstance(item, (list, tuple)) for item in value):
        points = [(float(item[0]), float(item[1])) for item in value if len(item) == 2]
        if len(points) != 4:
            raise ValueError("Table cell quadrilateral is invalid.")
        left, top = min(point[0] for point in points), min(point[1] for point in points)
        right, bottom = max(point[0] for point in points), max(point[1] for point in points)
    elif len(value) == 8:
        coordinates = [float(item) for item in value]
        left, right = min(coordinates[::2]), max(coordinates[::2])
        top, bottom = min(coordinates[1::2]), max(coordinates[1::2])
    elif len(value) == 4:
        left, top, right, bottom = (float(item) for item in value)
    else:
        raise ValueError("Table cell geometry must be an xyxy box or quadrilateral.")
    return left, top, right, bottom


def _page_bbox(
    value: Any,
    region: list[float],
    width: int,
    height: int,
    *,
    mode: Literal["crop", "page"] = "crop",
    clip_crop: bool = False,
) -> list[float]:
    left, top, right, bottom = _raw_bbox(value)
    region_left, region_top, region_right, region_bottom = region
    if mode == "crop" and clip_crop:
        crop_width = max(1.0, region_right - region_left)
        crop_height = max(1.0, region_bottom - region_top)
        left, right = max(0.0, left), min(crop_width, right)
        top, bottom = max(0.0, top), min(crop_height, bottom)
    offset_x = region_left if mode == "crop" else 0
    offset_y = region_top if mode == "crop" else 0
    result = [
        (offset_x + left) / width,
        (offset_y + top) / height,
        (offset_x + right) / width,
        (offset_y + bottom) / height,
    ]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError("Table cell geometry is invalid.")
    return result


def _table_coordinate_mode(
    boxes: list[Any],
    region_pixels: list[float],
    region_normalized: list[float],
    page: PageParse,
) -> Literal["crop", "page"]:
    def inside_count(mode: Literal["crop", "page"]) -> int:
        count = 0
        for box in boxes:
            try:
                normalized = _page_bbox(box, region_pixels, page.width, page.height, mode=mode)
            except (TypeError, ValueError):
                continue
            count += int(_inside(normalized, region_normalized, page))
        return count

    return "page" if inside_count("page") > inside_count("crop") else "crop"


def _center_in(block: list[float], container: list[float]) -> bool:
    x = (block[0] + block[2]) / 2
    y = (block[1] + block[3]) / 2
    return container[0] <= x <= container[2] and container[1] <= y <= container[3]


def _horizontal_overlap(first: list[float], second: list[float]) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    width = min(first[2] - first[0], second[2] - second[0])
    return intersection / width if width > 0 else 0.0


def _clip_bbox(box: list[float], container: list[float]) -> list[float]:
    return [
        max(box[0], container[0]),
        max(box[1], container[1]),
        min(box[2], container[2]),
        min(box[3], container[3]),
    ]


def _normalized_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(item, int | float) and 0 <= item <= 1 for item in value):
        return None
    bbox = [float(item) for item in value]
    return bbox if bbox[0] < bbox[2] and bbox[1] < bbox[3] else None


def _block_cell_coverage(block: list[float], cell: list[float]) -> float:
    intersection = max(0.0, min(block[2], cell[2]) - max(block[0], cell[0])) * max(
        0.0, min(block[3], cell[3]) - max(block[1], cell[1])
    )
    area = max(0.0, block[2] - block[0]) * max(0.0, block[3] - block[1])
    return intersection / area if area else 0.0


def _inside(box: list[float], container: list[float], page: PageParse) -> bool:
    tolerance_x = 1 / page.width
    tolerance_y = 1 / page.height
    return (
        box[0] >= container[0] - tolerance_x
        and box[1] >= container[1] - tolerance_y
        and box[2] <= container[2] + tolerance_x
        and box[3] <= container[3] + tolerance_y
    )


def _block_position(block: Any) -> tuple[float, float]:
    bbox = block.bbox
    return (bbox[1], bbox[0]) if bbox else (1.0, 1.0)


def table_word_evidence(page: PageParse, block_ids: set[str]) -> list[dict[str, Any]]:
    """Expose RapidOCR word boxes for table grounding without changing raw evidence."""
    raw = page.raw_evidence.get("word_results")
    if not isinstance(raw, list):
        return []
    output: list[dict[str, Any]] = []
    for block_id in sorted(block_ids, key=_block_id_number):
        index = _block_id_number(block_id)
        if index < 1 or index > len(raw) or not isinstance(raw[index - 1], list):
            continue
        for ordinal, value in enumerate(raw[index - 1], 1):
            if not isinstance(value, list) or len(value) < 3:
                continue
            text, score, polygon = value[0], value[1], value[2]
            try:
                left, top, right, bottom = _raw_bbox(polygon)
                bbox = [
                    left / page.width,
                    top / page.height,
                    right / page.width,
                    bottom / page.height,
                ]
                numeric_score = float(score)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            output.append(
                {
                    "id": f"{block_id}-w{ordinal}",
                    "block_id": block_id,
                    "page": page.page,
                    "text": str(text),
                    "score": numeric_score,
                    "bbox": bbox,
                }
            )
    return output


def _block_id_number(block_id: str) -> int:
    match = re.search(r"-b(\d+)$", block_id)
    return int(match.group(1)) if match else 0


def _union_boxes(boxes: list[list[float]]) -> list[float] | None:
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _positive_int(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


def _score(value: Any) -> float:
    score = float(value)
    if not 0 <= score <= 1:
        raise ValueError("Table model score must be between zero and one.")
    return score
