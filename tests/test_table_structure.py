from __future__ import annotations

import io
import json

from PIL import Image

from agentic_extractor.artifacts import build_local_artifacts
from agentic_extractor.models import (
    Block,
    LayoutBlockLink,
    LayoutRegion,
    ReadingOrderEvidence,
    TableCellEvidence,
    TableStructureEvidence,
)
from agentic_extractor.ocr import EngineProvenance, LocalParseResult
from agentic_extractor.parse import PageParse, document_markdown
from agentic_extractor.table_structure import (
    _matches_applied_correction,
    apply_table_reviews,
    enrich_page_tables,
    normalize_table_result,
    table_review_bbox,
    table_review_block_ids,
)


def _table_region() -> LayoutRegion:
    return LayoutRegion(
        id="p1-l1",
        page=1,
        cls_id=21,
        label="table",
        score=0.98,
        coordinate=[10, 20, 190, 180],
        bbox=[0.05, 0.10, 0.95, 0.90],
        raw_order=None,
    )


def test_table_structure_is_grounded_in_rapidocr_blocks() -> None:
    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[
            Block(id="p1-b1", page=1, text="Item", ocr_score=0.99, bbox=[0.10, 0.20, 0.45, 0.35]),
            Block(id="p1-b2", page=1, text="Price", ocr_score=0.98, bbox=[0.55, 0.20, 0.90, 0.35]),
            Block(id="p1-b3", page=1, text="Book", ocr_score=0.97, bbox=[0.10, 0.50, 0.45, 0.65]),
            Block(id="p1-b4", page=1, text="$12", ocr_score=0.96, bbox=[0.55, 0.50, 0.90, 0.65]),
        ],
        layout_regions=[_table_region()],
        raw_evidence={"texts": ["Item", "Price", "Book", "$12"]},
    )
    raw = {
        "classifier": {"label_names": ["wired_table"], "scores": [0.93]},
        "structure": {
            "bbox": [
                [10, 10, 80, 10, 80, 50, 10, 50],
                [100, 10, 170, 50],
                [10, 70, 80, 120],
                [100, 70, 170, 120],
            ],
            "structure": [
                "<html><body><table><tr><th></th><th></th></tr><tr><td></td><td></td></tr></table></body></html>"
            ],
            "structure_score": 0.91,
        },
    }

    table = normalize_table_result("p1-l1", raw, page)
    original = page.raw_evidence.copy()
    enrich_page_tables(page, [table])

    assert table.status == "valid"
    assert [[cell.row, cell.column] for cell in table.cells] == [[1, 1], [1, 2], [2, 1], [2, 2]]
    assert [cell.source_block_ids for cell in table.cells] == [
        ["p1-b1"],
        ["p1-b2"],
        ["p1-b3"],
        ["p1-b4"],
    ]
    assert table.markdown == (
        "<table><tr><th>Item</th><th>Price</th></tr><tr><td>Book</td><td>$12</td></tr></table>"
    )
    assert page.raw_evidence == original
    assert page.table_structures[0].classifier_model == "PP-LCNet_x1_0_table_cls"


def test_two_four_column_infusion_tables_remain_html_tables() -> None:
    regions = [
        LayoutRegion(
            id="p4-l6",
            page=4,
            cls_id=21,
            label="table",
            score=0.96,
            coordinate=[0, 0, 400, 200],
            bbox=[0, 0, 1, 0.4],
        ),
        LayoutRegion(
            id="p4-l7",
            page=4,
            cls_id=21,
            label="table",
            score=0.96,
            coordinate=[0, 220, 400, 420],
            bbox=[0, 0.44, 1, 0.84],
        ),
    ]
    labels = ["Step", "Duration", "FENTANYL", "BACLOFEN"]
    blocks = [
        Block(
            id=f"p4-b{index + table_index * 4}",
            page=4,
            text=label,
            ocr_score=0.99,
            bbox=[
                (index - 1) * 0.25 + 0.02,
                top + 0.02,
                index * 0.25 - 0.02,
                top + 0.18,
            ],
        )
        for table_index, top in enumerate((0.0, 0.44))
        for index, label in enumerate(labels, 1)
    ]
    page = PageParse(
        page=4,
        width=400,
        height=500,
        blocks=blocks,
        layout_regions=regions,
    )
    raw = {
        "classifier": {"label_names": ["wireless_table"], "scores": [0.9]},
        "structure_model": "SLANet_plus",
        "structure_geometry_valid": True,
        "structure": {
            "bbox": [
                [0, 0, 100, 100],
                [100, 0, 200, 100],
                [200, 0, 300, 100],
                [300, 0, 400, 100],
            ],
            "structure": ["<table><tr><td></td><td></td><td></td><td></td></tr></table>"],
            "structure_score": 0.99,
        },
    }

    tables = [normalize_table_result(region.id, raw, page) for region in regions]
    enrich_page_tables(page, tables)
    markdown = document_markdown([page])

    assert [table.status for table in tables] == ["valid", "valid"]
    assert [max(cell.column for cell in table.cells) for table in tables] == [4, 4]
    assert markdown.count("<table>") == 2
    assert markdown.count("<td>FENTANYL</td><td>BACLOFEN</td>") == 2


def test_table_enrichment_preserves_established_column_major_block_order() -> None:
    page = PageParse(
        page=1,
        width=100,
        height=100,
        blocks=[
            Block(id="p1-b1", page=1, text="left top", bbox=[0.0, 0.0, 0.4, 0.1]),
            Block(id="p1-b2", page=1, text="left bottom", bbox=[0.0, 0.5, 0.4, 0.6]),
            Block(id="p1-b3", page=1, text="right top", bbox=[0.6, 0.0, 1.0, 0.1]),
            Block(id="p1-b4", page=1, text="right bottom", bbox=[0.6, 0.5, 1.0, 0.6]),
        ],
    )

    enrich_page_tables(page, [])

    assert [chunk.text for chunk in page.chunks] == [
        "left top",
        "left bottom",
        "right top",
        "right bottom",
    ]


def test_table_structure_accepts_page_space_cell_coordinates_without_double_offset() -> None:
    region = LayoutRegion(
        id="p1-l1",
        page=1,
        cls_id=21,
        label="table",
        score=0.98,
        coordinate=[100, 100, 300, 250],
        bbox=[0.25, 0.25, 0.75, 0.625],
    )
    page = PageParse(
        page=1,
        width=400,
        height=400,
        blocks=[Block(id="p1-b1", page=1, text="Item", bbox=[0.3, 0.3, 0.6, 0.4])],
        layout_regions=[region],
    )

    table = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wired_table"], "scores": [0.9]},
            "structure": {
                "bbox": [[110, 110, 250, 170]],
                "structure": ["<table><tr><td></td></tr></table>"],
                "structure_score": 0.9,
            },
        },
        page,
    )

    assert table.status == "valid"
    assert table.cells[0].bbox == [0.275, 0.275, 0.625, 0.425]
    assert table.cells[0].source_block_ids == ["p1-b1"]


def test_merged_table_uses_generated_html_and_invalid_structure_requires_review() -> None:
    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[Block(id="p1-b1", page=1, text="Total", bbox=[0.1, 0.2, 0.8, 0.4])],
        layout_regions=[_table_region()],
        reading_order_evidence=ReadingOrderEvidence(page=1, ordered_region_ids=["p1-l1"]),
    )
    merged = {
        "classifier": {"label_names": ["wireless_table"], "scores": [0.88]},
        "structure": {
            "bbox": [[10, 10, 170, 50]],
            "structure": ["<table><tr><td colspan='2'></td></tr></table>"],
            "structure_score": 0.86,
        },
    }
    table = normalize_table_result("p1-l1", merged, page)
    assert table.style == "wireless"
    assert table.cells[0].column_span == 2
    assert table.markdown == '<table><tr><td colspan="2">Total</td></tr></table>'

    invalid = normalize_table_result(
        "p1-l1",
        {**merged, "structure": {**merged["structure"], "bbox": []}},
        page,
    )
    assert invalid.status == "invalid"
    assert invalid.review_required is True


def test_invalid_table_keeps_provisional_cells_only_as_review_evidence() -> None:
    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[
            Block(id="p1-b1", page=1, text="Total", bbox=[0.1, 0.2, 0.8, 0.4]),
            Block(id="p1-b2", page=1, text="spillover", bbox=[0.1, 0.8, 0.8, 0.85]),
        ],
        layout_regions=[_table_region()],
    )
    table = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wireless_table"], "scores": [0.88]},
            "structure": {
                "bbox": [[10, 10, 170, 50]],
                "structure": ["<table><tr><td></td></tr></table>"],
                "structure_score": 0.86,
            },
        },
        page,
    )

    enrich_page_tables(page, [table])

    assert table.status == "invalid"
    assert table.review_required is True
    assert len(table.cells) == 1
    assert "<table>" not in document_markdown([page])


def test_table_structure_rejects_cells_outside_parent_region() -> None:
    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[Block(id="p1-b1", page=1, text="Total", bbox=[0.1, 0.2, 0.8, 0.4])],
        layout_regions=[_table_region()],
        raw_evidence={"texts": ["Total"]},
    )
    raw = {
        "classifier": {"label_names": ["wired_table"], "scores": [0.9]},
        "structure": {
            "bbox": [[10, 10, 170, 190]],
            "structure": ["<table><tr><td></td></tr></table>"],
            "structure_score": 0.9,
        },
    }

    table = normalize_table_result("p1-l1", raw, page)

    assert table.status == "invalid"
    assert len(table.cells) == 1
    assert table.markdown == ""
    enrich_page_tables(page, [table])
    assert "<table>" not in document_markdown([page])
    assert "outside its parent table region" in table.warnings[0]


def test_table_structure_uses_overlapping_model_boxes_only_as_assignment_hints() -> None:
    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[
            Block(id="p1-b1", page=1, text="Item", bbox=[0.1, 0.2, 0.4, 0.4]),
            Block(id="p1-b2", page=1, text="Price", bbox=[0.5, 0.2, 0.8, 0.4]),
        ],
        layout_regions=[_table_region()],
    )
    raw = {
        "classifier": {"label_names": ["wired_table"], "scores": [0.9]},
        "structure": {
            "bbox": [[10, 10, 120, 60], [80, 10, 170, 60]],
            "structure": ["<table><tr><td></td><td></td></tr></table>"],
            "structure_score": 0.9,
        },
    }

    table = normalize_table_result("p1-l1", raw, page)

    assert table.status == "valid"
    assert [cell.source_block_ids for cell in table.cells] == [["p1-b1"], ["p1-b2"]]
    assert not any("overlap" in warning for warning in table.warnings)


def test_table_structure_clips_minor_crop_edge_overshoot_and_records_fallback_model() -> None:
    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[Block(id="p1-b1", page=1, text="Total", bbox=[0.1, 0.2, 0.8, 0.4])],
        layout_regions=[_table_region()],
    )

    table = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wired_table"], "scores": [0.9]},
            "structure_model": "SLANet_plus",
            "structure_fallback": True,
            "structure": {
                "bbox": [[-2, 8, 184, 8, 184, 164, -2, 164]],
                "structure": ["<table><tr><td></td></tr></table>"],
                "structure_score": 0.9,
            },
        },
        page,
    )

    assert table.status == "valid"
    assert table.style == "wireless"
    assert table.structure_model == "SLANet_plus"
    assert table.cells[0].bbox == [0.05, 0.14, 0.95, 0.9]


def test_v3_reading_order_record_preserves_unordered_and_unmatched_evidence() -> None:
    from agentic_extractor.layout import enrich_page_layout

    ordered = LayoutRegion(
        id="p1-l1",
        page=1,
        cls_id=22,
        label="text",
        score=0.9,
        coordinate=[0, 0, 100, 40],
        bbox=[0, 0, 0.5, 0.2],
        raw_order=4,
        normalized_order=1,
    )
    unordered = _table_region().model_copy(update={"id": "p1-l2"})
    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[
            Block(id="p1-b2", page=1, text="outside", bbox=[0.6, 0.9, 0.9, 0.98]),
            Block(id="p1-b1", page=1, text="inside", bbox=[0.05, 0.05, 0.4, 0.15]),
        ],
    )

    enrich_page_layout(page, [ordered, unordered])

    assert page.reading_order_evidence.source == "PP-DocLayoutV3"
    assert page.reading_order_evidence.ordered_region_ids == ["p1-l1"]
    assert page.reading_order_evidence.unordered_region_ids == ["p1-l2"]
    assert page.reading_order_evidence.unmatched_block_ids == ["p1-b2"]
    assert page.reading_order_evidence.status == "ambiguous"


def test_sol_table_correction_requires_existing_rapidocr_grounding() -> None:
    from types import SimpleNamespace

    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[Block(id="p1-b1", page=1, text="Total", bbox=[0.1, 0.2, 0.8, 0.4])],
        layout_regions=[_table_region()],
    )
    table = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wireless_table"], "scores": [0.9]},
            "structure": {
                "bbox": [[10, 10, 170, 50]],
                "structure": ["<table><tr><td></td></tr></table>"],
                "structure_score": 0.8,
            },
        },
        page,
    )
    enrich_page_tables(page, [table])

    audits, warnings = apply_table_reviews(
        [page],
        [
            SimpleNamespace(
                page=1,
                table_id=table.id,
                outcome="corrected",
                visible_cell_count=1,
                warning=None,
                cells=[
                    SimpleNamespace(
                        row=1,
                        column=1,
                        row_span=1,
                        column_span=1,
                        tag="td",
                        text="Total",
                        source_block_ids=["missing"],
                    )
                ],
            )
        ],
    )

    assert audits[0]["status"] == "unresolved"
    assert "lacked valid RapidOCR grounding" in warnings[0]
    assert page.table_structures[0].review_required is True
    assert all(chunk.type != "table" for chunk in page.chunks)


def test_sol_cannot_promote_locally_invalid_region_from_two_cell_row_alone() -> None:
    from types import SimpleNamespace

    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[
            Block(
                id=f"p1-b{index}",
                page=1,
                text=f"Field {index}",
                bbox=[0.1 if index <= 3 else 0.5, 0.2, 0.4 if index <= 3 else 0.8, 0.4],
            )
            for index in range(1, 7)
        ],
        layout_regions=[_table_region()],
    )
    invalid = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wired_table"], "scores": [0.9]},
            "structure": {
                "bbox": [],
                "structure": ["<table><tr><td></td></tr></table>"],
                "structure_score": 0.8,
            },
        },
        page,
    )
    enrich_page_tables(page, [invalid])
    correction = SimpleNamespace(
        page=1,
        table_id=invalid.id,
        outcome="corrected",
        visible_cell_count=2,
        warning=None,
        cells=[
            SimpleNamespace(
                row=1,
                column=1,
                row_span=1,
                column_span=1,
                tag="td",
                text="invented value",
                source_block_ids=["p1-b1", "p1-b2", "p1-b3"],
            ),
            SimpleNamespace(
                row=1,
                column=2,
                row_span=1,
                column_span=1,
                tag="td",
                text="another invention",
                source_block_ids=["p1-b4", "p1-b5", "p1-b6"],
            ),
        ],
    )

    audits, warnings = apply_table_reviews([page], [correction])

    assert audits[0]["status"] == "rejected"
    assert warnings == []
    assert page.table_structures[0].review_required is False


def test_table_repair_excludes_nearby_blocks_linked_to_a_different_layout_region() -> None:
    from types import SimpleNamespace

    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[
            Block(id="p1-b1", page=1, text="Item", bbox=[0.1, 0.2, 0.4, 0.4]),
            Block(id="p1-b2", page=1, text="Footer", bbox=[0.1, 0.3, 0.4, 0.35]),
        ],
        layout_regions=[_table_region()],
        layout_block_links=[
            LayoutBlockLink(
                page=1,
                block_id="p1-b1",
                primary_region_id="p1-l1",
                region_ids=["p1-l1"],
                block_coverage=1,
            ),
            LayoutBlockLink(
                page=1,
                block_id="p1-b2",
                primary_region_id="p1-footer",
                region_ids=["p1-footer", "p1-l1"],
                block_coverage=1,
            ),
        ],
    )
    invalid = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wired_table"], "scores": [0.9]},
            "structure": {"bbox": [], "structure": ["<table></table>"], "structure_score": 0.1},
        },
        page,
    )
    enrich_page_tables(page, [invalid])
    review = SimpleNamespace(
        page=1,
        table_id=invalid.id,
        outcome="corrected",
        visible_cell_count=1,
        warning=None,
        cells=[
            SimpleNamespace(
                row=1,
                column=1,
                row_span=1,
                column_span=1,
                tag="td",
                text="ignored",
                source_block_ids=["p1-b1"],
            )
        ],
    )

    audits, warnings = apply_table_reviews([page], [review])

    assert warnings == []
    assert audits[0]["status"] == "accepted"
    assert page.table_structures[0].cells[0].source_text == "Item"


def test_table_repair_can_audit_layout_spillover_without_forcing_it_into_a_cell() -> None:
    from types import SimpleNamespace

    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[
            Block(id="p1-b1", page=1, text="Item", bbox=[0.1, 0.2, 0.4, 0.3]),
            Block(id="p1-b2", page=1, text="Unrelated note", bbox=[0.1, 0.7, 0.4, 0.8]),
        ],
        layout_regions=[_table_region()],
        layout_block_links=[
            LayoutBlockLink(
                page=1,
                block_id=block_id,
                primary_region_id="p1-l1",
                region_ids=["p1-l1"],
                block_coverage=1,
            )
            for block_id in ("p1-b1", "p1-b2")
        ],
    )
    invalid = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wireless_table"], "scores": [0.9]},
            "structure": {"bbox": [], "structure": ["<table></table>"], "structure_score": 0.1},
        },
        page,
    )
    enrich_page_tables(page, [invalid])
    review = SimpleNamespace(
        page=1,
        table_id=invalid.id,
        outcome="corrected",
        visible_cell_count=1,
        warning=None,
        excluded_source_block_ids=["p1-b2"],
        cells=[
            SimpleNamespace(
                row=1,
                column=1,
                row_span=1,
                column_span=1,
                tag="td",
                source_block_ids=["p1-b1"],
            )
        ],
    )

    audits, warnings = apply_table_reviews([page], [review])

    assert warnings == []
    assert audits[0]["excluded_source_block_ids"] == ["p1-b2"]
    assert page.table_structures[0].bbox == [0.1, 0.2, 0.4, 0.3]


def test_table_repair_can_split_one_rapidocr_line_using_immutable_word_boxes() -> None:
    from types import SimpleNamespace

    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[Block(id="p1-b1", page=1, text="NPI: 123 TIN: 456", bbox=[0.1, 0.2, 0.9, 0.3])],
        layout_regions=[_table_region()],
        raw_evidence={
            "word_results": [
                [
                    ["NPI: 123", 0.99, [[20, 40], [80, 40], [80, 60], [20, 60]]],
                    ["TIN: 456", 0.98, [[110, 40], [180, 40], [180, 60], [110, 60]]],
                ]
            ]
        },
    )
    invalid = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wireless_table"], "scores": [0.9]},
            "structure": {"bbox": [], "structure": ["<table></table>"], "structure_score": 0.1},
        },
        page,
    )
    enrich_page_tables(page, [invalid])
    review = SimpleNamespace(
        page=1,
        table_id=invalid.id,
        outcome="corrected",
        visible_cell_count=2,
        warning=None,
        excluded_source_block_ids=[],
        excluded_source_word_ids=[],
        cells=[
            SimpleNamespace(
                row=1,
                column=1,
                row_span=1,
                column_span=1,
                tag="td",
                source_block_ids=["p1-b1"],
                source_word_ids=["p1-b1-w1"],
            ),
            SimpleNamespace(
                row=1,
                column=2,
                row_span=1,
                column_span=1,
                tag="td",
                source_block_ids=["p1-b1"],
                source_word_ids=["p1-b1-w2"],
            ),
        ],
    )

    audits, warnings = apply_table_reviews([page], [review])

    assert warnings == []
    assert audits[0]["status"] == "accepted"
    assert [cell.source_text for cell in page.table_structures[0].cells] == [
        "NPI: 123",
        "TIN: 456",
    ]
    assert [cell.source_block_ids for cell in page.table_structures[0].cells] == [
        ["p1-b1"],
        ["p1-b1"],
    ]


def test_table_repair_may_omit_review_window_words_outside_detected_table() -> None:
    from types import SimpleNamespace

    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[
            Block(id="p1-b1", page=1, text="Value", bbox=[0.1, 0.2, 0.4, 0.3]),
            Block(id="p1-b2", page=1, text="Adjacent note", bbox=[0.1, 0.91, 0.4, 0.94]),
        ],
        layout_regions=[_table_region()],
        layout_block_links=[
            LayoutBlockLink(
                page=1,
                block_id=block_id,
                primary_region_id="p1-l1",
                region_ids=["p1-l1"],
                block_coverage=1,
            )
            for block_id in ("p1-b1", "p1-b2")
        ],
        raw_evidence={
            "word_results": [
                [["Value", 0.99, [[20, 40], [80, 40], [80, 60], [20, 60]]]],
                [["Adjacent note", 0.99, [[20, 182], [80, 182], [80, 188], [20, 188]]]],
            ]
        },
    )
    table = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wireless_table"], "scores": [0.9]},
            "structure": {"bbox": [], "structure": ["<table></table>"], "structure_score": 0.1},
        },
        page,
    )
    enrich_page_tables(page, [table])
    review = SimpleNamespace(
        page=1,
        table_id=table.id,
        outcome="corrected",
        visible_cell_count=1,
        warning=None,
        excluded_source_block_ids=[],
        excluded_source_word_ids=[],
        cells=[
            SimpleNamespace(
                row=1,
                column=1,
                row_span=1,
                column_span=1,
                tag="td",
                text="",
                bbox=None,
                source_block_ids=[],
                source_word_ids=["p1-b1-w1"],
            )
        ],
    )

    audits, warnings = apply_table_reviews([page], [review])

    assert warnings == []
    assert audits[0]["status"] == "accepted"
    assert page.table_structures[0].cells[0].source_text == "Value"


def test_valid_table_repair_requires_local_cell_evidence_not_detector_spillover() -> None:
    from types import SimpleNamespace

    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[
            Block(id="p1-b1", page=1, text="Table value", bbox=[0.1, 0.2, 0.4, 0.3]),
            Block(id="p1-b2", page=1, text="Specialty", bbox=[0.1, 0.8, 0.4, 0.85]),
        ],
        layout_regions=[_table_region()],
        layout_block_links=[
            LayoutBlockLink(
                page=1,
                block_id=block_id,
                primary_region_id="p1-l1",
                region_ids=["p1-l1"],
                block_coverage=1,
            )
            for block_id in ("p1-b1", "p1-b2")
        ],
        raw_evidence={
            "word_results": [
                [["Table value", 0.99, [[20, 40], [80, 40], [80, 60], [20, 60]]]],
                [["Specialty", 0.99, [[20, 160], [80, 160], [80, 170], [20, 170]]]],
            ]
        },
    )
    table = TableStructureEvidence(
        id="p1-l1-table",
        page=1,
        layout_region_id="p1-l1",
        bbox=[0.05, 0.1, 0.95, 0.9],
        style="wireless",
        classifier_score=0.9,
        structure_score=0.9,
        classifier_model="PP-LCNet_x1_0_table_cls",
        structure_model="SLANet_plus",
        cells=[
            TableCellEvidence(
                id="p1-l1-cell-1",
                row=1,
                column=1,
                bbox=[0.1, 0.2, 0.4, 0.3],
                source_block_ids=["p1-b1"],
                source_text="Table value",
                raw_scores=[0.99],
            )
        ],
        markdown="<table><tr><td>Table value</td></tr></table>",
        status="valid",
        review_required=True,
    )
    page.table_structures = [table]
    review = SimpleNamespace(
        page=1,
        table_id=table.id,
        outcome="corrected",
        visible_cell_count=1,
        warning=None,
        excluded_source_block_ids=[],
        excluded_source_word_ids=[],
        cells=[
            SimpleNamespace(
                row=1,
                column=1,
                row_span=1,
                column_span=1,
                tag="td",
                text="",
                bbox=None,
                source_block_ids=["p1-b1"],
                source_word_ids=["p1-b1-w1"],
            )
        ],
    )

    audits, warnings = apply_table_reviews([page], [review])

    assert warnings == []
    assert audits[0]["status"] == "accepted"
    assert page.table_structures[0].cells[0].source_text == "Table value"


def test_table_repair_accepts_nonoverlapping_mixed_word_and_block_grounding() -> None:
    from types import SimpleNamespace

    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[
            Block(id="p1-b1", page=1, text="NPI 123 TIN 456", bbox=[0.1, 0.2, 0.9, 0.3]),
            Block(id="p1-b2", page=1, text="Address", bbox=[0.1, 0.4, 0.9, 0.5]),
        ],
        layout_regions=[_table_region()],
        raw_evidence={
            "word_results": [
                [
                    ["NPI 123", 0.99, [[20, 40], [80, 40], [80, 60], [20, 60]]],
                    ["TIN 456", 0.99, [[110, 40], [180, 40], [180, 60], [110, 60]]],
                ],
                [["Address", 0.99, [[20, 80], [180, 80], [180, 100], [20, 100]]]],
            ]
        },
    )
    table = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wireless_table"], "scores": [0.9]},
            "structure": {"bbox": [], "structure": ["<table></table>"], "structure_score": 0.1},
        },
        page,
    )
    enrich_page_tables(page, [table])
    review = SimpleNamespace(
        page=1,
        table_id=table.id,
        outcome="corrected",
        visible_cell_count=2,
        warning=None,
        excluded_source_block_ids=[],
        excluded_source_word_ids=[],
        cells=[
            SimpleNamespace(
                row=1,
                column=1,
                row_span=1,
                column_span=1,
                tag="td",
                text="",
                bbox=None,
                source_block_ids=[],
                source_word_ids=["p1-b1-w1", "p1-b1-w2"],
            ),
            SimpleNamespace(
                row=2,
                column=1,
                row_span=1,
                column_span=1,
                tag="td",
                text="",
                bbox=None,
                source_block_ids=["p1-b2"],
                source_word_ids=[],
            ),
        ],
    )

    audits, warnings = apply_table_reviews([page], [review])

    assert warnings == []
    assert audits[0]["status"] == "accepted"
    assert [cell.source_text for cell in page.table_structures[0].cells] == [
        "NPI 123 TIN 456",
        "Address",
    ]


def test_grounded_table_grid_is_authoritative_over_redundant_visible_count() -> None:
    from types import SimpleNamespace

    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[Block(id="p1-b1", page=1, text="Value", bbox=[0.1, 0.2, 0.4, 0.3])],
        layout_regions=[_table_region()],
    )
    table = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wireless_table"], "scores": [0.9]},
            "structure": {"bbox": [], "structure": ["<table></table>"], "structure_score": 0.1},
        },
        page,
    )
    enrich_page_tables(page, [table])
    review = SimpleNamespace(
        page=1,
        table_id=table.id,
        outcome="corrected",
        visible_cell_count=2,
        warning=None,
        excluded_source_block_ids=[],
        excluded_source_word_ids=[],
        cells=[
            SimpleNamespace(
                row=1,
                column=1,
                row_span=1,
                column_span=1,
                tag="td",
                text="",
                bbox=None,
                source_block_ids=["p1-b1"],
                source_word_ids=[],
            )
        ],
    )

    audits, warnings = apply_table_reviews([page], [review])

    assert warnings == []
    assert audits[0]["status"] == "accepted"
    assert len(page.table_structures[0].cells) == 1
    assert _matches_applied_correction(page.table_structures[0], review)


def test_table_repair_preserves_visually_grounded_blank_cells() -> None:
    from types import SimpleNamespace

    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[Block(id="p1-b1", page=1, text="Value", bbox=[0.1, 0.2, 0.4, 0.3])],
        layout_regions=[_table_region()],
    )
    invalid = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wireless_table"], "scores": [0.9]},
            "structure": {"bbox": [], "structure": ["<table></table>"], "structure_score": 0.1},
        },
        page,
    )
    enrich_page_tables(page, [invalid])
    review = SimpleNamespace(
        page=1,
        table_id=invalid.id,
        outcome="corrected",
        visible_cell_count=2,
        warning=None,
        excluded_source_block_ids=[],
        excluded_source_word_ids=[],
        cells=[
            SimpleNamespace(
                row=1,
                column=1,
                row_span=1,
                column_span=1,
                tag="td",
                text="ignored",
                bbox=None,
                source_block_ids=["p1-b1"],
                source_word_ids=[],
            ),
            SimpleNamespace(
                row=1,
                column=2,
                row_span=1,
                column_span=1,
                tag="td",
                text="",
                bbox=[0.5, 0.2, 0.8, 0.3],
                source_block_ids=[],
                source_word_ids=[],
            ),
        ],
    )

    audits, warnings = apply_table_reviews([page], [review])

    table = page.table_structures[0]
    assert warnings == []
    assert audits[0]["status"] == "accepted"
    assert len(table.cells) == 2
    assert table.cells[1].source_text == ""
    assert table.cells[1].bbox == [0.5, 0.2, 0.8, 0.3]
    assert table.markdown == "<table><tr><td>Value</td><td></td></tr></table>"


def test_sol_can_reject_a_false_positive_table_without_requesting_review() -> None:
    from types import SimpleNamespace

    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[Block(id="p1-b1", page=1, text="Name: Example", bbox=[0.1, 0.2, 0.8, 0.4])],
        layout_regions=[_table_region()],
    )
    table = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wireless_table"], "scores": [0.9]},
            "structure": {
                "bbox": [[10, 10, 170, 50]],
                "structure": ["<table><tr><td></td></tr></table>"],
                "structure_score": 0.8,
            },
        },
        page,
    )
    enrich_page_tables(page, [table])

    audits, warnings = apply_table_reviews(
        [page],
        [
            SimpleNamespace(
                page=1,
                table_id=table.id,
                outcome="not_table",
                visible_cell_count=0,
                warning="Single-record key-value region.",
                cells=[],
            )
        ],
    )

    assert warnings == []
    assert audits[0]["status"] == "rejected"
    assert page.table_structures[0].status == "invalid"
    assert page.table_structures[0].review_required is False
    assert page.table_structures[0].cells == []
    assert all(chunk.type != "table" for chunk in page.chunks)


def test_confirmed_table_must_not_supply_a_different_grid() -> None:
    from types import SimpleNamespace

    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[Block(id="p1-b1", page=1, text="Value", bbox=[0.1, 0.2, 0.8, 0.4])],
        layout_regions=[_table_region()],
    )
    table = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wireless_table"], "scores": [0.9]},
            "structure": {
                "bbox": [[10, 10, 170, 50]],
                "structure": ["<table><tr><td></td></tr></table>"],
                "structure_score": 0.8,
            },
        },
        page,
    )
    enrich_page_tables(page, [table])

    audits, warnings = apply_table_reviews(
        [page],
        [
            SimpleNamespace(
                page=1,
                table_id=table.id,
                outcome="confirmed",
                visible_cell_count=1,
                warning=None,
                cells=[SimpleNamespace(row=1, column=1)],
            )
        ],
    )

    assert audits[0]["status"] == "unresolved"
    assert "confirmed outcome also supplied replacement cells" in warnings[0]
    assert page.table_structures[0].review_required is True


def test_table_review_window_recovers_clipped_edge_rows_without_crossing_next_table() -> None:
    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[
            Block(id="p1-b1", page=1, text="Title", bbox=[0.1, 0.16, 0.8, 0.19]),
            Block(id="p1-b2", page=1, text="Body", bbox=[0.1, 0.22, 0.8, 0.3]),
            Block(id="p1-b3", page=1, text="Next", bbox=[0.1, 0.57, 0.8, 0.59]),
        ],
        layout_regions=[_table_region()],
    )
    first = TableStructureEvidence(
        id="p1-l1-table",
        page=1,
        layout_region_id="p1-l1",
        bbox=[0.05, 0.2, 0.95, 0.5],
        style="wireless",
        classifier_score=0.9,
        structure_score=0.9,
        structure_model="SLANet_plus",
        status="valid",
    )
    second = first.model_copy(
        update={"id": "p1-l2-table", "layout_region_id": "p1-l2", "bbox": [0.05, 0.55, 0.95, 0.8]}
    )
    page.table_structures = [first, second]

    assert table_review_bbox(page, first) == [0.04, 0.14, 0.96, 0.55]
    assert table_review_bbox(page, second)[1] == 0.55
    assert table_review_block_ids(page, first) == {"p1-b1", "p1-b2"}


def test_table_and_order_evidence_are_exported_in_public_and_audit_contracts() -> None:
    image = Image.new("RGB", (200, 200), "white")
    image_bytes = io.BytesIO()
    image.save(image_bytes, "JPEG")
    page = PageParse(
        page=1,
        width=200,
        height=200,
        image_bytes=image_bytes.getvalue(),
        blocks=[Block(id="p1-b1", page=1, text="Total", bbox=[0.1, 0.2, 0.8, 0.4])],
        layout_regions=[_table_region()],
        reading_order_evidence=ReadingOrderEvidence(page=1, ordered_region_ids=["p1-l1"]),
    )
    table = normalize_table_result(
        "p1-l1",
        {
            "classifier": {"label_names": ["wired_table"], "scores": [0.9]},
            "structure": {
                "bbox": [[10, 10, 170, 50]],
                "structure": ["<table><tr><td></td></tr></table>"],
                "structure_score": 0.8,
            },
        },
        page,
    )
    enrich_page_tables(page, [table])
    result = LocalParseResult(
        document_metadata={"filename": "table.jpg"},
        selected_pages=[1],
        pages=[page],
        markdown=f"<!-- page: 1 -->\n\n{table.markdown}",
        engine=EngineProvenance("RapidOCR", "test", "CPU"),
        timings={"ocr_seconds": 0.1, "table_structure_seconds": 0.2},
    )

    artifacts = build_local_artifacts(result)
    parsed = json.loads(artifacts.parse_result)

    table_node = parsed["structure"]["children"][0]["children"][0]
    assert table_node["type"] == "table"
    assert table_node["children"][0]["row"] == 0
    assert "table_structures" in artifacts.manifest["pages"][0]
    assert artifacts.annotated_pdf.startswith(b"%PDF")
