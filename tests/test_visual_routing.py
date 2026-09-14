from agentic_extractor.models import (
    Block,
    LayoutRegion,
    LocalCheckboxCandidate,
    LocalRedactionCandidate,
    TableStructureEvidence,
)
from agentic_extractor.parse import PageParse
from agentic_extractor.visual_routing import plan_visual_review_regions


def _page(*blocks: Block, layout_signals: dict[str, object] | None = None) -> PageParse:
    return PageParse(
        page=1,
        width=1000,
        height=1400,
        blocks=list(blocks),
        layout_signals=layout_signals or {},
    )


def _block(identifier: str, score: float, bbox: list[float]) -> Block:
    return Block(id=identifier, page=1, text=identifier, ocr_score=score, bbox=bbox)


def test_confident_plain_page_needs_no_high_detail_region() -> None:
    page = _page(_block("p1-b1", 0.99, [0.1, 0.1, 0.4, 0.2]))

    assert plan_visual_review_regions(page) == []


def test_low_confidence_block_is_padded_and_grounded() -> None:
    page = _page(_block("p1-b1", 0.849, [0.1, 0.2, 0.4, 0.3]))

    region = plan_visual_review_regions(page)[0]

    assert region.id == "p1-vr1"
    assert region.bbox == [0.08, 0.18, 0.42, 0.32]
    assert region.reason_codes == ["low_ocr_confidence"]
    assert region.source_block_ids == ["p1-b1"]
    assert not region.page_wide


def test_checkbox_region_includes_control_and_grounded_label() -> None:
    label = _block("p1-b1", 0.99, [0.2, 0.2, 0.6, 0.25])
    page = _page(label)
    page.local_checkbox_candidates = [
        LocalCheckboxCandidate(
            id="p1-cv1",
            page=1,
            state="CHECKED",
            control_bbox=[0.1, 0.2, 0.156, 0.24],
            detector_score=0.9,
            border_coverage=0.8,
            interior_ink_ratio=0.2,
            label_block_id="p1-b1",
            source_text="p1-b1",
            ocr_score=0.99,
            ocr_grounding_unique=True,
            engine_version="5.0.0",
        )
    ]

    region = plan_visual_review_regions(page)[0]

    assert region.bbox == [0.08, 0.18, 0.62, 0.27]
    assert region.source_checkbox_ids == ["p1-cv1"]
    assert region.source_block_ids == ["p1-b1"]


def test_ungrounded_checkbox_proposal_does_not_create_a_high_detail_crop() -> None:
    page = _page(_block("p1-b1", 0.99, [0.2, 0.2, 0.6, 0.25]))
    page.local_checkbox_candidates = [
        LocalCheckboxCandidate(
            id="p1-cv1",
            page=1,
            state="CHECKED",
            control_bbox=[0.1, 0.2, 0.156, 0.24],
            detector_score=0.9,
            border_coverage=0.8,
            interior_ink_ratio=0.2,
            source_text="ambiguous header",
            ocr_score=0.99,
            ocr_grounding_unique=False,
            risks=["ambiguous RapidOCR label"],
            engine_version="5.0.0",
        )
    ]

    assert plan_visual_review_regions(page) == []


def test_redaction_candidate_routes_its_mask_and_source_label() -> None:
    label = _block("p1-b1", 0.99, [0.1, 0.2, 0.25, 0.23])
    page = _page(label)
    page.local_redaction_candidates = [
        LocalRedactionCandidate(
            id="p1-red1",
            page=1,
            bbox=[0.27, 0.2, 0.47, 0.23],
            detector_score=0.9,
            fill_ratio=0.8,
            context="labeled_value",
            label_block_id="p1-b1",
            engine_version="test",
        )
    ]

    region = plan_visual_review_regions(page)[0]

    assert region.reason_codes == ["redaction_candidate"]
    assert region.source_redaction_ids == ["p1-red1"]
    assert region.source_block_ids == ["p1-b1"]


def test_missing_geometry_and_forced_page_use_page_wide_region() -> None:
    page = _page(Block(id="p1-b1", page=1, text="uncertain", ocr_score=None))

    missing = plan_visual_review_regions(page)[0]
    forced = plan_visual_review_regions(page, forced=True)[0]

    assert missing.bbox == [0.0, 0.0, 1.0, 1.0]
    assert missing.page_wide
    assert "missing_uncertain_geometry" in missing.reason_codes
    assert forced.reason_codes == ["user_forced"]


def test_many_regions_are_merged_without_losing_sources() -> None:
    blocks = [
        _block(
            f"p1-b{index}",
            0.5,
            [
                0.05 + ((index - 1) % 4) * 0.22,
                0.05 + ((index - 1) // 4) * 0.22,
                0.07 + ((index - 1) % 4) * 0.22,
                0.07 + ((index - 1) // 4) * 0.22,
            ],
        )
        for index in range(1, 17)
    ]

    regions = plan_visual_review_regions(_page(*blocks), max_regions=12)

    assert len(regions) == 12
    assert {source for region in regions for source in region.source_block_ids} == {
        block.id for block in blocks
    }


def test_page_level_signals_use_low_detail_overview_without_a_page_spanning_crop() -> None:
    blocks = [
        _block("p1-b1", 0.99, [0.1, 0.1, 0.4, 0.2]),
        _block("p1-b2", 0.99, [0.6, 0.1, 0.9, 0.2]),
    ]
    page = _page(*blocks, layout_signals={"columns_detected": 2})

    regions = plan_visual_review_regions(
        page,
        quality={"blur_warning": True, "skew_degrees": 1.5, "estimated_dpi": 120},
    )

    assert regions == []


def test_suspicious_critical_token_routes_only_its_block() -> None:
    page = _page(_block("p1-b1", 0.99, [0.1, 0.2, 0.4, 0.3]))
    page.blocks[0].text = "Date: 3|31983"

    region = plan_visual_review_regions(page)[0]

    assert region.source_block_ids == ["p1-b1"]
    assert region.reason_codes == ["high_risk_ocr_pattern"]
    assert not region.page_wide


def test_high_confidence_medical_identifier_ambiguity_routes_for_visual_review() -> None:
    page = _page(
        Block(
            id="p1-b1",
            page=1,
            text="195.1-458.0 Orthostatic hypotension, Active",
            ocr_score=0.99332,
            bbox=[0.1, 0.2, 0.8, 0.23],
        ),
        Block(
            id="p1-b2",
            page=1,
            text="173.9-443.9 Peripheral vascular disease, unspecified, Active",
            ocr_score=0.99235,
            bbox=[0.1, 0.24, 0.9, 0.27],
        ),
    )

    regions = plan_visual_review_regions(page)

    assert len(regions) == 1
    assert regions[0].reason_codes == ["high_risk_ocr_pattern"]
    assert regions[0].source_block_ids == ["p1-b1", "p1-b2"]
    assert not regions[0].page_wide


def test_dense_confident_form_does_not_create_a_page_wide_high_detail_crop() -> None:
    blocks = [
        Block(
            id=f"p1-b{index}",
            page=1,
            text=f"Field {index}: value",
            type="key_value",
            ocr_score=0.99,
            bbox=[0.05, index / 100, 0.45, index / 100 + 0.005],
        )
        for index in range(1, 81)
    ]

    assert plan_visual_review_regions(_page(*blocks)) == []


def test_invalid_table_structure_is_left_to_the_dedicated_table_review() -> None:
    block = _block("p1-b1", 0.99, [0.1, 0.1, 0.8, 0.2])
    page = _page(block)
    page.layout_regions = [
        LayoutRegion(
            id="p1-l1",
            page=1,
            cls_id=21,
            label="table",
            score=0.99,
            coordinate=[50, 100, 900, 400],
            bbox=[0.05, 0.1, 0.9, 0.4],
        )
    ]
    page.layout_block_links = []
    page.table_structures = [
        TableStructureEvidence(
            id="p1-l1-table",
            page=1,
            layout_region_id="p1-l1",
            bbox=[0.05, 0.1, 0.9, 0.4],
            style="wired",
            classifier_score=0.9,
            structure_score=0.9,
            structure_model="SLANeXt_wired",
            status="invalid",
            review_required=True,
        )
    ]

    assert plan_visual_review_regions(page) == []


def test_low_layout_confidence_alone_does_not_create_a_high_resolution_crop() -> None:
    page = _page(_block("p1-b1", 0.99, [0.1, 0.1, 0.8, 0.2]))
    page.layout_regions = [
        LayoutRegion(
            id="p1-l1",
            page=1,
            cls_id=1,
            label="text",
            score=0.1,
            coordinate=[50, 100, 900, 400],
            bbox=[0.05, 0.1, 0.9, 0.4],
        )
    ]

    assert plan_visual_review_regions(page) == []


def test_adjacent_uncertain_lines_do_not_merge_into_a_page_scale_crop() -> None:
    blocks = [
        _block(
            f"p1-b{index}",
            0.5,
            [0.05, 0.05 + index * 0.07, 0.95, 0.08 + index * 0.07],
        )
        for index in range(10)
    ]

    regions = plan_visual_review_regions(_page(*blocks), max_regions=12)

    assert len(regions) > 1
    assert (
        max(
            (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
            for region in regions
        )
        <= 0.15
    )


def test_region_limit_never_forces_unrelated_uncertainty_into_a_large_crop() -> None:
    blocks = [
        *[
            _block(f"p1-h{index}", 0.5, [0.05, 0.08 + index * 0.12, 0.95, 0.13 + index * 0.12])
            for index in range(7)
        ],
        *[
            _block(f"p1-v{index}", 0.5, [0.08 + index * 0.14, 0.05, 0.13 + index * 0.14, 0.95])
            for index in range(6)
        ],
    ]

    regions = plan_visual_review_regions(_page(*blocks), max_regions=12)

    assert (
        max(
            (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
            for region in regions
        )
        <= 0.15
    )
    assert {source for region in regions for source in region.source_block_ids} == {
        block.id for block in blocks
    }
    assert all(not region.page_wide for region in regions)


def test_single_oversized_local_uncertainty_is_tiled_not_sent_as_near_full_page() -> None:
    page = _page(_block("p1-b1", 0.5, [0.01, 0.01, 0.99, 0.7]))

    regions = plan_visual_review_regions(page)

    assert len(regions) > 1
    assert all(
        (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1]) <= 0.15
        for region in regions
    )
    assert all(region.source_block_ids == ["p1-b1"] for region in regions)
    assert all("oversized_region_tiled" in region.reason_codes for region in regions)
    assert all(not region.page_wide for region in regions)
