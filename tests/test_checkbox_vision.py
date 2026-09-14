from __future__ import annotations

import io

from PIL import Image, ImageDraw

from agentic_extractor.checkbox_vision import (
    detect_page_checkboxes,
    is_automation_eligible_checkbox,
    is_credible_checkbox_candidate,
)
from agentic_extractor.models import Block, CheckboxState, LayoutRegion
from agentic_extractor.parse import PageParse


def _page(*, ambiguous_label: bool = False) -> PageParse:
    image = Image.new("RGB", (500, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 25, 55, 55), outline="black", width=3)
    draw.rectangle((25, 100, 55, 130), outline="black", width=3)
    draw.line((30, 115, 39, 125, 51, 105), fill="black", width=4)
    # A radio button must not become a square-control candidate.
    draw.ellipse((300, 25, 330, 55), outline="black", width=3)
    output = io.BytesIO()
    image.save(output, "PNG")
    blocks = [
        Block(
            id="p1-b1",
            page=1,
            text="Send email",
            ocr_score=0.97,
            bbox=[0.14, 0.13, 0.38, 0.32],
        ),
        Block(
            id="p1-b2",
            page=1,
            text="Send SMS",
            ocr_score=0.96,
            bbox=[0.14, 0.54, 0.38, 0.74],
        ),
    ]
    if ambiguous_label:
        blocks.append(
            Block(
                id="p1-b3",
                page=1,
                text="Send text message",
                ocr_score=0.99,
                bbox=[0.14, 0.55, 0.39, 0.73],
            )
        )
    return PageParse(page=1, width=500, height=180, image_bytes=output.getvalue(), blocks=blocks)


def test_detector_finds_square_controls_and_grounds_rapidocr_labels() -> None:
    candidates = detect_page_checkboxes(_page())

    assert len(candidates) == 2
    assert [item.id for item in candidates] == ["p1-cv1", "p1-cv2"]
    assert [item.state for item in candidates] == [
        CheckboxState.UNCHECKED,
        CheckboxState.CHECKED,
    ]
    assert [item.label_block_id for item in candidates] == ["p1-b1", "p1-b2"]
    assert all(item.ocr_grounding_unique for item in candidates)
    assert all(item.engine == "OpenCV" for item in candidates)


def test_detector_keeps_ambiguous_label_but_disallows_unique_grounding() -> None:
    candidates = detect_page_checkboxes(_page(ambiguous_label=True))

    assert len(candidates) == 2
    assert candidates[1].ocr_grounding_unique is False
    assert "ambiguous RapidOCR label" in candidates[1].risks


def test_detector_output_has_normalized_geometry_and_measurements() -> None:
    candidate = detect_page_checkboxes(_page())[0]

    assert all(0 <= value <= 1 for value in candidate.control_bbox)
    assert candidate.control_bbox[0] < candidate.control_bbox[2]
    assert candidate.control_bbox[1] < candidate.control_bbox[3]
    assert 0 <= candidate.detector_score <= 1
    assert 0 <= candidate.border_coverage <= 1
    assert 0 <= candidate.interior_ink_ratio <= 1


def test_detector_does_not_attach_a_distant_same_row_label() -> None:
    page = _page()
    page.blocks[0].bbox = [0.22, 0.13, 0.47, 0.32]

    candidates = detect_page_checkboxes(page)

    assert candidates[0].label_block_id is None
    assert candidates[0].ocr_grounding_unique is False


def test_detector_accepts_small_rapidocr_label_overlap_with_control() -> None:
    page = _page()
    page.blocks[0].bbox = [0.10, 0.13, 0.38, 0.32]

    candidate = detect_page_checkboxes(page)[0]

    assert candidate.label_block_id == "p1-b1"
    assert candidate.label_bbox == page.blocks[0].bbox
    assert candidate.ocr_grounding_unique is True


def test_detector_accepts_thin_real_world_checkbox_border() -> None:
    page = _page()
    image = Image.new("RGB", (500, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 25, 55, 55), outline="black", width=2)
    output = io.BytesIO()
    image.save(output, "PNG")
    page.image_bytes = output.getvalue()

    candidates = detect_page_checkboxes(page)

    assert len(candidates) == 1
    assert candidates[0].border_coverage >= 0.27
    assert is_credible_checkbox_candidate(page, candidates[0])


def test_detector_closes_tiny_fax_break_in_checkbox_border() -> None:
    page = _page()
    image = Image.new("RGB", (500, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 25, 55, 55), outline="black", width=2)
    draw.line((54, 39, 56, 39), fill="white", width=1)
    output = io.BytesIO()
    image.save(output, "PNG")
    page.image_bytes = output.getvalue()

    candidates = detect_page_checkboxes(page)

    assert len(candidates) == 1
    assert candidates[0].label_block_id == "p1-b1"
    assert is_credible_checkbox_candidate(page, candidates[0])


def test_detector_accepts_rapidocr_line_that_contains_control() -> None:
    page = _page()
    page.blocks[0].text = "Place of service: Hospital"
    page.blocks[0].bbox = [0.02, 0.13, 0.125, 0.32]

    candidate = detect_page_checkboxes(page)[0]

    assert candidate.label_block_id == "p1-b1"
    assert candidate.ocr_grounding_unique is True


def test_detector_skips_standalone_selection_mark_when_grounding_label() -> None:
    page = _page()
    page.blocks[0] = Block(
        id="p1-mark",
        page=1,
        text="X",
        ocr_score=0.96,
        bbox=[0.05, 0.13, 0.11, 0.32],
    )
    page.blocks.append(
        Block(
            id="p1-label",
            page=1,
            text="Nonparticipating",
            ocr_score=0.99,
            bbox=[0.11, 0.13, 0.38, 0.32],
        )
    )

    candidate = detect_page_checkboxes(page)[0]

    assert candidate.label_block_id == "p1-label"
    assert candidate.source_text == "Nonparticipating"
    assert candidate.ocr_grounding_unique is True


def test_detector_removes_selection_mark_fused_to_checked_label() -> None:
    page = _page()
    page.blocks[1].text = "XSend SMS"
    page.blocks[1].bbox = [0.083, 0.54, 0.38, 0.74]

    candidate = detect_page_checkboxes(page)[1]

    assert candidate.state is CheckboxState.CHECKED
    assert candidate.label_block_id == "p1-b2"
    assert candidate.source_text == "Send SMS"
    assert page.blocks[1].text == "XSend SMS"


def test_numeric_value_box_is_not_a_credible_checkbox() -> None:
    page = _page()
    page.blocks[0].text = "120.0"

    candidate = detect_page_checkboxes(page)[0]

    assert not is_credible_checkbox_candidate(page, candidate)


def test_detector_rejects_square_glyph_deep_inside_ocr_text() -> None:
    page = _page()
    page.blocks[0].bbox = [0.02, 0.13, 0.38, 0.32]

    candidate = detect_page_checkboxes(page)[0]

    assert candidate.label_block_id is None
    assert candidate.ocr_grounding_unique is False


def test_table_region_shape_is_not_eligible_checkbox_evidence() -> None:
    page = _page()
    page.layout_regions = [
        LayoutRegion(
            id="p1-l1",
            page=1,
            cls_id=21,
            label="table",
            score=0.99,
            coordinate=[0, 80, 300, 150],
            bbox=[0, 0.44, 0.6, 0.84],
        )
    ]

    candidates = detect_page_checkboxes(page)

    assert "inside a table region" not in candidates[0].risks
    assert "inside a table region" in candidates[1].risks
    assert is_automation_eligible_checkbox(page, candidates[0])
    assert not is_automation_eligible_checkbox(page, candidates[1])
    assert is_credible_checkbox_candidate(page, candidates[1])


def test_detector_rejects_text_glyph_sized_rectangles() -> None:
    image = Image.new("RGB", (500, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 25, 38, 41), outline="black", width=2)
    output = io.BytesIO()
    image.save(output, "PNG")
    page = PageParse(
        page=1,
        width=500,
        height=180,
        image_bytes=output.getvalue(),
        blocks=[
            Block(
                id="p1-b1",
                page=1,
                text="Header text",
                ocr_score=0.99,
                bbox=[0.09, 0.13, 0.3, 0.24],
            )
        ],
    )

    assert detect_page_checkboxes(page) == []


def test_detector_finds_checked_light_on_dark_control() -> None:
    image = Image.new("RGB", (500, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 20, 490, 70), fill="black")
    draw.rectangle((25, 30, 55, 60), fill="white")
    draw.line((31, 36, 49, 54), fill="black", width=3)
    draw.line((49, 36, 31, 54), fill="black", width=3)
    output = io.BytesIO()
    image.save(output, "PNG")
    page = PageParse(
        page=1,
        width=500,
        height=180,
        image_bytes=output.getvalue(),
        blocks=[
            Block(
                id="p1-b1",
                page=1,
                text="Nonparticipating",
                ocr_score=0.99,
                bbox=[0.12, 0.16, 0.38, 0.35],
            )
        ],
    )

    candidates = detect_page_checkboxes(page)

    assert len(candidates) == 1
    assert candidates[0].state is CheckboxState.CHECKED
    assert candidates[0].label_block_id == "p1-b1"
