import io

from PIL import Image, ImageDraw

from agentic_extractor.artifacts import build_local_artifacts
from agentic_extractor.models import (
    Block,
    DocumentRequest,
    LayoutRegion,
    UsageRecord,
)
from agentic_extractor.ocr import EngineProvenance, LocalParseResult
from agentic_extractor.openai_refiner import CloudEvidence, CloudRefinement, CloudResult
from agentic_extractor.parse import PageParse
from agentic_extractor.pipeline import refine_local_parse
from agentic_extractor.redaction_vision import detect_page_redactions


def _png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def _page() -> PageParse:
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((270, 100, 470, 130), fill="black")
    draw.rectangle((190, 150, 300, 180), fill="black")
    draw.rectangle((180, 240, 360, 265), fill="black")
    draw.rectangle((100, 30, 320, 55), fill="black")
    draw.rectangle((300, 400, 700, 440), fill="black")
    blocks = [
        Block(
            id="p1-b1",
            page=1,
            type="key_value",
            text="Patient Name:",
            bbox=[0.1, 0.071, 0.25, 0.093],
        ),
        Block(
            id="p1-b2",
            page=1,
            type="key_value",
            text="DOB:",
            bbox=[0.1, 0.107, 0.17, 0.129],
        ),
        Block(id="p1-b3", page=1, text="Session Report", bbox=[0.18, 0.143, 0.36, 0.164]),
        Block(id="p1-b4", page=1, text="REPORT-123", bbox=[0.18, 0.196, 0.36, 0.218]),
        Block(id="p1-b5", page=1, text="Pump Segment", bbox=[0.3, 0.286, 0.5, 0.314]),
    ]
    return PageParse(
        page=1,
        width=1000,
        height=1400,
        blocks=blocks,
        image_bytes=_png(image),
        layout_image_bytes=_png(image),
        raw_evidence={"texts": [block.text for block in blocks]},
        layout_regions=[
            LayoutRegion(
                id="p1-l1",
                page=1,
                cls_id=21,
                label="table",
                score=0.99,
                coordinate=[250, 380, 750, 500],
                bbox=[0.25, 0.271, 0.75, 0.357],
            )
        ],
    )


def test_detector_keeps_labeled_and_contextual_masks_but_rejects_dark_ui_regions() -> None:
    candidates = detect_page_redactions(_page())

    assert [candidate.context for candidate in candidates] == [
        "labeled_value",
        "labeled_value",
        "standalone_line",
    ]
    assert [candidate.label_block_id for candidate in candidates] == [
        "p1-b1",
        "p1-b2",
        None,
    ]
    assert all(candidate.bbox[1] > 0.06 for candidate in candidates)
    assert all(candidate.bbox[1] < 0.20 for candidate in candidates)


class _ConfirmingRefiner:
    def validate_configuration(self) -> None:
        return None

    def refine(self, pages, *args, **kwargs):
        refinements = [
            CloudRefinement(
                page=candidate.page,
                block_id=candidate.id,
                corrected_text="[REDACTED]",
                verified=True,
                evidence=[
                    CloudEvidence(
                        page=candidate.page,
                        bbox=candidate.bbox,
                        source="gpt-visual",
                    )
                ],
            )
            for page in pages
            for candidate in page.local_redaction_candidates
        ]
        return (
            CloudResult(refined_markdown="", reviewed_pages=[1], refinements=refinements),
            UsageRecord(call_count=1),
        )


def test_only_luna_confirmed_masks_become_grounded_canonical_blocks() -> None:
    page = _page()
    local = LocalParseResult(
        document_metadata={},
        selected_pages=[1],
        pages=[page],
        markdown="",
        engine=EngineProvenance("RapidOCR", "test", "CPU"),
        timings={"ocr_seconds": 0.1},
    )
    raw_evidence = dict(page.raw_evidence)

    result = refine_local_parse(
        local,
        DocumentRequest(file_name="redacted.png", file_bytes=b"source"),
        _ConfirmingRefiner(),
    )

    derived = [block for block in result.pages[0].blocks if block.source == "gpt-visual"]
    assert len(derived) == 3
    assert all(block.text == "[REDACTED]" and block.bbox for block in derived)
    assert result.markdown.count("[REDACTED]") == 3
    assert "Patient Name:\n[REDACTED]\nDOB:\n[REDACTED]" in result.markdown
    assert result.pages[0].raw_evidence == raw_evidence
    assert all(record.status == "accepted" for record in result.refinements)
    artifacts = build_local_artifacts(result)
    html = artifacts.html.decode("utf-8")
    assert 'data-layer="refined" data-source-id="p1-red1-block"' in html
    assert 'data-layer="raw" data-source-id="p1-red1-block"' not in html
    assert len(artifacts.manifest["pages"][0]["redaction_candidates"]) == 3


class _UngroundedRefiner(_ConfirmingRefiner):
    def refine(self, pages, *args, **kwargs):
        candidate = pages[0].local_redaction_candidates[0]
        return (
            CloudResult(
                refined_markdown="",
                reviewed_pages=[1],
                refinements=[
                    CloudRefinement(
                        page=1,
                        block_id=candidate.id,
                        corrected_text="[REDACTED]",
                        verified=True,
                        evidence=[
                            CloudEvidence(
                                page=1,
                                bbox=[0.8, 0.8, 0.9, 0.9],
                                source="gpt-visual",
                            )
                        ],
                    )
                ],
            ),
            UsageRecord(call_count=1),
        )


def test_unmatched_visual_evidence_cannot_publish_a_placeholder() -> None:
    page = _page()
    local = LocalParseResult(
        document_metadata={},
        selected_pages=[1],
        pages=[page],
        markdown="",
        engine=EngineProvenance("RapidOCR", "test", "CPU"),
        timings={},
    )

    result = refine_local_parse(
        local,
        DocumentRequest(file_name="redacted.png", file_bytes=b"source"),
        _UngroundedRefiner(),
    )

    assert "[REDACTED]" not in result.markdown
    assert result.refinements[0].status == "rejected"
