import io
import json
import zipfile
from typing import cast

import pytest
from PIL import Image

from agentic_extractor.artifacts import build_local_artifacts, build_local_bundle
from agentic_extractor.models import (
    Block,
    Capability,
    DocumentRequest,
    LayoutBlockLink,
    LayoutRegion,
    ProcessingMode,
    TableCellEvidence,
    TableStructureEvidence,
    UsageRecord,
)
from agentic_extractor.ocr import EngineProvenance, LocalParseResult
from agentic_extractor.openai_refiner import (
    CloudEvidence,
    CloudRefinement,
    CloudResult,
    CloudSemanticRegion,
    CloudTableCell,
    CloudTableReview,
)
from agentic_extractor.parse import PageParse
from agentic_extractor.pipeline import _validated_semantic_chunks, refine_local_parse
from agentic_extractor.workflow import WorkflowState, evaluate_workflow


class FakeRefiner:
    def __init__(self) -> None:
        self.calls = 0

    def refine(self, *args, **kwargs):
        self.calls += 1
        self.pages = [page.page for page in args[0]]
        self.image_pages = args[1]
        evidence = CloudEvidence(page=1, bbox=[0, 0, 0.1, 0.1], quote="T0tal", source="gpt-visual")
        return (
            CloudResult(
                refined_markdown="ignored free-form markdown",
                reviewed_pages=self.pages,
                refinements=[
                    CloudRefinement(
                        page=1,
                        block_id="p1-b1",
                        corrected_text="Total",
                        block_type="key_value",
                        evidence=[evidence],
                        verified=True,
                    )
                ],
            ),
            UsageRecord(call_count=1, total_tokens=10),
        )

    def validate_configuration(self) -> None:
        return None


def local_result(score: float = 0.5, second_score: float | None = None) -> LocalParseResult:
    pages = []
    for page_number, page_score in enumerate(
        [score] if second_score is None else [score, second_score], 1
    ):
        block = Block(
            id=f"p{page_number}-b1",
            page=page_number,
            text="T0tal" if page_number == 1 else "Clean text",
            ocr_score=page_score,
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            bbox=[0, 0, 0.1, 0.1],
        )
        pages.append(
            PageParse(
                page=page_number,
                width=100,
                height=100,
                blocks=[block],
                image_bytes=b"jpeg",
                raw_evidence={"texts": [block.text], "scores": [page_score]},
            )
        )
    return LocalParseResult(
        document_metadata={},
        selected_pages=[page.page for page in pages],
        pages=pages,
        markdown="T0tal",
        engine=EngineProvenance("RapidOCR", "test", "CPU"),
        timings={"ocr_seconds": 1.0},
    )


def request(mode: ProcessingMode) -> DocumentRequest:
    return DocumentRequest(file_name="x.png", file_bytes=b"x", mode=mode)


def test_balanced_records_grounded_correction_without_mutating_raw_ocr() -> None:
    refiner = FakeRefiner()
    result = refine_local_parse(local_result(), request(ProcessingMode.BALANCED), refiner)
    assert refiner.calls == 1
    assert result.cloud_pages == [1]
    assert result.pages[0].blocks[0].text == "T0tal"
    assert result.pages[0].blocks[0].source == "rapidocr"
    assert result.pages[0].raw_evidence == {"texts": ["T0tal"], "scores": [0.5]}
    assert result.markdown == "<!-- page: 1 -->\n\nTotal"
    assert result.refinements[0].status == "accepted"
    assert result.refinements[0].block_id == "p1-b1"
    assert result.refinements[0].proposed_text == "Total"


def test_optional_workflow_receives_refined_markdown_after_parse() -> None:
    class MarkdownRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            self.parse_capabilities = args[2]
            return super().refine(*args, **kwargs)

        def refine_markdown(self, markdown, pages, capabilities, allowed_classes, schema):
            self.workflow_markdown = markdown
            self.workflow_capabilities = capabilities
            return (
                CloudResult(refined_markdown=markdown, reviewed_pages=[1]),
                UsageRecord(call_count=1, total_tokens=5),
            )

    refiner = MarkdownRefiner()
    extraction_request = DocumentRequest(
        file_name="x.png",
        file_bytes=b"x",
        capabilities={Capability.PARSE, Capability.CLASSIFY},
        allowed_classes=["invoice"],
    )

    result = refine_local_parse(local_result(), extraction_request, refiner)

    assert refiner.parse_capabilities == {"Parse"}
    assert refiner.workflow_capabilities == {"Classify"}
    assert refiner.workflow_markdown == "<!-- page: 1 -->\n\nTotal"
    assert result.usage.call_count == 2
    assert [attempt["purpose"] for attempt in result.cloud_attempts] == [
        "parse_refinement",
        "markdown_workflow",
    ]


def test_unsupported_refinement_is_rejected_and_cannot_change_markdown() -> None:
    class UnsupportedRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            self.calls += 1
            return (
                CloudResult(
                    refined_markdown="invented",
                    reviewed_pages=[1],
                    refinements=[
                        CloudRefinement(
                            page=1,
                            block_id="p1-b1",
                            corrected_text="Invented",
                            verified=True,
                        )
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(
        local_result(), request(ProcessingMode.BALANCED), UnsupportedRefiner()
    )

    assert result.pages[0].blocks[0].text == "T0tal"
    assert result.markdown == "<!-- page: 1 -->\n\nT0tal"
    assert result.refinements[0].status == "rejected"


def test_grounded_semantic_regions_drive_compact_canonical_markdown() -> None:
    local = local_result(0.99)
    local.pages[0].blocks = [
        Block(
            id="p1-b1",
            page=1,
            text="Claim review",
            type="paragraph",
            ocr_score=0.99,
            bbox=[0.1, 0.1, 0.8, 0.15],
        ),
        Block(
            id="p1-b2",
            page=1,
            text="First sentence",
            type="paragraph",
            ocr_score=0.99,
            bbox=[0.1, 0.2, 0.8, 0.25],
        ),
        Block(
            id="p1-b3",
            page=1,
            text="continues here.",
            type="paragraph",
            ocr_score=0.99,
            bbox=[0.1, 0.25, 0.8, 0.3],
        ),
    ]

    class SemanticRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            return (
                CloudResult(
                    refined_markdown="ignored free-form rewrite",
                    reviewed_pages=[1],
                    semantic_regions=[
                        CloudSemanticRegion(
                            id="p1-s1",
                            page=1,
                            type="heading",
                            reading_order=1,
                            source_block_ids=["p1-b1"],
                            heading_level=1,
                        ),
                        CloudSemanticRegion(
                            id="p1-s2",
                            page=1,
                            type="text",
                            reading_order=2,
                            source_block_ids=["p1-b2", "p1-b3"],
                            join_style="space",
                        ),
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(local, request(ProcessingMode.BALANCED), SemanticRefiner())

    assert result.markdown == (
        "<!-- page: 1 -->\n\n# Claim review\n\nFirst sentence continues here."
    )
    assert [chunk.type for chunk in result.pages[0].chunks] == ["heading", "text"]
    assert result.pages[0].chunks[1].source_block_ids == ["p1-b2", "p1-b3"]


def test_page_scoped_geometry_fallback_semantic_id_is_accepted() -> None:
    local = local_result(0.99)

    class FallbackSemanticRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            return (
                CloudResult(
                    refined_markdown="",
                    reviewed_pages=[1],
                    semantic_regions=[
                        CloudSemanticRegion(
                            id="p1-s1",
                            page=1,
                            type="text",
                            reading_order=1,
                            source_block_ids=["p1-b1"],
                            source_layout_region_ids=["p1-sr-fallback-1"],
                        )
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(local, request(ProcessingMode.BALANCED), FallbackSemanticRefiner())

    assert [chunk.id for chunk in result.pages[0].chunks] == ["p1-s1"]
    assert not any("semantic reconstruction was rejected" in warning for warning in result.warnings)


def test_incomplete_semantic_partition_is_completed_from_grounded_blocks() -> None:
    local = local_result(0.99)
    local.pages[0].blocks.append(
        Block(
            id="p1-b2",
            page=1,
            text="Second grounded line",
            ocr_score=0.99,
            bbox=[0.1, 0.2, 0.8, 0.25],
        )
    )

    class IncompleteSemanticRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            return (
                CloudResult(
                    refined_markdown="First line only",
                    reviewed_pages=[1],
                    semantic_regions=[
                        CloudSemanticRegion(
                            id="p1-s1",
                            page=1,
                            type="text",
                            reading_order=1,
                            source_block_ids=["p1-b1"],
                        )
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(
        local, request(ProcessingMode.BALANCED), IncompleteSemanticRefiner()
    )

    assert "T0tal" in result.markdown
    assert "Second grounded line" in result.markdown
    assert "First line only" not in result.markdown
    assert not any("semantic reconstruction was rejected" in warning for warning in result.warnings)
    assert {
        block_id for chunk in result.pages[0].chunks for block_id in chunk.source_block_ids
    } == {
        "p1-b1",
        "p1-b2",
    }


def test_duplicate_semantic_reading_orders_are_normalized_without_page_fallback() -> None:
    local = local_result(0.99)
    local.pages[0].blocks.append(
        Block(
            id="p1-b2",
            page=1,
            text="Second grounded line",
            ocr_score=0.99,
            bbox=[0.1, 0.2, 0.8, 0.25],
        )
    )

    class DuplicateOrderRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            return (
                CloudResult(
                    refined_markdown="",
                    reviewed_pages=[1],
                    semantic_regions=[
                        CloudSemanticRegion(
                            id="p1-s1",
                            page=1,
                            type="text",
                            reading_order=1,
                            source_block_ids=["p1-b1"],
                        ),
                        CloudSemanticRegion(
                            id="p1-s2",
                            page=1,
                            type="text",
                            reading_order=1,
                            source_block_ids=["p1-b2"],
                        ),
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(local, request(ProcessingMode.BALANCED), DuplicateOrderRefiner())

    assert [chunk.reading_order for chunk in result.pages[0].chunks] == [1, 2]
    assert not any("semantic reconstruction was rejected" in warning for warning in result.warnings)


def test_visually_grounded_logo_survives_pp_doclayout_text_label() -> None:
    local = local_result(0.99)
    page = local.pages[0]
    page.blocks[0].text = "Example Health"
    page.layout_regions = [
        LayoutRegion(
            id="p1-l1",
            page=1,
            cls_id=22,
            label="text",
            score=0.99,
            coordinate=[0, 0, 10, 10],
            bbox=[0, 0, 0.1, 0.1],
        )
    ]
    page.layout_block_links = [
        LayoutBlockLink(
            page=1,
            block_id="p1-b1",
            primary_region_id="p1-l1",
            region_ids=["p1-l1"],
            block_coverage=1,
        )
    ]

    class LogoRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            return (
                CloudResult(
                    refined_markdown="",
                    reviewed_pages=[1],
                    semantic_regions=[
                        CloudSemanticRegion(
                            id="p1-s1",
                            page=1,
                            type="logo",
                            reading_order=1,
                            source_block_ids=["p1-b1"],
                            source_layout_region_ids=["p1-l1"],
                            evidence=[
                                CloudEvidence(
                                    page=1,
                                    bbox=[0, 0, 0.1, 0.1],
                                    source="gpt-visual",
                                )
                            ],
                        )
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(local, request(ProcessingMode.BALANCED), LogoRefiner())

    assert [chunk.type for chunk in result.pages[0].chunks] == ["logo"]
    assert not any("semantic reconstruction was rejected" in warning for warning in result.warnings)


def test_small_header_wordmark_linked_to_pp_text_stays_text() -> None:
    local = local_result(0.99)
    page = local.pages[0]
    page.blocks[0].text = "Manufacturer"
    page.blocks[0].bbox = [0.78, 0.04, 0.92, 0.06]
    page.layout_regions = [
        LayoutRegion(
            id="p1-l1",
            page=1,
            cls_id=22,
            label="text",
            score=0.99,
            coordinate=[78, 4, 92, 6],
            bbox=[0.78, 0.04, 0.92, 0.06],
        )
    ]
    page.layout_block_links = [
        LayoutBlockLink(
            page=1,
            block_id="p1-b1",
            primary_region_id="p1-l1",
            region_ids=["p1-l1"],
            block_coverage=1,
        )
    ]

    class HeaderWordmarkRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            return (
                CloudResult(
                    refined_markdown="",
                    reviewed_pages=[1],
                    semantic_regions=[
                        CloudSemanticRegion(
                            id="p1-s1",
                            page=1,
                            type="logo",
                            reading_order=1,
                            source_block_ids=["p1-b1"],
                            source_layout_region_ids=["p1-l1"],
                            evidence=[
                                CloudEvidence(
                                    page=1,
                                    bbox=[0.78, 0.04, 0.92, 0.06],
                                    source="gpt-visual",
                                )
                            ],
                        )
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(local, request(ProcessingMode.BALANCED), HeaderWordmarkRefiner())

    assert [chunk.type for chunk in result.pages[0].chunks] == ["text"]


def test_graphical_logo_extent_survives_when_ocr_covers_only_its_wordmark() -> None:
    local = local_result(0.99)
    page = local.pages[0]
    page.blocks[0].text = "Amerigroup"
    page.blocks[0].bbox = [0.78, 0.04, 0.92, 0.06]

    class LogoRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            return (
                CloudResult(
                    refined_markdown="",
                    reviewed_pages=[1],
                    semantic_regions=[
                        CloudSemanticRegion(
                            id="p1-s1",
                            page=1,
                            type="logo",
                            reading_order=1,
                            source_block_ids=["p1-b1"],
                            evidence=[
                                CloudEvidence(
                                    page=1,
                                    bbox=[0.68, 0.02, 0.94, 0.10],
                                    source="gpt-visual",
                                )
                            ],
                        )
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(local, request(ProcessingMode.BALANCED), LogoRefiner())

    assert [chunk.type for chunk in result.pages[0].chunks] == ["logo"]


def test_grounded_attestation_language_survives_without_synthetic_marker() -> None:
    local = local_result(0.99)
    local.pages[0].blocks[0].text = "I attest that I reviewed and agree with this plan."

    class AttestationRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            return (
                CloudResult(
                    refined_markdown="",
                    reviewed_pages=[1],
                    semantic_regions=[
                        CloudSemanticRegion(
                            id="p1-s1",
                            page=1,
                            type="attestation",
                            reading_order=1,
                            source_block_ids=["p1-b1"],
                        )
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(local, request(ProcessingMode.BALANCED), AttestationRefiner())

    assert [chunk.type for chunk in result.pages[0].chunks] == ["attestation"]


@pytest.mark.parametrize("semantic_type", ["figure", "scan_code"])
def test_visual_only_semantic_types_require_grounded_pixels(semantic_type: str) -> None:
    local = local_result(0.99)
    page = local.pages[0]
    page.blocks[0].bbox = [0.1, 0.1, 0.4, 0.2]

    accepted = CloudSemanticRegion(
        id="p1-s1",
        page=1,
        type=semantic_type,
        reading_order=1,
        source_block_ids=["p1-b1"],
        evidence=[
            CloudEvidence(
                page=1,
                bbox=[0.05, 0.05, 0.5, 0.3],
                source="gpt-visual",
            )
        ],
    )
    rejected = accepted.model_copy(
        update={
            "id": "p1-s2",
            "evidence": [CloudEvidence(page=1, bbox=[0.7, 0.7, 0.8, 0.8], source="gpt-visual")],
        }
    )

    assert _validated_semantic_chunks(page, [accepted])[0].type == semantic_type
    assert _validated_semantic_chunks(page, [rejected])[0].type == "text"


def test_unsupported_special_type_downgrades_without_rejecting_page_partition() -> None:
    local = local_result(0.99)

    class UnsupportedAttestationRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            return (
                CloudResult(
                    refined_markdown="",
                    reviewed_pages=[1],
                    semantic_regions=[
                        CloudSemanticRegion(
                            id="p1-s1",
                            page=1,
                            type="attestation",
                            reading_order=1,
                            source_block_ids=["p1-b1"],
                        )
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(
        local, request(ProcessingMode.BALANCED), UnsupportedAttestationRefiner()
    )

    assert [chunk.type for chunk in result.pages[0].chunks] == ["text"]
    assert not any("semantic reconstruction was rejected" in warning for warning in result.warnings)


def test_independent_attestation_blocks_remain_independent_semantic_chunks() -> None:
    local = local_result(0.99)
    local.pages[0].blocks = [
        Block(
            id="p1-b1",
            page=1,
            text="[E-SIGNED]\nElectronically signed by: Clinician",
            ocr_score=0.99,
            bbox=[0.1, 0.1, 0.8, 0.15],
        ),
        Block(
            id="p1-b2",
            page=1,
            text="[SIGNED]\n[ILLEGIBLE_SIGNATURE]",
            ocr_score=0.99,
            bbox=[0.1, 0.16, 0.8, 0.25],
        ),
    ]

    class AttestationRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            return (
                CloudResult(
                    refined_markdown="",
                    reviewed_pages=[1],
                    semantic_regions=[
                        CloudSemanticRegion(
                            id="p1-s1",
                            page=1,
                            type="attestation",
                            reading_order=1,
                            source_block_ids=["p1-b1", "p1-b2"],
                        )
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(local, request(ProcessingMode.BALANCED), AttestationRefiner())

    assert [chunk.type for chunk in result.pages[0].chunks] == [
        "attestation",
        "attestation",
    ]
    assert [chunk.source_block_ids for chunk in result.pages[0].chunks] == [
        ["p1-b1"],
        ["p1-b2"],
    ]


def test_balanced_uses_compact_context_for_localized_risk() -> None:
    refiner = FakeRefiner()
    result = refine_local_parse(local_result(0.5, 0.99), request(ProcessingMode.BALANCED), refiner)
    assert refiner.calls == 1
    assert refiner.pages == [1, 2]
    assert result.cloud_pages == [1, 2]
    assert result.cloud_image_pages == [1, 2]
    assert refiner.image_pages == set()
    assert result.pages[0].visual_review_regions[0].reason_codes == ["low_ocr_confidence"]
    assert result.pages[1].visual_review_regions == []
    assert result.visual_routing["overview_pages"] == [1, 2]


def test_high_accuracy_keeps_full_context_but_not_confident_high_detail_regions() -> None:
    refiner = FakeRefiner()
    result = refine_local_parse(
        local_result(0.99, 0.99), request(ProcessingMode.HIGH_ACCURACY), refiner
    )
    assert refiner.calls == 1
    assert result.cloud_pages == [1, 2]
    assert result.cloud_image_pages == [1, 2]
    assert refiner.image_pages == {1, 2}
    assert all(not page.visual_review_regions for page in result.pages)


def test_high_accuracy_requires_a_grounded_outcome_for_each_low_confidence_block() -> None:
    class NoOutcomeRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            self.calls += 1
            return CloudResult(refined_markdown="", reviewed_pages=[1]), UsageRecord(call_count=1)

    extraction_request = request(ProcessingMode.HIGH_ACCURACY)
    result = refine_local_parse(local_result(0.849), extraction_request, NoOutcomeRefiner())
    workflow = evaluate_workflow(
        result, extraction_request, CloudResult.model_validate(result.cloud_output)
    )

    assert result.document_metadata["low_confidence_block_reviews"] == [
        {"block_id": "p1-b1", "page": 1, "ocr_score": 0.849, "status": "missing"}
    ]
    assert workflow.current_state is WorkflowState.REVIEW_REQUIRED
    assert any("p1-b1" in reason for reason in workflow.review_required)
    assert workflow.review_items[0].stage == "parse"
    assert workflow.review_items[0].retryable is False
    assert workflow.review_items[0].source_ids == ["p1-b1"]


def test_high_accuracy_grounded_correction_resolves_low_confidence_block() -> None:
    extraction_request = request(ProcessingMode.HIGH_ACCURACY)
    result = refine_local_parse(local_result(0.849), extraction_request, FakeRefiner())
    workflow = evaluate_workflow(
        result, extraction_request, CloudResult.model_validate(result.cloud_output)
    )

    block_reviews = cast(
        list[dict[str, object]], result.document_metadata["low_confidence_block_reviews"]
    )
    assert block_reviews[0]["status"] == "accepted"
    assert workflow.current_state is WorkflowState.ACCEPTED


def test_high_accuracy_abstention_preserves_block_and_requires_review() -> None:
    class AbstainingRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            return (
                CloudResult(
                    refined_markdown="",
                    reviewed_pages=[1],
                    refinements=[
                        CloudRefinement(
                            page=1,
                            block_id="p1-b1",
                            abstained=True,
                            warning="Image evidence is ambiguous.",
                        )
                    ],
                ),
                UsageRecord(call_count=1),
            )

    extraction_request = request(ProcessingMode.HIGH_ACCURACY)
    result = refine_local_parse(local_result(0.5), extraction_request, AbstainingRefiner())
    workflow = evaluate_workflow(
        result, extraction_request, CloudResult.model_validate(result.cloud_output)
    )

    assert result.pages[0].blocks[0].text == "T0tal"
    block_reviews = cast(
        list[dict[str, object]], result.document_metadata["low_confidence_block_reviews"]
    )
    assert block_reviews[0]["status"] == "abstained"
    assert workflow.current_state is WorkflowState.REVIEW_REQUIRED


def test_balanced_does_not_enforce_per_block_review_coverage() -> None:
    class NoOutcomeRefiner(FakeRefiner):
        def refine(self, *args, **kwargs):
            return CloudResult(refined_markdown="", reviewed_pages=[1]), UsageRecord(call_count=1)

    result = refine_local_parse(
        local_result(0.5), request(ProcessingMode.BALANCED), NoOutcomeRefiner()
    )

    assert "low_confidence_block_reviews" not in result.document_metadata


def test_grounded_layout_refinements_rebuild_markdown_from_copied_blocks() -> None:
    blocks = [
        Block(
            id=f"p1-b{index}",
            page=1,
            text=text,
            ocr_score=0.99,
            bbox=[0.1, 0.1 * index, 0.8, 0.1 * index + 0.05],
        )
        for index, text in enumerate(("Item | Qty", "Widget | 2"), 1)
    ]
    local = LocalParseResult(
        document_metadata={},
        selected_pages=[1],
        pages=[PageParse(page=1, width=100, height=100, blocks=blocks)],
        markdown="Item | Qty\n\nWidget | 2",
        engine=EngineProvenance("RapidOCR", "test", "CPU"),
        timings={"ocr_seconds": 1.0},
    )

    class TableRefiner:
        def validate_configuration(self) -> None:
            pass

        def refine(self, *args, **kwargs):
            refinements = [
                CloudRefinement(
                    page=1,
                    block_id=block.id,
                    block_type="table_row",
                    reading_order=index,
                    evidence=[
                        CloudEvidence(
                            page=1,
                            block_id=block.id,
                            quote=block.text,
                            source="rapidocr",
                        )
                    ],
                    verified=True,
                )
                for index, block in enumerate(blocks, 1)
            ]
            return (
                CloudResult(refined_markdown="", reviewed_pages=[1], refinements=refinements),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(local, request(ProcessingMode.BALANCED), TableRefiner())

    assert [block.type for block in result.pages[0].blocks] == ["text", "text"]
    assert (
        "<table><tr><th>Item</th><th>Qty</th></tr><tr><td>Widget</td><td>2</td></tr></table>"
    ) in result.markdown
    assert [item.status for item in result.refinements] == ["accepted", "accepted"]


def test_semantic_reconstruction_preserves_tables_sharing_source_blocks() -> None:
    block = Block(id="p1-b1", page=1, text="Shared", bbox=[0.1, 0.2, 0.8, 0.3])
    page = PageParse(page=1, width=100, height=100, blocks=[block])
    page.table_structures = [
        TableStructureEvidence(
            id=f"p1-table-{index}",
            page=1,
            layout_region_id=f"p1-l{index}",
            bbox=[0.1, 0.2, 0.8, 0.3],
            style="wired",
            classifier_score=0.9,
            structure_score=0.9,
            structure_model="SLANeXt_wired",
            cells=[
                TableCellEvidence(
                    id=f"p1-cell-{index}",
                    row=1,
                    column=1,
                    bbox=[0.1, 0.2, 0.8, 0.3],
                    source_block_ids=["p1-b1"],
                    source_text="Shared",
                )
            ],
            markdown=f"<table><tr><td>Shared {index}</td></tr></table>",
            status="valid",
            review_required=False,
        )
        for index in (1, 2)
    ]
    proposal = CloudSemanticRegion(
        id="p1-s1",
        page=1,
        type="table",
        reading_order=1,
        source_block_ids=["p1-b1"],
    )

    chunks = _validated_semantic_chunks(page, [proposal])

    assert [chunk.type for chunk in chunks] == ["table", "table"]


def test_accepted_table_repair_is_persisted_for_artifact_consumers() -> None:
    image = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(image, "JPEG")
    blocks = [
        Block(id="p1-b1", page=1, text="Item", bbox=[0.1, 0.2, 0.4, 0.3]),
        Block(id="p1-b2", page=1, text="Price", bbox=[0.5, 0.2, 0.8, 0.3]),
    ]
    table = TableStructureEvidence(
        id="p1-l1-table",
        page=1,
        layout_region_id="p1-l1",
        bbox=[0.05, 0.1, 0.9, 0.4],
        style="wired",
        classifier_score=0.9,
        structure_score=0.2,
        structure_model="SLANeXt_wired",
        status="invalid",
        review_required=True,
    )
    local = LocalParseResult(
        document_metadata={},
        selected_pages=[1],
        pages=[
            PageParse(
                page=1,
                width=100,
                height=100,
                blocks=blocks,
                table_structures=[table],
                image_bytes=image.getvalue(),
            )
        ],
        markdown="Item\n\nPrice",
        engine=EngineProvenance("RapidOCR", "test", "CPU"),
        timings={"ocr_seconds": 1.0},
    )

    class TableRepairRefiner:
        def validate_configuration(self) -> None:
            return None

        def refine(self, *args, **kwargs):
            review = CloudTableReview(
                page=1,
                table_id="p1-l1-table",
                outcome="corrected",
                visible_cell_count=2,
                cells=[
                    CloudTableCell(row=1, column=1, tag="th", source_block_ids=["p1-b1"]),
                    CloudTableCell(row=1, column=2, tag="th", source_block_ids=["p1-b2"]),
                ],
            )
            return (
                CloudResult(
                    refined_markdown="",
                    reviewed_pages=[1],
                    table_reviews=[review],
                    semantic_regions=[
                        CloudSemanticRegion(
                            id="p1-s1",
                            page=1,
                            type="table",
                            reading_order=1,
                            source_block_ids=["p1-b1"],
                        ),
                        CloudSemanticRegion(
                            id="p1-s2",
                            page=1,
                            type="table",
                            reading_order=2,
                            source_block_ids=["p1-b2"],
                        ),
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(local, request(ProcessingMode.BALANCED), TableRepairRefiner())

    assert result.pages[0].table_structures[0].status == "valid"
    assert result.pages[0].table_structures[0].review_required is False
    assert [chunk.type for chunk in result.pages[0].chunks] == ["table"]
    assert not any("semantic reconstruction was rejected" in warning for warning in result.warnings)
    assert [block.text for block in result.pages[0].blocks] == ["Item", "Price"]
    assert "<table><tr><th>Item</th><th>Price</th></tr></table>" in result.markdown
    table_reviews = cast(list[dict[str, object]], result.document_metadata["table_reviews"])
    assert table_reviews[0]["status"] == "accepted"

    artifacts = build_local_artifacts(result)
    repaired_table = artifacts.manifest["pages"][0]["table_structures"][0]
    parsed = json.loads(artifacts.parse_result)
    table_node = parsed["structure"]["children"][0]["children"][0]
    with zipfile.ZipFile(io.BytesIO(build_local_bundle(result))) as archive:
        bundled_markdown = archive.read("document.md").decode()
        bundled_manifest = json.loads(archive.read("manifest.json"))

    assert [cell["source_text"] for cell in repaired_table["cells"]] == ["Item", "Price"]
    assert table_node["type"] == "table"
    assert len(table_node["children"]) == 2
    assert "<table><tr><th>Item</th><th>Price</th></tr></table>" in bundled_markdown
    assert [
        cell["source_text"] for cell in bundled_manifest["pages"][0]["table_structures"][0]["cells"]
    ] == ["Item", "Price"]


def test_critical_token_correction_requires_visual_evidence() -> None:
    local = local_result(0.99)
    local.pages[0].blocks[0].text = "Diagnosis: 195.1"

    class TextOnlyCorrectionRefiner:
        def validate_configuration(self) -> None:
            return None

        def refine(self, *args, **kwargs):
            refinement = CloudRefinement(
                page=1,
                block_id="p1-b1",
                corrected_text="Diagnosis: I95.1",
                verified=True,
                evidence=[
                    CloudEvidence(
                        page=1,
                        block_id="p1-b1",
                        quote="Diagnosis: 195.1",
                        source="rapidocr",
                    )
                ],
            )
            return (
                CloudResult(refined_markdown="", reviewed_pages=[1], refinements=[refinement]),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(
        local, request(ProcessingMode.BALANCED), TextOnlyCorrectionRefiner()
    )

    assert result.refinements[0].status == "rejected"
    assert "Diagnosis: 195.1" in result.markdown
    assert "Diagnosis: I95.1" not in result.markdown


def test_visual_signature_replaces_ocr_noise_and_preserves_attestation_semantics() -> None:
    local = local_result(0.99)
    local.pages[0].blocks = [
        Block(
            id="p1-b1",
            page=1,
            text="Electronically signed by: Shaun C Jackson MD on 03/24/2024 03:26 PM",
            ocr_score=0.99174,
            bbox=[0.12, 0.21, 0.66, 0.24],
        ),
        Block(
            id="p1-b2",
            page=1,
            type="heading",
            text="Bhal m",
            ocr_score=0.60909,
            bbox=[0.13, 0.24, 0.50, 0.33],
        ),
    ]
    local.pages[0].raw_evidence = {"texts": ["Electronically signed by: ...", "Bhal m"]}
    local.pages[0].layout_regions = [
        LayoutRegion(
            id="p1-l1",
            page=1,
            cls_id=15,
            label="image",
            score=0.81,
            coordinate=[14.0, 24.0, 51.0, 33.0],
            bbox=[0.14, 0.24, 0.51, 0.33],
        )
    ]

    class SignatureRefiner:
        def validate_configuration(self) -> None:
            return None

        def refine(self, *args, **kwargs):
            return (
                CloudResult(
                    refined_markdown="",
                    reviewed_pages=[1],
                    refinements=[
                        CloudRefinement(
                            page=1,
                            block_id="p1-b2",
                            block_type="heading",
                            abstained=True,
                            warning=(
                                "The visual region shows a handwritten signature rather than "
                                "legible heading text."
                            ),
                            evidence=[
                                CloudEvidence(
                                    page=1,
                                    block_id="p1-b2",
                                    bbox=[0.13, 0.24, 0.50, 0.33],
                                    quote="visible handwritten signature ink",
                                    source="gpt-visual",
                                )
                            ],
                        ),
                        CloudRefinement(
                            page=1,
                            block_id="p1-b2",
                            corrected_text="Shaun C Jackson MD",
                            verified=True,
                            evidence=[
                                CloudEvidence(
                                    page=1,
                                    bbox=[0.13, 0.24, 0.50, 0.33],
                                    source="gpt-visual",
                                )
                            ],
                        ),
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(local, request(ProcessingMode.BALANCED), SignatureRefiner())

    assert "[E-SIGNED]\nElectronically signed by:" in result.markdown
    assert "[SIGNED]\n[ILLEGIBLE_SIGNATURE]" in result.markdown
    assert "Bhal m" not in result.markdown
    assert "## [SIGNED]" not in result.markdown
    assert [block.text for block in result.pages[0].blocks] == [
        "Electronically signed by: Shaun C Jackson MD on 03/24/2024 03:26 PM",
        "Bhal m",
    ]
    assert result.pages[0].raw_evidence == {"texts": ["Electronically signed by: ...", "Bhal m"]}
    assert [record.status for record in result.refinements] == [
        "abstained",
        "accepted",
        "accepted",
        "rejected",
    ]


@pytest.mark.parametrize(
    ("original", "corrected"),
    [
        (
            "195.1-458.0 Orthostatic hypotension, Active",
            "I95.1-458.0 Orthostatic hypotension, Active",
        ),
        (
            "173.9-443.9 Peripheral vascular disease, unspecified, Active",
            "I73.9-443.9 Peripheral vascular disease, unspecified, Active",
        ),
    ],
)
def test_visually_grounded_medical_identifier_correction_is_accepted(
    original: str, corrected: str
) -> None:
    local = local_result(0.99)
    local.pages[0].blocks[0].text = original

    class VisualCorrectionRefiner:
        def validate_configuration(self) -> None:
            return None

        def refine(self, *args, **kwargs):
            return (
                CloudResult(
                    refined_markdown="",
                    reviewed_pages=[1],
                    refinements=[
                        CloudRefinement(
                            page=1,
                            block_id="p1-b1",
                            corrected_text=corrected,
                            verified=True,
                            evidence=[
                                CloudEvidence(
                                    page=1,
                                    bbox=[0, 0, 0.1, 0.1],
                                    source="gpt-visual",
                                )
                            ],
                        )
                    ],
                ),
                UsageRecord(call_count=1),
            )

    result = refine_local_parse(local, request(ProcessingMode.BALANCED), VisualCorrectionRefiner())

    assert result.pages[0].blocks[0].text == original
    assert result.refinements[0].status == "accepted"
    assert corrected in result.markdown
    assert original not in result.markdown
