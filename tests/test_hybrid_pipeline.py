from typing import cast

from agentic_extractor.models import Block, Capability, DocumentRequest, ProcessingMode, UsageRecord
from agentic_extractor.ocr import EngineProvenance, LocalParseResult
from agentic_extractor.openai_refiner import CloudEvidence, CloudRefinement, CloudResult
from agentic_extractor.parse import PageParse
from agentic_extractor.pipeline import refine_local_parse
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


def test_balanced_sends_all_images_but_full_context_only_for_risk() -> None:
    refiner = FakeRefiner()
    result = refine_local_parse(local_result(0.5, 0.99), request(ProcessingMode.BALANCED), refiner)
    assert refiner.calls == 1
    assert refiner.pages == [1, 2]
    assert result.cloud_pages == [1, 2]
    assert result.cloud_image_pages == [1, 2]
    assert refiner.image_pages == {1}


def test_high_accuracy_sends_every_selected_page_image() -> None:
    refiner = FakeRefiner()
    result = refine_local_parse(
        local_result(0.99, 0.99), request(ProcessingMode.HIGH_ACCURACY), refiner
    )
    assert refiner.calls == 1
    assert result.cloud_pages == [1, 2]
    assert result.cloud_image_pages == [1, 2]


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
    assert "| Item | Qty |\n| --- | --- |\n| Widget | 2 |" in result.markdown
    assert [item.status for item in result.refinements] == ["accepted", "accepted"]
