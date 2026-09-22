import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from agentic_extractor.ingest import IngestedDocument, InputPage
from agentic_extractor.models import Block, DocumentRequest, UsageRecord
from agentic_extractor.ocr import EngineProvenance, LocalParseResult, OCRResource
from agentic_extractor.openai_refiner import CloudResult
from agentic_extractor.parse import PageParse
from agentic_extractor.pipeline import GPTRefinementError, _process_local_document, process_document


def parsed_page() -> PageParse:
    return PageParse(
        page=1,
        width=10,
        height=10,
        image_bytes=b"jpeg",
        blocks=[
            Block(
                id="p1-b1",
                page=1,
                text="hello",
                ocr_score=0.99,
                polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
                bbox=[0, 0, 0.1, 0.1],
            )
        ],
    )


class GoodRefiner:
    def __init__(self) -> None:
        self.calls = 0

    def validate_configuration(self) -> None:
        return None

    def refine(self, *args, **kwargs):
        self.calls += 1
        self.full_context_pages = args[1]
        return CloudResult(refined_markdown="refined", reviewed_pages=[1]), UsageRecord(
            call_count=1, total_tokens=7
        )


class BadRefiner:
    def validate_configuration(self) -> None:
        return None

    def refine(self, *args, **kwargs):
        raise RuntimeError("offline")


def setup_pipeline(monkeypatch) -> None:
    document = IngestedDocument(
        "x.png", "image/png", b"png", [InputPage(1, Image.new("RGB", (10, 10)))]
    )
    monkeypatch.setattr("agentic_extractor.pipeline.load_document", lambda *args: document)
    monkeypatch.setattr("agentic_extractor.pipeline.ocr_page", lambda *args: parsed_page())


def test_pipeline_rejects_ungrounded_free_form_markdown(monkeypatch) -> None:
    setup_pipeline(monkeypatch)
    resource = OCRResource(SimpleNamespace(), "CPU")
    refiner = GoodRefiner()
    result = process_document(
        DocumentRequest(file_name="x.png", file_bytes=b"png"),
        ocr_resource=resource,
        refiner=refiner,
    )
    assert result.status == "complete"
    assert result.markdown == "<!-- page: 1 -->\n\nhello"
    assert result.usage.total_tokens == 7
    assert refiner.calls == 1
    assert refiner.full_context_pages == set()


def test_pipeline_fails_when_gpt_refinement_fails(monkeypatch) -> None:
    setup_pipeline(monkeypatch)
    resource = OCRResource(SimpleNamespace(), "CPU", "fallback")
    with pytest.raises(GPTRefinementError, match="gpt-6-sol refinement failed"):
        process_document(
            DocumentRequest(file_name="x.png", file_bytes=b"png"),
            ocr_resource=resource,
            refiner=BadRefiner(),
        )


def test_page_preparation_is_bounded_concurrent_and_source_ordered(monkeypatch) -> None:
    pages = [InputPage(number, Image.new("RGB", (10, 10))) for number in range(1, 5)]
    document = IngestedDocument("x.pdf", "application/pdf", b"pdf", pages)
    active = 0
    peak_active = 0
    lock = threading.Lock()

    class Diagnostic:
        def __init__(self, page: int) -> None:
            self.page = page
            self.preprocessing_actions: list[str] = []

        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"page": self.page, "preprocessing_actions": self.preprocessing_actions}

    def analyze(page: int, *_args, **_kwargs) -> Diagnostic:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return Diagnostic(page)

    def parse(document, selected, resource) -> LocalParseResult:
        assert selected == {1, 2, 3, 4}
        return LocalParseResult(
            document_metadata={},
            selected_pages=sorted(selected),
            pages=[PageParse(page=page.number, width=10, height=10) for page in document.pages],
            markdown="",
            engine=EngineProvenance("RapidOCR", "test", "CPU"),
            timings={},
        )

    monkeypatch.setenv("ADE_OCR_MAX_WORKERS", "3")
    monkeypatch.setattr("agentic_extractor.pipeline.load_document", lambda *_args: document)
    monkeypatch.setattr("agentic_extractor.pipeline.analyze_page", analyze)
    monkeypatch.setattr("agentic_extractor.pipeline.parse_document_local", parse)

    result = _process_local_document(
        DocumentRequest(file_name="x.pdf", file_bytes=b"pdf", selected_pages={1, 2, 3, 4}),
        ocr_resource=OCRResource(object(), "CPU"),
    )

    assert peak_active == 3
    assert [item["page"] for item in result.quality_diagnostics] == [1, 2, 3, 4]
    assert result.adaptive_processing["page_preparation_workers"] == 3
    assert result.timings["quality_seconds"] > 0
