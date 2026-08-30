from types import SimpleNamespace

import pytest
from PIL import Image

from agentic_extractor.ingest import IngestedDocument, InputPage
from agentic_extractor.models import Block, DocumentRequest, UsageRecord
from agentic_extractor.ocr import OCRResource
from agentic_extractor.openai_refiner import CloudResult
from agentic_extractor.parse import PageParse
from agentic_extractor.pipeline import GPTRefinementError, process_document


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


def test_pipeline_fails_when_gpt_refinement_fails(monkeypatch) -> None:
    setup_pipeline(monkeypatch)
    resource = OCRResource(SimpleNamespace(), "CPU", "fallback")
    with pytest.raises(GPTRefinementError, match="GPT-5.6-luna refinement failed"):
        process_document(
            DocumentRequest(file_name="x.png", file_bytes=b"png"),
            ocr_resource=resource,
            refiner=BadRefiner(),
        )
