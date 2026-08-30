from agentic_extractor.capabilities import validate_cloud_result
from agentic_extractor.models import Block
from agentic_extractor.openai_refiner import (
    CloudClassification,
    CloudEvidence,
    CloudResult,
    CloudSplit,
    ExtractedField,
)
from agentic_extractor.parse import PageParse


def page() -> PageParse:
    block = Block(
        id="p1-b1",
        page=1,
        text="Total 42",
        ocr_score=0.9,
        polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
        bbox=[0, 0, 1, 1],
    )
    return PageParse(page=1, width=1, height=1, blocks=[block])


def test_validates_labels_evidence_splits_and_schema() -> None:
    evidence = CloudEvidence(page=1, block_id="p1-b1", quote="Total", source="rapidocr")
    cloud = CloudResult(
        refined_markdown="Total 42",
        reviewed_pages=[1],
        classifications=[
            CloudClassification(label="invented", page_start=1, page_end=1, evidence=[evidence])
        ],
        splits=[CloudSplit(name="only", page_start=1, page_end=1)],
        extracted_fields=[ExtractedField(path="total", value=42, evidence=[evidence])],
    )
    classifications, _, splits, extraction, warnings = validate_cloud_result(
        cloud,
        [page()],
        ["invoice"],
        {"type": "object", "properties": {"total": {"type": "number"}}, "required": ["total"]},
    )
    assert classifications[0].label == "unknown"
    assert splits[0].name == "only"
    assert extraction and extraction.valid
    assert not warnings


def test_discards_invalid_evidence_and_split_coverage() -> None:
    bad = CloudEvidence(page=1, block_id="missing", bbox=[-1, 0, 1, 1])
    cloud = CloudResult(
        refined_markdown="x",
        reviewed_pages=[1],
        classifications=[
            CloudClassification(label="unknown", page_start=1, page_end=1, evidence=[bad])
        ],
        splits=[CloudSplit(name="gap", page_start=2, page_end=2)],
    )
    classifications, _, splits, _, warnings = validate_cloud_result(cloud, [page()], [], None)
    assert classifications[0].evidence == []
    assert splits == []
    assert warnings


def test_nested_field_json_string_is_decoded_before_schema_validation() -> None:
    evidence = CloudEvidence(page=1, block_id="p1-b1", quote="Total")
    cloud = CloudResult(
        refined_markdown="Total 42",
        reviewed_pages=[1],
        extracted_fields=[
            ExtractedField(path="items", value='[{"amount":42}]', evidence=[evidence])
        ],
    )
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"amount": {"type": "number"}},
                    "required": ["amount"],
                },
            }
        },
        "required": ["items"],
    }

    *_, extraction, warnings = validate_cloud_result(cloud, [page()], [], schema)

    assert extraction and extraction.valid
    assert extraction.values["items"] == [{"amount": 42}]
    assert not warnings
