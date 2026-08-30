import io
import json
import zipfile

from PIL import Image
from pypdf import PdfReader

from agentic_extractor.artifacts import (
    build_local_artifacts,
    build_local_bundle,
    markdown_for_display,
)
from agentic_extractor.models import (
    Block,
    CheckboxRecord,
    CheckboxState,
    RefinementRecord,
    UsageRecord,
)
from agentic_extractor.ocr import EngineProvenance, LocalParseResult
from agentic_extractor.parse import PageParse


def parse_result() -> LocalParseResult:
    image = Image.new("RGB", (120, 80), "white")
    block = Block(
        id="p2-b1",
        page=2,
        text="Total: 42",
        ocr_score=0.8,
        polygon=[[10, 10], [90, 10], [90, 30], [10, 30]],
        bbox=[10 / 120, 10 / 80, 90 / 120, 30 / 80],
    )
    page = PageParse(
        page=2,
        width=120,
        height=80,
        blocks=[block],
        image_bytes=_jpeg(image),
        raw_evidence={"boxes": [block.polygon], "texts": [block.text], "scores": [0.8]},
        layout_signals={"orientation_degrees": None, "reading_order": "top-to-bottom"},
    )
    return LocalParseResult(
        document_metadata={
            "file_name": "source.pdf",
            "mime_type": "application/pdf",
            "byte_size": 100,
        },
        selected_pages=[2],
        pages=[page],
        markdown="<!-- page: 2 -->\n\nTotal: 42",
        engine=EngineProvenance(name="RapidOCR", version="3.9.3.dev8", device="CPU"),
        timings={"total_seconds": 0.2},
        usage=UsageRecord(cache_write_input_tokens=7),
        workflow_manifest={"current_state": "ACCEPTED", "events": [{"action": "accept"}]},
    )


def _jpeg(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, "JPEG")
    return output.getvalue()


def test_artifacts_are_real_and_grounded() -> None:
    artifacts = build_local_artifacts(parse_result())
    assert artifacts.annotated_pdf.startswith(b"%PDF")
    html = artifacts.html.decode()
    assert "Total: 42" in html
    assert "<p>Total: 42</p>" in html
    assert "data:image" not in html
    assert 'data-grounding-id="p2-b1"' in html
    assert 'data-block-type="text"' in html
    assert 'data-bbox="0.0833, 0.1250, 0.7500, 0.3750"' in html
    assert "Grounding context" in html
    assert artifacts.markdown.startswith(b"<!-- page: 2 -->")
    parsed = json.loads(artifacts.parse_result)
    assert parsed["contract_version"] == 3
    assert artifacts.manifest["manifest_version"] == 3
    assert "ocr_attempts" in parsed
    assert "gpt_attempts" in artifacts.manifest
    assert parsed["pages"][0]["blocks"][0]["id"] == "p2-b1"
    assert parsed["pages"][0]["blocks"][0]["source_id"] == "p2-b1"
    assert parsed["pages"][0]["raw_evidence"]["texts"] == ["Total: 42"]
    assert parsed["chunks"][0]["source_id"] == "p2-c1"
    assert parsed["chunks"][0]["source_block_ids"] == ["p2-b1"]
    assert parsed["chunks"][0]["raw_scores"] == [0.8]
    assert parsed["coordinate_spaces"]["bbox"] == "normalized_page_xyxy"
    assert parsed["grounding"]["p2-b1"]["page"] == 2
    assert parsed["grounding"]["p2-b1"]["polygon"] == [
        [10, 10],
        [90, 10],
        [90, 30],
        [10, 30],
    ]
    assert artifacts.manifest["selected_pages"] == [2]
    assert "cloud_consent" not in artifacts.manifest["processing"]
    assert "gpt" not in artifacts.manifest["usage_and_cost"]
    assert "gpt_model" not in artifacts.manifest["processing"]
    assert "gpt_rate_assumptions" not in artifacts.manifest


def test_markdown_display_normalizes_html_tables_and_page_markers() -> None:
    raw = (
        "<!-- page: 2 -->\nBefore\n"
        '<table><tr><td colspan="2">Member information</td></tr>'
        "<tr><td>Name: A | B</td><td>Plan: H&amp;W</td></tr></table>\nAfter"
    )

    displayed = markdown_for_display(raw)

    assert "#### Page 2" in displayed
    assert "| Member information |  |" in displayed
    assert r"| Name: A \| B | Plan: H&W |" in displayed
    assert "<table" not in displayed
    assert (
        displayed.index("Before") < displayed.index("Member information") < displayed.index("After")
    )


def test_manifest_reports_gpt_only_after_a_real_cloud_call() -> None:
    result = parse_result()
    result.usage.call_count = 1
    result.usage.calls = [
        {
            "purpose": "refinement",
            "context": {
                "kind": "parse",
                "prompt_characters": 100,
                "evidence_characters": 40,
                "source_text_characters": 9,
                "block_count": 1,
                "compact_pages": [2],
                "full_context_pages": [],
            },
        }
    ]
    result.cloud_output = {"markdown": "refined"}
    result.refinements = [
        RefinementRecord(
            page=2,
            block_id="p2-b1",
            status="accepted",
            proposed_text="Total: 42.00",
        )
    ]
    artifacts = build_local_artifacts(result)

    assert artifacts.manifest["processing"]["gpt_model"] == "gpt-5.6-luna"
    assert artifacts.manifest["processing"]["reasoning_effort"] == "medium"
    assert artifacts.manifest["usage_and_cost"]["gpt"]["call_count"] == 1
    assert (
        artifacts.manifest["usage_and_cost"]["gpt"]["calls"][0]["context"]["evidence_characters"]
        == 40
    )
    assert artifacts.manifest["refinement_layer"][0]["block_id"] == "p2-b1"
    parsed = json.loads(artifacts.parse_result)
    assert parsed["refinement_layer"][0]["proposed_text"] == "Total: 42.00"


def test_bundle_contains_required_artifacts_and_manifest() -> None:
    result = parse_result()
    result.quality_diagnostics = [{"page": 2, "blur_warning": False}]
    result.adaptive_processing = {"initial_batch_size": 1, "memory_retries": 0}
    bundle = build_local_bundle(result)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert set(archive.namelist()) == {
            "document.md",
            "parse-result.json",
            "annotated.pdf",
            "document.html",
            "manifest.json",
        }
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["pages"][0]["source_page"] == 2
        assert manifest["artifacts"]["annotated_pdf"]["bytes"] > 0
        assert manifest["artifacts"]["parse_result"]["name"] == "parse-result.json"
        assert manifest["agent_workflow"]["events"][0]["action"] == "accept"
        assert manifest["quality_diagnostics"][0]["page"] == 2
        assert manifest["adaptive_processing"]["initial_batch_size"] == 1


def test_checkbox_evidence_is_in_all_artifacts_and_bundle_crop() -> None:
    result = parse_result()
    result.checkboxes = [
        CheckboxRecord(
            id="p2-c1",
            page=2,
            label="Approved",
            state=CheckboxState.CHECKED,
            control_bbox=[0.1, 0.1, 0.2, 0.25],
            label_block_id="p2-b1",
            source_text="Total: 42",
            confidence=0.99,
            discovery_state=CheckboxState.CHECKED,
            discovery_confidence=0.99,
            decision_status="automated",
            crop_ref="checkboxes/p2-c1.jpg",
        )
    ]
    artifacts = build_local_artifacts(result)

    parsed = json.loads(artifacts.parse_result)
    assert parsed["checkboxes"][0]["id"] == "p2-c1"
    assert artifacts.manifest["checkboxes"][0]["state"] == "CHECKED"
    assert 'data-checkbox-id="p2-c1"' in artifacts.html.decode()
    assert "checkboxes/p2-c1.jpg" in artifacts.checkbox_crops
    with zipfile.ZipFile(io.BytesIO(build_local_bundle(result))) as archive:
        assert "checkboxes/p2-c1.jpg" in archive.namelist()


def test_artifacts_include_selected_pages_only() -> None:
    artifacts = build_local_artifacts(parse_result())
    html = artifacts.html.decode()
    assert 'data-source-page="2"' in html
    assert 'data-source-page="1"' not in html
    assert len(PdfReader(io.BytesIO(artifacts.annotated_pdf)).pages) == 1
