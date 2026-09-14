import hashlib
import io
import json
import zipfile

import pytest
from PIL import Image
from pypdf import PdfReader

import agentic_extractor.artifacts as artifacts_module
from agentic_extractor.artifacts import (
    build_local_artifacts,
    build_local_bundle,
    markdown_for_display,
)
from agentic_extractor.models import (
    Block,
    CheckboxRecord,
    CheckboxState,
    LayoutRegion,
    LocalCheckboxCandidate,
    ReadingOrderEvidence,
    RefinementRecord,
    TableCellEvidence,
    TableStructureEvidence,
    UsageRecord,
)
from agentic_extractor.ocr import EngineProvenance, LocalParseResult
from agentic_extractor.parse import PageParse
from agentic_extractor.pipeline import render_result_markdown
from agentic_extractor.table_structure import enrich_page_tables


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
        layout_image_bytes=_png(image),
        raw_evidence={"boxes": [block.polygon], "texts": [block.text], "scores": [0.8]},
        layout_signals={"orientation_degrees": None, "reading_order": "top-to-bottom"},
        local_checkbox_candidates=[
            LocalCheckboxCandidate(
                id="p2-cv1",
                page=2,
                state="UNCHECKED",
                control_bbox=[0.01, 0.1, 0.06, 0.18],
                detector_score=0.9,
                border_coverage=0.8,
                interior_ink_ratio=0.01,
                label_block_id="p2-b1",
                source_text="Total: 42",
                ocr_score=0.8,
                ocr_grounding_unique=False,
                engine_version="5.0.0",
            )
        ],
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


def _png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def test_artifacts_are_real_and_grounded() -> None:
    artifacts = build_local_artifacts(parse_result())
    assert artifacts.annotated_pdf.startswith(b"%PDF")
    html = artifacts.html.decode()
    assert "Total: 42" in html
    assert "data:image/png;base64," in html
    assert 'data-source-id="p2-b1"' in html
    assert 'data-layer="raw"' in html
    assert 'data-layer="refined"' in html
    assert 'data-block-type="text"' in html
    assert 'data-bbox="0.0833, 0.1250, 0.7500, 0.3750"' in html
    assert (
        'style="left:8.3333%;top:12.5000%;width:66.6667%;height:25.0000%;'
        'font-size:14.1667cqw;"' in html
    )
    assert 'id="zoom-in"' in html
    assert 'id="search"' in html
    assert "Grounding inspector" in html
    assert artifacts.markdown.startswith(b"Total: 42")
    parsed = json.loads(artifacts.parse_result)
    assert set(parsed) == {"markdown", "metadata", "structure"}
    assert artifacts.manifest["manifest_version"] == 7
    assert parsed["metadata"]["range_units"] == "unicode_codepoints"
    assert "gpt_attempts" in artifacts.manifest
    assert artifacts.manifest["pages"][0]["source_page"] == 2
    assert artifacts.manifest["pages"][0]["layout_regions"] == []
    assert "visual_routing" in artifacts.manifest
    page_node = parsed["structure"]["children"][0]
    assert page_node["grounding"]["page"] == 2
    box = page_node["children"][0]["atomic_grounding"][0]["box"]
    assert box["xmin"] == pytest.approx(1 / 12)
    assert box["ymin"] == pytest.approx(0.125)
    assert box["xmax"] == pytest.approx(0.75)
    assert box["ymax"] == pytest.approx(0.375)
    assert artifacts.manifest["selected_pages"] == [2]
    assert artifacts.manifest["html_layout"] == {
        "renderer_version": 1,
        "coordinate_space": "normalized_page_xyxy",
        "page_background": "ocr_aligned_lossless_png",
        "selected_pages": [2],
        "self_contained": True,
        "text_layers": ["refined", "raw"],
        "page_images": [
            {
                "source_page": 2,
                "mime_type": "image/png",
                "sha256": hashlib.sha256(_png(Image.new("RGB", (120, 80), "white"))).hexdigest(),
            }
        ],
    }
    assert "cloud_consent" not in artifacts.manifest["processing"]
    assert "gpt" not in artifacts.manifest["usage_and_cost"]
    assert "gpt_model" not in artifacts.manifest["processing"]
    assert "gpt_rate_assumptions" not in artifacts.manifest


def test_expensive_artifacts_are_generated_once_on_first_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def annotated_pdf(_result) -> bytes:
        calls.append("pdf")
        return b"%PDF-lazy"

    def coordinate_html(_result) -> str:
        calls.append("html")
        return "<html>lazy</html>"

    monkeypatch.setattr(artifacts_module, "_annotated_pdf", annotated_pdf)
    monkeypatch.setattr(artifacts_module, "build_coordinate_html", coordinate_html)
    monkeypatch.setattr(artifacts_module, "coordinate_html_page_images", lambda _result: [])

    artifacts = build_local_artifacts(parse_result())

    assert calls == []
    assert artifacts._result.timings["artifact_finalization_seconds"] >= 0
    assert artifacts.artifact_metadata("annotated.pdf")["generated"] is False
    assert artifacts.artifact_metadata("document.html")["generated"] is False
    assert artifacts.artifact_metadata("bundle.zip")["generated"] is False

    assert artifacts.annotated_pdf == b"%PDF-lazy"
    assert artifacts.annotated_pdf == b"%PDF-lazy"
    assert calls == ["pdf"]
    assert artifacts._result.timings["annotated_pdf_generation_seconds"] >= 0
    assert artifacts.html == b"<html>lazy</html>"
    assert artifacts.html == b"<html>lazy</html>"
    assert calls == ["pdf", "html"]
    assert artifacts._result.timings["html_generation_seconds"] >= 0

    first_bundle = artifacts.bundle
    assert first_bundle == artifacts.bundle
    assert calls == ["pdf", "html"]
    assert artifacts._result.timings["zip_packaging_seconds"] >= 0
    assert artifacts.artifact_metadata("bundle.zip")["generated"] is True
    assert artifacts.manifest["timing_analysis"]["bottleneck"] is not None


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
                "batch_kind": "compact",
                "batch_index": 1,
                "batch_count": 1,
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
    assert artifacts.manifest["usage_and_cost"]["gpt"]["calls"][0]["context"]["batch_kind"] == (
        "compact"
    )
    assert artifacts.manifest["refinement_layer"][0]["block_id"] == "p2-b1"
    html = artifacts.html.decode()
    assert "Total: 42.00" in html
    assert "Total: 42" in html
    assert artifacts.manifest["refinement_layer"][0]["proposed_text"] == "Total: 42.00"


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

    assert artifacts.manifest["checkboxes"][0]["id"] == "p2-c1"
    assert artifacts.manifest["checkboxes"][0]["state"] == "CHECKED"
    assert 'data-checkbox-id="p2-c1"' in artifacts.html.decode()
    assert "checkboxes/p2-c1.jpg" in artifacts.checkbox_crops
    with zipfile.ZipFile(io.BytesIO(build_local_bundle(result))) as archive:
        assert "checkboxes/p2-c1.jpg" in archive.namelist()


def test_review_required_checkboxes_never_enter_canonical_markdown_artifacts() -> None:
    result = parse_result()
    accepted = CheckboxRecord(
        id="accepted",
        page=2,
        label="Accepted option",
        state=CheckboxState.CHECKED,
        control_bbox=[0.1, 0.5, 0.15, 0.58],
        discovery_state=CheckboxState.CHECKED,
        decision_status="automated",
        agreement="consensus",
    )
    result.checkboxes = [
        accepted,
        CheckboxRecord(
            id="not-determinable",
            page=2,
            label="Human could not determine this option",
            state=CheckboxState.NOT_DETERMINABLE,
            control_bbox=[0.3, 0.5, 0.35, 0.58],
            discovery_state=CheckboxState.NOT_DETERMINABLE,
            decision_status="user_verified",
        ),
        *[
            CheckboxRecord(
                id=f"review-{index}",
                page=2,
                label=f"Review-only option {index}",
                state=CheckboxState.CHECKED,
                control_bbox=[0.2, 0.5, 0.25, 0.58],
                discovery_state=CheckboxState.CHECKED,
                decision_status="review_required",
            )
            for index in range(66)
        ],
    ]
    result.markdown = render_result_markdown(result)

    artifacts = build_local_artifacts(result)
    parse_markdown = json.loads(artifacts.parse_result)["markdown"]
    html = artifacts.html.decode()
    with zipfile.ZipFile(io.BytesIO(build_local_bundle(result))) as archive:
        bundled_markdown = archive.read("document.md").decode()
        bundled_html = archive.read("document.html").decode()

    markdown_outputs = (
        result.markdown,
        artifacts.markdown.decode(),
        parse_markdown,
        bundled_markdown,
    )
    for markdown in markdown_outputs:
        assert "[x] Accepted option" in markdown
        assert "Review-only option" not in markdown
        assert "Human could not determine this option" not in markdown
    for layout_html in (html, bundled_html):
        assert 'data-checkbox-id="accepted"' in layout_html
        assert 'data-checkbox-id="review-' not in layout_html
        assert 'data-checkbox-id="not-determinable"' not in layout_html

    # Uncertain candidates remain available as audit/review data, not document content.
    assert len(artifacts.manifest["checkboxes"]) == 68


def test_artifacts_include_selected_pages_only() -> None:
    artifacts = build_local_artifacts(parse_result())
    html = artifacts.html.decode()
    assert 'data-source-page="2"' in html
    assert 'data-source-page="1"' not in html
    assert len(PdfReader(io.BytesIO(artifacts.annotated_pdf)).pages) == 1


def test_html_never_positions_unsupported_or_unsafe_refinements() -> None:
    result = parse_result()
    result.pages[0].blocks[0].text = '</span><script id="injected">alert(1)</script>'
    result.refinements = [
        RefinementRecord(
            page=2,
            block_id=None,
            status="accepted",
            proposed_text="Unsupported page-level claim",
        ),
        RefinementRecord(
            page=2,
            block_id="p2-b1",
            status="rejected",
            proposed_text="Rejected claim",
        ),
    ]

    html = build_local_artifacts(result).html.decode()

    assert '<script id="injected">' not in html
    assert "&lt;/span&gt;&lt;script id=&quot;injected&quot;&gt;alert(1)&lt;/script&gt;" in html
    assert "Unsupported page-level claim" not in html
    assert "Rejected claim" not in html


def test_html_uses_detected_reading_order_and_auditable_geometry() -> None:
    result = parse_result()
    page = result.pages[0]
    page.blocks.append(
        Block(
            id="p2-b2",
            page=2,
            text="First in reading order",
            ocr_score=0.95,
            polygon=[[10, 40], [90, 40], [90, 60], [10, 60]],
            bbox=[10 / 120, 40 / 80, 90 / 120, 60 / 80],
        )
    )
    page.reading_order_evidence = ReadingOrderEvidence(
        page=2,
        ordered_block_ids=["p2-b2", "p2-b1"],
    )
    page.layout_regions = [
        LayoutRegion(
            id="p2-l1",
            page=2,
            cls_id=0,
            label="text",
            score=0.9,
            coordinate=[10, 10, 90, 60],
            bbox=[10 / 120, 10 / 80, 90 / 120, 60 / 80],
            polygon=[[10, 10], [90, 10], [90, 60], [10, 60]],
        )
    ]
    page.table_structures = [
        TableStructureEvidence(
            id="p2-t1",
            page=2,
            layout_region_id="p2-l1",
            bbox=[0.1, 0.1, 0.8, 0.7],
            style="wired",
            classifier_score=0.9,
            structure_score=0.9,
            structure_model="SLANeXt_wired",
            cells=[
                TableCellEvidence(
                    id="p2-t1-r1-c1",
                    row=1,
                    column=1,
                    bbox=[0.1, 0.1, 0.8, 0.3],
                    source_text="cell",
                )
            ],
            status="valid",
        )
    ]
    result.checkboxes = [
        CheckboxRecord(
            id="p2-check1",
            page=2,
            label="Approved",
            state=CheckboxState.CHECKED,
            control_bbox=[0.1, 0.7, 0.2, 0.8],
            discovery_state=CheckboxState.CHECKED,
            decision_status="automated",
        )
    ]
    result.refinements = [
        RefinementRecord(
            page=2,
            block_id="p2-b1",
            status="accepted",
            proposed_text="Total: 42.00",
        )
    ]

    html = build_local_artifacts(result).html.decode()

    refined_b2 = html.index('data-layer="refined" data-source-id="p2-b2"')
    refined_b1 = html.index('data-layer="refined" data-source-id="p2-b1"')
    assert refined_b2 < refined_b1
    assert 'data-provenance="gpt-5.6-luna + rapidocr"' in html
    assert 'data-layout-region-id="p2-l1"' in html
    assert 'data-table-cell-id="p2-t1-r1-c1"' in html
    assert 'data-checkbox-id="p2-check1"' in html


def test_artifacts_do_not_draw_rejected_table_cells() -> None:
    result = parse_result()
    result.pages[0].table_structures = [
        TableStructureEvidence(
            id="p2-t1",
            page=2,
            layout_region_id="p2-l1",
            bbox=[0.1, 0.1, 0.8, 0.7],
            style="wired",
            classifier_score=0.9,
            structure_score=0.9,
            structure_model="SLANeXt_wired",
            cells=[
                TableCellEvidence(
                    id="p2-t1-rejected-cell",
                    row=1,
                    column=1,
                    bbox=[0.1, 0.1, 0.8, 0.3],
                    source_text="rejected geometry",
                )
            ],
            status="invalid",
            review_required=True,
        )
    ]

    artifacts = build_local_artifacts(result)

    assert "p2-t1-rejected-cell" not in artifacts.html.decode()


def test_artifacts_do_not_draw_locally_valid_but_unresolved_table_cells() -> None:
    result = parse_result()
    result.pages[0].table_structures = [
        TableStructureEvidence(
            id="p2-t1",
            page=2,
            layout_region_id="p2-l1",
            bbox=[0.1, 0.1, 0.8, 0.7],
            style="wired",
            classifier_score=0.9,
            structure_score=0.9,
            structure_model="SLANeXt_wired",
            cells=[
                TableCellEvidence(
                    id="p2-t1-unresolved-cell",
                    row=1,
                    column=1,
                    bbox=[0.1, 0.1, 0.8, 0.3],
                    source_text="unresolved geometry",
                )
            ],
            status="valid",
        )
    ]
    result.document_metadata["table_reviews"] = [
        {"table_id": "p2-t1", "page": 2, "status": "unresolved", "reason": "abstained"}
    ]

    artifacts = build_local_artifacts(result)

    assert "p2-t1-unresolved-cell" not in artifacts.html.decode()


def test_accepted_table_is_synchronized_into_every_packaged_artifact() -> None:
    result = parse_result()
    page = result.pages[0]
    table = TableStructureEvidence(
        id="p2-t1",
        page=2,
        layout_region_id="p2-l1",
        bbox=[0.1, 0.1, 0.8, 0.4],
        style="wired",
        classifier_score=0.9,
        structure_score=0.9,
        structure_model="SLANeXt_wired",
        cells=[
            TableCellEvidence(
                id="p2-t1-r1-c1",
                row=1,
                column=1,
                bbox=[0.1, 0.1, 0.8, 0.4],
                source_block_ids=["p2-b1"],
                source_text="Total: 42",
            )
        ],
        markdown="<table><tr><td>Total: 42</td></tr></table>",
        status="valid",
        review_required=False,
    )
    enrich_page_tables(page, [table])
    result.document_metadata["table_reviews"] = [
        {"table_id": table.id, "page": 2, "status": "accepted", "outcome": "corrected"}
    ]
    result.markdown = "<!-- page: 2 -->\n\nTotal: 42"

    artifacts = build_local_artifacts(result)
    packaged = build_local_bundle(result)

    assert artifacts.markdown.decode().count("<table>") == 1
    assert (
        json.loads(artifacts.parse_result)["structure"]["children"][0]["children"][0]["type"]
        == "table"
    )
    assert artifacts.manifest["pages"][0]["table_structures"][0]["status"] == "valid"
    assert 'data-table-cell-id="p2-t1-r1-c1"' in artifacts.html.decode()
    with zipfile.ZipFile(io.BytesIO(packaged)) as archive:
        assert archive.read("document.md").decode().count("<table>") == 1
        assert b'"type": "table"' in archive.read("parse-result.json")
