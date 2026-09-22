import io
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from streamlit.testing.v1 import AppTest

from agentic_extractor.artifacts import build_local_artifacts
from agentic_extractor.models import (
    Block,
    CheckboxRecord,
    CheckboxState,
    ProcessingMode,
    UsageRecord,
)
from agentic_extractor.ocr import EngineProvenance, LocalParseResult
from agentic_extractor.parse import PageParse
from agentic_extractor.ui_state import AppState, output_file_name
from agentic_extractor.workflow import AgentWorkflowResult, ReviewItem, WorkflowState


def png_upload(file_name: str = "scan.png", color: str = "white") -> tuple[str, bytes, str]:
    output = io.BytesIO()
    Image.new("RGB", (20, 10), color).save(output, "PNG")
    return file_name, output.getvalue(), "image/png"


def test_ui_exposes_every_required_application_state() -> None:
    assert {state.value for state in AppState} == {
        "Idle",
        "Validating",
        "Ready",
        "Processing",
        "Completed",
        "Partial failure",
        "Failed",
    }


def test_output_file_names_preserve_original_file_stem() -> None:
    source = "Masked_Amerigroup_RealSolutions_1.pdf"

    assert output_file_name(source, ".md") == "Masked_Amerigroup_RealSolutions_1.md"
    assert output_file_name(source, ".parse.json") == (
        "Masked_Amerigroup_RealSolutions_1.parse.json"
    )
    assert output_file_name(source, ".annotated.pdf") == (
        "Masked_Amerigroup_RealSolutions_1.annotated.pdf"
    )
    assert output_file_name(source, ".html") == "Masked_Amerigroup_RealSolutions_1.html"
    assert output_file_name(source, ".zip") == "Masked_Amerigroup_RealSolutions_1.zip"

    app_source = (Path(__file__).parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    html_page_source = (Path(__file__).parents[1] / "app_pages" / "html.py").read_text(
        encoding="utf-8"
    )
    for suffix in (".md", ".parse.json", ".annotated.pdf", ".html", ".zip"):
        assert f'output_file_name(source_file_name, "{suffix}")' in app_source
    assert 'output_file_name(name, ".html")' in html_page_source


def test_processing_progress_bar_displays_numeric_percentages() -> None:
    source = (Path(__file__).parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    assert 'st.progress(0, text="0% — Preparing extraction")' in source
    assert (
        'progress.progress(35, text="35% — Running OCR, reading order, and table structure")'
        in source
    )
    assert 'cost_row.metric("PP-DocLayoutV3 API cost", "$0.00")' in source
    assert 'cost_row.metric("Table structure API cost", "$0.00")' in source
    assert 'progress.progress(100, text=f"100% — {final_state.value}")' in source


def test_annotated_pdf_uses_native_viewer_instead_of_blocked_data_url() -> None:
    source = (Path(__file__).parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    assert "st.pdf(st.session_state.artifacts.annotated_pdf" in source
    assert "data:application/pdf" not in source


def test_expensive_artifacts_are_requested_lazily() -> None:
    source = (Path(__file__).parents[1] / "streamlit_app.py").read_text(encoding="utf-8")

    assert 'on_change="rerun"' in source
    assert "if pdf_tab.open:" in source
    assert source.count("prepare_lazy_download(") == 3
    assert '"document.html"' in source
    assert '"bundle.zip"' in source
    assert "build_local_bundle(result)" not in source


def test_ui_has_no_cloud_consent_control() -> None:
    app = AppTest.from_file(
        Path(__file__).parents[1] / "streamlit_app.py", default_timeout=20
    ).run()
    assert not app.exception
    assert all("consent" not in checkbox.label.lower() for checkbox in app.checkbox)
    extract = next(button for button in app.button if button.label == "Extract document")
    assert extract.disabled is True


def test_ui_exposes_atomic_grounding_json_control_enabled_by_default() -> None:
    app = AppTest.from_file(
        Path(__file__).parents[1] / "streamlit_app.py", default_timeout=20
    ).run()

    control = next(checkbox for checkbox in app.checkbox if checkbox.label == "Atomic grounding")
    assert control.value is True
    source = (Path(__file__).parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Include atomic grounding parts array in JSON response." in source


def test_parse_is_always_enabled_and_other_workflows_are_optional() -> None:
    source = (Path(__file__).parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Parse · always enabled" in source
    assert '"Optional workflows"' in source
    assert "if item is not Capability.PARSE" in source
    assert "| {Capability.PARSE}" in source


def test_extract_schema_supports_json_and_markdown_file_uploads() -> None:
    source = (Path(__file__).parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    assert '"Upload JSON schema", type=["json"]' in source
    assert '"Upload Markdown schema", type=["md", "markdown"]' in source
    assert source.count('.decode("utf-8-sig")') == 2


def test_ui_reports_engine_configuration_without_exposing_key(
    monkeypatch,
) -> None:
    secret = "test-secret-must-not-render"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=20).run()
    app.switch_page("app_pages/diagnostics.py").run()
    assert not app.exception
    visible = " ".join(item.value for item in [*app.success, *app.error, *app.caption])
    assert "OpenAI API configured" in visible
    assert "RapidOCR installed" in visible
    assert secret not in visible


def test_processing_warnings_are_only_shown_once_on_diagnostics_page() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=20).run()
    warning = "Table p2-l7-table requires review."
    app.session_state["result"] = SimpleNamespace(
        warnings=[warning, warning, warning],
        failed_pages=[],
        engine=None,
        timings={},
    )

    app.switch_page("app_pages/diagnostics.py").run()

    assert [item.value for item in app.warning] == [f"{warning} (recorded 3 times)"]


def test_same_upload_preserves_completed_state_across_reruns() -> None:
    app = AppTest.from_file(
        Path(__file__).parents[1] / "streamlit_app.py", default_timeout=20
    ).run()
    app.file_uploader[0].set_value(png_upload()).run()
    assert app.session_state["app_state"] is AppState.READY

    app.session_state["app_state"] = AppState.COMPLETED
    app.run()

    assert app.session_state["app_state"] is AppState.COMPLETED


def test_new_upload_clears_stale_results_and_returns_to_ready() -> None:
    app = AppTest.from_file(
        Path(__file__).parents[1] / "streamlit_app.py", default_timeout=20
    ).run()
    app.file_uploader[0].set_value(png_upload()).run()
    app.session_state["result"] = "stale"
    app.session_state["workflow"] = "stale"

    app.file_uploader[0].set_value(png_upload("replacement.png", "black")).run()

    assert not app.exception
    assert app.session_state["result"] is None
    assert app.session_state["workflow"] is None
    assert app.session_state["app_state"] is AppState.READY


def test_usage_panel_shows_current_and_cumulative_session_totals() -> None:
    app = AppTest.from_file(
        Path(__file__).parents[1] / "streamlit_app.py", default_timeout=20
    ).run()
    current_usage = UsageRecord(
        call_count=2,
        input_tokens=200,
        cached_input_tokens=20,
        output_tokens=40,
        total_tokens=240,
        input_cost_usd=0.0000364,
        output_cost_usd=0.000048,
        total_cost_usd=0.0000844,
        cost_status="exact",
        calls=[
            {
                "purpose": "refinement",
                "pages": [1],
                "image_pages": [1],
                "context": {
                    "kind": "parse",
                    "prompt_characters": 120,
                    "evidence_characters": 50,
                    "block_count": 3,
                    "compact_pages": [1],
                    "full_context_pages": [],
                },
            }
        ],
    )
    prior_usage = UsageRecord(
        call_count=1,
        input_tokens=100,
        cached_input_tokens=10,
        output_tokens=20,
        total_tokens=120,
        input_cost_usd=0.0000182,
        output_cost_usd=0.000024,
        total_cost_usd=0.0000422,
        cost_status="exact",
    )
    app.session_state["result"] = SimpleNamespace(
        warnings=[],
        markdown="Grounded result",
        pages=[],
        engine=SimpleNamespace(name="RapidOCR", version="test", device="CPU", model_metadata={}),
        usage=current_usage,
        effective_mode=ProcessingMode.BALANCED,
        cloud_image_pages=[],
        timings={
            "ocr_seconds": 0.5,
            "ocr_wall_seconds": 0.5,
            "layout_detection_seconds": 0.2,
            "gpt_refinement_seconds": 1.5,
        },
        adaptive_processing={
            "actual_workers": 1,
            "ocr_cache_hits": 1,
            "ocr_cache_misses": 0,
            "ocr_cached_seconds_avoided": 0.25,
            "render_cache": {"hits": 1, "misses": 0},
        },
    )
    app.session_state["artifacts"] = SimpleNamespace(
        annotated_pdf=b"%PDF",
        html=b"<html></html>",
        manifest={},
        markdown=b"Grounded result",
        parse_result=b"{}",
    )
    app.session_state["usage_history"] = [
        {
            "mode": ProcessingMode.HIGH_ACCURACY.value,
            "usage": prior_usage.model_dump(mode="json"),
            "rapidocr_seconds": 0.25,
        },
        {
            "mode": ProcessingMode.BALANCED.value,
            "usage": current_usage.model_dump(mode="json"),
            "rapidocr_seconds": 0.5,
        },
    ]
    app.run()

    assert [tab.label for tab in app.tabs] == [
        "Source preview",
        "Rendered Markdown",
        "Raw Markdown",
        "Annotated PDF",
        "Grounded blocks",
        "Artifact metadata",
        "Usage",
    ]
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Session GPT calls"] == "3"
    assert metrics["Session total tokens"] == "360"
    assert metrics["Current GPT calls"] == "2"
    assert metrics["RapidOCR API cost"] == "$0.00"
    assert metrics["Slowest stage"] == "GPT parse refinement"
    assert metrics["Stage time"] == "1.500s"
    assert any("rendered pages: 1 hit / 0 miss" in caption.value for caption in app.caption)
    timing_table = next(frame.value for frame in app.dataframe if "share_percent" in frame.value)
    assert timing_table["stage"][0] == "GPT parse refinement"
    call_table = next(frame.value for frame in app.dataframe if "context_kind" in frame.value)
    assert call_table["context_kind"][0] == "parse"
    assert call_table["evidence_characters"][0] == 50


def test_visual_review_updates_exports_without_calling_an_engine(monkeypatch) -> None:
    from agentic_extractor.openai_refiner import CloudResult, OpenAIRefiner
    from agentic_extractor.pipeline import render_result_markdown
    from agentic_extractor.rich_document import VisualObject

    def forbidden(*args, **kwargs):
        raise AssertionError("UI review must not make a paid call")

    monkeypatch.setattr(OpenAIRefiner, "_request", forbidden)
    item = VisualObject(
        id="visual-1",
        page=1,
        kind="text",
        bbox=[0.1, 0.1, 0.8, 0.5],
        content="Total 42",
        reading_order=1,
    )
    local = LocalParseResult(
        document_metadata={"file_name": "synthetic.png"},
        selected_pages=[1],
        pages=[PageParse(page=1, width=20, height=10, image_bytes=png_upload()[1])],
        markdown="",
        engine=EngineProvenance("RapidOCR", "test", "CPU"),
        timings={},
        cloud_output=CloudResult(
            refined_markdown="", reviewed_pages=[1], visual_objects=[item]
        ).model_dump(mode="json"),
    )
    local.markdown = render_result_markdown(local)
    workflow = AgentWorkflowResult(
        current_state=WorkflowState.REVIEW_REQUIRED,
        events=[],
        review_required=["Visual object 'visual-1' requires review: crop not verified."],
    )
    app = AppTest.from_file(
        Path(__file__).parents[1] / "streamlit_app.py", default_timeout=20
    ).run()
    app.session_state["result"] = local
    app.session_state["workflow"] = workflow
    app.session_state["artifacts"] = build_local_artifacts(local)
    app.run()
    assert not app.exception
    next(widget for widget in app.selectbox if widget.label == "Decision").set_value("correct")
    next(widget for widget in app.text_area if widget.label == "Corrected content").set_value(
        "Total 43"
    )
    next(widget for widget in app.text_input if widget.label == "Review reason").set_value(
        "Read crop"
    )
    next(widget for widget in app.button if widget.label == "Record visual decision").click().run()
    assert not app.exception
    reviewed = app.session_state["result"]
    assert "Total 43" in reviewed.markdown
    assert reviewed.pages[0].blocks == []
    assert reviewed.pages[0].visual_audits[0]["actor"] == "user"
    assert b"Total 43" in app.session_state["artifacts"].markdown
    assert app.session_state["workflow"].current_state is WorkflowState.ACCEPTED


def test_checkbox_review_records_audited_user_decision() -> None:
    image = Image.new("RGB", (100, 100), "white")
    output = io.BytesIO()
    image.save(output, "JPEG")
    image_bytes = output.getvalue()
    block = Block(
        id="p1-b1",
        page=1,
        text="Approved",
        bbox=[0.25, 0.1, 0.8, 0.2],
        polygon=[[25, 10], [80, 10], [80, 20], [25, 20]],
    )
    checkbox = CheckboxRecord(
        id="p1-c1",
        page=1,
        label="Approved",
        state=CheckboxState.CHECKED,
        control_bbox=[0.1, 0.1, 0.2, 0.2],
        label_block_id="p1-b1",
        source_text="Approved",
        confidence=0.7,
        discovery_state=CheckboxState.CHECKED,
        discovery_confidence=0.7,
        decision_status="review_required",
        review_reason="Discovery confidence is below 0.98.",
        crop_ref="checkboxes/p1-c1.jpg",
    )
    local = LocalParseResult(
        document_metadata={"file_name": "form.jpg"},
        selected_pages=[1],
        pages=[
            PageParse(
                page=1,
                width=100,
                height=100,
                blocks=[block],
                image_bytes=image_bytes,
            )
        ],
        markdown="Approved",
        engine=EngineProvenance("RapidOCR", "test", "CPU"),
        timings={"ocr_seconds": 0.1},
        checkboxes=[checkbox],
    )
    workflow = AgentWorkflowResult(
        current_state=WorkflowState.REVIEW_REQUIRED,
        events=[],
        checkboxes=[checkbox],
        review_required=["Checkbox 'p1-c1' requires review."],
        review_items=[
            ReviewItem(
                id="review-1",
                stage="checkbox",
                code="CHECKBOX_REVIEW",
                message="Checkbox 'p1-c1' requires review.",
                source_ids=["p1-c1"],
            )
        ],
    )
    local.workflow_manifest = workflow.manifest()

    app = AppTest.from_file(
        Path(__file__).parents[1] / "streamlit_app.py", default_timeout=20
    ).run()
    app.session_state["result"] = local
    app.session_state["workflow"] = workflow
    app.session_state["artifacts"] = build_local_artifacts(local)
    app.run()

    next(item for item in app.selectbox if item.label == "Verified state").set_value(
        CheckboxState.UNCHECKED
    )
    next(item for item in app.text_input if item.label == "Review reason").set_value(
        "Confirmed from source crop"
    )
    next(item for item in app.button if item.label == "Record checkbox decision").click().run()

    reviewed = app.session_state["workflow"]
    assert reviewed.checkboxes[0].state is CheckboxState.UNCHECKED
    assert reviewed.checkboxes[0].decision_status == "user_verified"
    assert reviewed.checkbox_corrections[0].reason == "Confirmed from source crop"
