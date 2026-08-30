"""Streamlit entry point: controls left, source and artifacts center."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import streamlit as st

from agentic_extractor.artifacts import (
    build_local_artifacts,
    build_local_bundle,
    markdown_for_display,
)
from agentic_extractor.config import SETTINGS
from agentic_extractor.costs import aggregate_usage
from agentic_extractor.ingest import (
    IngestedDocument,
    IngestError,
    load_document,
    validate_page_range,
)
from agentic_extractor.models import (
    Capability,
    CheckboxState,
    DocumentRequest,
    ProcessingMode,
    UsageRecord,
)
from agentic_extractor.ocr import create_rapidocr_engine, rapidocr_package_version
from agentic_extractor.openai_refiner import (
    OpenAIConfigurationError,
    OpenAIRefiner,
)
from agentic_extractor.pipeline import render_result_markdown
from agentic_extractor.quality import analyze_page
from agentic_extractor.schema_input import (
    builder_to_schema,
    markdown_to_schema,
    parse_json_schema,
)
from agentic_extractor.ui_state import AppState
from agentic_extractor.workflow import (
    WorkflowState,
    record_checkbox_override,
    run_agent_workflow,
)


@st.cache_data(show_spinner=False, max_entries=4)
def inspect_upload(file_name: str, data: bytes) -> IngestedDocument:
    return load_document(file_name, data)


@st.cache_resource(show_spinner="Loading RapidOCR models…")
def get_ocr_resource():
    return create_rapidocr_engine()


@st.cache_resource(show_spinner=False)
def get_openai_refiner() -> OpenAIRefiner:
    return OpenAIRefiner()


def set_state(state: AppState) -> None:
    st.session_state.app_state = state


def reset() -> None:
    st.session_state.clear()


def _money(value: float | None, status: str) -> str:
    if value is None or status == "unavailable":
        return "Unavailable"
    suffix = " estimated" if status == "estimate" else ""
    return f"${value:.6f}{suffix}"


def _tokens(value: int | None) -> str:
    return "Unavailable" if value is None else f"{value:,}"


def _call_rows(calls: list[dict]) -> list[dict]:
    rows = []
    for number, call in enumerate(calls, 1):
        tokens = call.get("tokens", {})
        cost = call.get("cost", {})
        context = call.get("context", {})
        rows.append(
            {
                "call": number,
                "purpose": call.get("purpose"),
                "model": call.get("model"),
                "effort": call.get("reasoning_effort"),
                "pages": call.get("pages"),
                "image_pages": call.get("image_pages"),
                "context_kind": context.get("kind"),
                "prompt_characters": context.get("prompt_characters"),
                "evidence_characters": context.get("evidence_characters"),
                "evidence_blocks": context.get("block_count"),
                "compact_pages": context.get("compact_pages"),
                "full_context_pages": context.get("full_context_pages"),
                "input_tokens": tokens.get("input_tokens"),
                "cached_input_tokens": tokens.get("cached_input_tokens"),
                "output_tokens": tokens.get("output_tokens"),
                "total_tokens": tokens.get("total_tokens"),
                "input_cost_usd": cost.get("input_cost_usd"),
                "output_cost_usd": cost.get("output_cost_usd"),
                "total_cost_usd": cost.get("total_cost_usd"),
                "cost_status": cost.get("status"),
            }
        )
    return rows


def _page_usage_rows(calls: list[dict]) -> list[dict]:
    rows = []
    for call_number, call in enumerate(calls, 1):
        estimate = call.get("per_page_usage_estimate", {})
        image_pages = set(call.get("image_pages", []))
        for page in call.get("pages", []):
            rows.append(
                {
                    "call": call_number,
                    "page": page,
                    "image_sent": page in image_pages,
                    **estimate,
                    "cost_usd": call.get("per_page_cost_usd_estimate"),
                    "basis": "equal allocation estimate",
                }
            )
    return rows


st.session_state.setdefault("app_state", AppState.IDLE)
st.session_state.setdefault("result", None)
st.session_state.setdefault("original", None)
st.session_state.setdefault("usage_history", [])
st.session_state.setdefault("workflow", None)
st.session_state.setdefault("upload_identity", None)

st.title("Agentic document extraction")
st.caption("Local RapidOCR followed by required GPT-5.6-luna visual and semantic refinement")

document: IngestedDocument | None = None
validation_error: str | None = None
schema = None
schema_version = "1.0"
business_rules: list[dict[str, object]] = []

with st.sidebar:
    st.header("Document controls")
    uploaded = st.file_uploader(
        "Source document",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
        max_upload_size=50,
        help="PDF, PNG, JPEG, or TIFF; maximum 50 MiB.",
    )
    if uploaded is not None:
        upload_data = uploaded.getvalue()
        upload_identity = (uploaded.name, hashlib.sha256(upload_data).digest())
        upload_changed = upload_identity != st.session_state.upload_identity
        if upload_changed:
            st.session_state.upload_identity = upload_identity
            st.session_state["result"] = None
            st.session_state["workflow"] = None
            st.session_state["original"] = None
            st.session_state.pop("artifacts", None)
            set_state(AppState.VALIDATING)
        try:
            document = inspect_upload(uploaded.name, upload_data)
            if upload_changed:
                set_state(AppState.READY)
        except IngestError as exc:
            validation_error = str(exc)
            set_state(AppState.FAILED)
    elif st.session_state.upload_identity is not None:
        st.session_state.upload_identity = None
        st.session_state["result"] = None
        st.session_state["workflow"] = None
        st.session_state["original"] = None
        st.session_state.pop("artifacts", None)
        set_state(AppState.IDLE)

    mode_value = st.segmented_control(
        "Extraction mode",
        [mode.value for mode in ProcessingMode],
        default=ProcessingMode.BALANCED.value,
        key="mode",
    )
    mode = ProcessingMode(mode_value)
    st.subheader("Engine status")
    openai_key_present = SETTINGS.openai_configured
    if openai_key_present:
        st.success("OpenAI API configured · gpt-5.6-luna")
    else:
        st.error(
            "OpenAI is not configured. Add OPENAI_API_KEY to the launcher environment, "
            "restart the app, and retry."
        )
    rapidocr_version = rapidocr_package_version()
    if rapidocr_version:
        st.success(f"RapidOCR installed · {rapidocr_version}")
    else:
        st.error(
            "RapidOCR is unavailable. Run 'uv sync --all-groups' and verify ONNX Runtime setup."
        )
    enable_preprocessing = st.checkbox(
        "Apply only measurably improved local preprocessing",
        help=(
            "Currently retains same-geometry contrast normalization only when its "
            "diagnostic score improves."
        ),
    )

    st.caption(":material/check_circle: Parse · always enabled")
    optional_capabilities = [item for item in Capability if item is not Capability.PARSE]
    capability_values = st.pills(
        "Optional workflows",
        [item.value for item in optional_capabilities],
        selection_mode="multi",
        default=[],
    )
    capabilities = {Capability(value) for value in capability_values} | {Capability.PARSE}

    classes: list[str] = []
    if Capability.CLASSIFY in capabilities:
        class_text = st.text_input("Allowed classes", placeholder="invoice, receipt, contract")
        classes = [item.strip() for item in class_text.split(",") if item.strip()]

    split_text = ""
    split_override_reason = ""
    if Capability.SPLIT in capabilities:
        split_text = st.text_input(
            "Override split starts",
            placeholder="Example: 4, 9",
            help="Each number starts a new document and retains its original source page number.",
        )
        split_override_reason = st.text_input(
            "Split override reason",
            placeholder="Why these boundaries are authoritative",
            help="Required when override split starts are supplied; stored in the audit manifest.",
        )

    if Capability.EXTRACT in capabilities:
        schema_version = st.text_input("Schema version", value="1.0", max_chars=64)
        schema_mode = st.segmented_control(
            "Field schema", ["Guided", "JSON Schema", "Markdown"], default="Guided"
        )
        if schema_mode == "Guided":
            rows = st.data_editor(
                [
                    {
                        "name": "",
                        "type": "string",
                        "description": "",
                        "required": False,
                        "format": "",
                        "pattern": "",
                        "minimum": None,
                        "maximum": None,
                    }
                ],
                num_rows="dynamic",
                width="stretch",
                column_config={
                    "type": st.column_config.SelectboxColumn(
                        options=["string", "number", "integer", "boolean"]
                    )
                },
            )
            try:
                schema = (
                    builder_to_schema(rows)
                    if any(str(row.get("name", "")).strip() for row in rows)
                    else None
                )
            except ValueError as exc:
                schema = None
                st.error(f"Invalid guided schema: {exc}")
        elif schema_mode == "JSON Schema":
            schema_source = st.segmented_control(
                "JSON schema input", ["Paste", "Upload"], default="Paste"
            )
            raw_schema = ""
            if schema_source == "Upload":
                schema_file = st.file_uploader(
                    "Upload JSON schema", type=["json"], key="json_schema_file"
                )
                if schema_file is not None:
                    try:
                        raw_schema = schema_file.getvalue().decode("utf-8-sig")
                    except UnicodeDecodeError:
                        st.error("Invalid JSON Schema: the uploaded file must be UTF-8 text.")
            else:
                raw_schema = st.text_area(
                    "JSON Schema", height=180, placeholder='{"type":"object","properties":{}}'
                )
            try:
                schema = parse_json_schema(raw_schema) if raw_schema.strip() else None
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(f"Invalid JSON Schema: {exc}")
        else:
            schema_source = st.segmented_control(
                "Markdown schema input", ["Paste", "Upload"], default="Paste"
            )
            markdown_schema = ""
            if schema_source == "Upload":
                schema_file = st.file_uploader(
                    "Upload Markdown schema", type=["md", "markdown"], key="markdown_schema_file"
                )
                if schema_file is not None:
                    try:
                        markdown_schema = schema_file.getvalue().decode("utf-8-sig")
                    except UnicodeDecodeError:
                        st.error("Invalid Markdown schema: the uploaded file must be UTF-8 text.")
            else:
                markdown_schema = st.text_area(
                    "Markdown field definitions",
                    height=220,
                    placeholder=(
                        "## invoice_number\nUnique invoice identifier.\n\n"
                        "## invoice_date\nInvoice issue date in YYYY-MM-DD format.\n\n"
                        "## total_amount\nFinal payable amount, including tax."
                    ),
                )
            try:
                schema = markdown_to_schema(markdown_schema) if markdown_schema.strip() else None
            except ValueError as exc:
                st.error(f"Invalid Markdown schema: {exc}")
        rules_text = st.text_area(
            "Cross-field rules (JSON)",
            value="[]",
            help='Supported operations: "equals", "sum_equals", and "less_than_or_equal".',
        )
        try:
            parsed_rules = json.loads(rules_text)
            if not isinstance(parsed_rules, list):
                raise ValueError("Rules must be a JSON array.")
            business_rules = parsed_rules
        except (ValueError, json.JSONDecodeError) as exc:
            st.error(f"Invalid business rules: {exc}")
            schema = None
        if schema:
            schema["x-schema-version"] = schema_version

    start_page = end_page = 1
    if document and document.page_count > 1:
        st.subheader("Inclusive page range")
        start_page = st.number_input("Start page", 1, document.page_count, 1)
        end_page = st.number_input("End page", start_page, document.page_count, document.page_count)
    elif document:
        st.caption("Single-page source · page range fixed to 1")

    split_boundaries: list[int] = []
    try:
        split_boundaries = [int(item.strip()) for item in split_text.split(",") if item.strip()]
    except ValueError:
        st.error("Split starts must be comma-separated page numbers.")
    can_process = bool(
        document
        and not validation_error
        and openai_key_present
        and rapidocr_version
        and (Capability.CLASSIFY not in capabilities or classes)
        and (Capability.EXTRACT not in capabilities or schema)
        and (not split_boundaries or split_override_reason.strip())
    )
    process_clicked = st.button(
        "Extract document",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
        disabled=not can_process,
    )
    st.button("Reset", icon=":material/restart_alt:", width="stretch", on_click=reset)

state = st.session_state.app_state
state_color: Literal["green", "orange", "red", "blue", "gray"]
if state is AppState.COMPLETED:
    state_color = "green"
elif state is AppState.PARTIAL_FAILURE:
    state_color = "orange"
elif state is AppState.FAILED:
    state_color = "red"
elif state is AppState.PROCESSING:
    state_color = "blue"
else:
    state_color = "gray"
st.badge(state.value, color=state_color, icon=":material/radio_button_checked:")

if validation_error:
    st.error(validation_error, icon=":material/error:")

if document is None:
    with st.container(border=True):
        st.subheader(":material/preview: Source preview")
        st.caption("Upload a readable document from the control pane to begin.")
else:
    metadata = st.container(horizontal=True, border=True)
    metadata.metric("Filename", document.file_name)
    metadata.metric("Type", document.mime_type)
    metadata.metric("Size", f"{document.byte_size / 1024:,.1f} KiB")
    metadata.metric("Pages", document.page_count)
    with st.container(border=True):
        st.subheader(":material/preview: Source preview")
        st.image(
            document.pages[int(start_page) - 1].image,
            caption=f"Page {int(start_page)} of {document.page_count}",
            width="stretch",
        )
    with st.expander(":material/image_search: Image-quality preflight"):
        for page in document.pages[int(start_page) - 1 : int(end_page)]:
            diagnostic = analyze_page(
                page.number,
                page.image,
                dpi=page.source_dpi,
                source_format=page.source_format,
                orientation_correction_degrees=page.orientation_correction_degrees,
            )
            st.write(f"Page {page.number}")
            st.json(diagnostic.model_dump(mode="json"))
        st.caption(
            "No RapidOCR segmentation API is claimed. This build uses grounded layout "
            "reconstruction and one-page bounded OCR calls. Incorrect regions usually indicate "
            "layout/detection issues; correct regions with incorrect text usually indicate "
            "OCR or source-image quality issues."
        )

if process_clicked and document:
    progress = st.progress(0, text="0% — Preparing extraction")
    try:
        progress.progress(5, text="5% — Validating selected pages")
        selected = set(validate_page_range(int(start_page), int(end_page), document.page_count))
        set_state(AppState.PROCESSING)
        with st.status("Processing selected pages…", expanded=True) as status:
            progress.progress(15, text="15% — Validating OpenAI configuration")
            st.write("Validating OpenAI configuration without sending document content")
            cloud_refiner = get_openai_refiner()
            cloud_refiner.validate_configuration()
            progress.progress(25, text="25% — Preparing the extraction request")
            request = DocumentRequest(
                file_name=document.file_name,
                file_bytes=document.original_bytes,
                mode=mode,
                selected_pages=selected,
                capabilities=capabilities,
                allowed_classes=classes,
                extraction_schema=schema,
                split_boundaries=split_boundaries,
                split_override_reason=split_override_reason,
                business_rules=business_rules,
                enable_preprocessing=enable_preprocessing,
            )
            st.write("Running RapidOCR first, followed by required GPT-5.6-luna refinement")
            progress.progress(35, text="35% — Running RapidOCR and GPT refinement")
            result, workflow = run_agent_workflow(
                request,
                ocr_resource=get_ocr_resource(),
                refiner=cloud_refiner,
            )
            progress.progress(90, text="90% — Building local artifacts")
            artifacts = build_local_artifacts(result)
            final_state = (
                AppState.PARTIAL_FAILURE
                if result.failed_pages
                or workflow.current_state is WorkflowState.REVIEW_REQUIRED
                or workflow.current_state is WorkflowState.FAILED
                else AppState.COMPLETED
            )
            set_state(final_state)
            status.update(
                label=final_state.value,
                state=(
                    "error"
                    if result.failed_pages or workflow.current_state is WorkflowState.FAILED
                    else "complete"
                ),
            )
            progress.progress(100, text=f"100% — {final_state.value}")
        st.session_state.result = result
        st.session_state.workflow = workflow
        st.session_state.artifacts = artifacts
        st.session_state.usage_history.append(
            {
                "mode": result.effective_mode.value,
                "usage": result.usage.model_dump(mode="json"),
                "rapidocr_seconds": result.timings.get("ocr_seconds"),
            }
        )
        st.rerun()
    except OpenAIConfigurationError as exc:
        set_state(AppState.FAILED)
        progress.progress(100, text="100% — Failed")
        st.error(str(exc))
    except Exception as exc:
        set_state(AppState.FAILED)
        progress.progress(100, text="100% — Failed")
        st.error(f"Processing failed: {type(exc).__name__}: {exc}")

if result := st.session_state.result:
    st.subheader(":material/article: Results")
    for warning in result.warnings:
        st.warning(warning)
    rendered_tab, raw_tab, pdf_tab, html_tab, blocks_tab, artifacts_tab, usage_tab = st.tabs(
        [
            "Rendered Markdown",
            "Raw Markdown",
            "Annotated PDF",
            "HTML",
            "Grounded blocks",
            "Artifact metadata",
            "Usage",
        ]
    )
    with rendered_tab:
        st.markdown(markdown_for_display(result.markdown) or "_No text was extracted._")
    with raw_tab:
        st.code(result.markdown, language="markdown", wrap_lines=True)
    with pdf_tab:
        st.pdf(st.session_state.artifacts.annotated_pdf, height=720)
    with html_tab:
        st.caption("Refined Markdown rendered as HTML with page and grounding context.")
        st.iframe(st.session_state.artifacts.html.decode("utf-8"), height=760)
    with blocks_tab:
        blocks = [block.model_dump() for page in result.pages for block in page.blocks]
        st.dataframe(blocks, width="stretch")
    with artifacts_tab:
        st.json(st.session_state.artifacts.manifest)
    with usage_tab:
        st.json(
            {
                "timings": result.timings,
                "engine": {
                    "name": result.engine.name,
                    "version": result.engine.version,
                    "device": result.engine.device,
                    "model_metadata": result.engine.model_metadata,
                },
                "gpt": result.usage.model_dump(mode="json"),
            }
        )
    downloads = st.container(horizontal=True)
    downloads.download_button("Markdown", st.session_state.artifacts.markdown, "document.md")
    downloads.download_button(
        "Structured Parse", st.session_state.artifacts.parse_result, "parse-result.json"
    )
    downloads.download_button(
        "Annotated PDF", st.session_state.artifacts.annotated_pdf, "annotated.pdf"
    )

if workflow := st.session_state.workflow:
    st.subheader(":material/account_tree: Agent Workflow")
    st.badge(
        workflow.current_state.value,
        color=(
            "green"
            if workflow.current_state is WorkflowState.ACCEPTED
            else "red"
            if workflow.current_state is WorkflowState.FAILED
            else "orange"
        ),
    )
    st.dataframe([event.model_dump(mode="json") for event in workflow.events], width="stretch")
    if workflow.review_required:
        with st.expander("Review-required items", expanded=True):
            st.dataframe(
                [item.model_dump(mode="json") for item in workflow.review_items],
                hide_index=True,
                width="stretch",
            )
    if workflow.checkboxes:
        st.subheader(":material/check_box: Checkbox results")
        st.caption(
            "Automated checkbox decisions are best-effort. Risky controls require source review."
        )
        st.dataframe(
            [item.model_dump(mode="json") for item in workflow.checkboxes],
            hide_index=True,
            width="stretch",
        )
    checkbox_reviews = [
        checkbox
        for checkbox in workflow.checkboxes
        if checkbox.decision_status == "review_required"
    ]
    if checkbox_reviews:
        with st.expander(":material/fact_check: Checkbox review", expanded=True):
            selected_checkbox_id = st.selectbox(
                "Checkbox", [item.id for item in checkbox_reviews], key="checkbox_review_id"
            )
            selected_checkbox = next(
                item for item in checkbox_reviews if item.id == selected_checkbox_id
            )
            crop = st.session_state.artifacts.checkbox_crops.get(selected_checkbox.crop_ref or "")
            page = next(
                (item for item in result.pages if item.page == selected_checkbox.page), None
            )
            preview_columns = st.columns(2)
            if crop:
                preview_columns[0].image(
                    crop, caption=f"Checkbox crop · {selected_checkbox.id}", width="stretch"
                )
            if page:
                preview_columns[1].image(
                    page.original_image_bytes or page.image_bytes,
                    caption=f"Source page {page.page}",
                    width="stretch",
                )
            st.json(
                {
                    "label": selected_checkbox.label,
                    "discovery": {
                        "state": selected_checkbox.discovery_state,
                        "confidence": selected_checkbox.discovery_confidence,
                    },
                    "verification": {
                        "state": selected_checkbox.verification_state,
                        "confidence": selected_checkbox.verification_confidence,
                        "reason": selected_checkbox.verification_reason,
                    },
                    "review_reason": selected_checkbox.review_reason,
                    "RapidOCR_source": selected_checkbox.source_text,
                }
            )
            with st.form("checkbox_override"):
                checkbox_states = list(CheckboxState)
                reviewed_state = st.selectbox(
                    "Verified state",
                    checkbox_states,
                    index=checkbox_states.index(selected_checkbox.state),
                    format_func=lambda value: value.value.replace("_", " ").title(),
                )
                checkbox_reason = st.text_input("Review reason")
                checkbox_submitted = st.form_submit_button("Record checkbox decision")
            if checkbox_submitted:
                try:
                    st.session_state.workflow = record_checkbox_override(
                        workflow,
                        selected_checkbox.id,
                        reviewed_state,
                        checkbox_reason,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    result.checkboxes = st.session_state.workflow.checkboxes
                    result.checkbox_corrections = st.session_state.workflow.checkbox_corrections
                    result.workflow_manifest = st.session_state.workflow.manifest()
                    result.markdown = render_result_markdown(result)
                    st.session_state.artifacts = build_local_artifacts(result)
                    st.rerun()
    downloads.download_button("HTML", st.session_state.artifacts.html, "document.html")
    downloads.download_button(
        "ZIP package",
        build_local_bundle(result),
        "agentic-extraction.zip",
        "application/zip",
        icon=":material/download:",
    )

st.subheader(":material/payments: Usage & Cost")
st.caption("Per 1M tokens: input $0.20 · cached input $0.02 · cache writes $0.25 · output $1.20")
current = st.session_state.result
history = [
    entry
    for entry in st.session_state.usage_history
    if isinstance(entry, dict) and isinstance(entry.get("usage"), dict)
]
if current is None or not history:
    st.caption("No run usage recorded in this session.")
else:
    usage = current.usage
    session_usage = aggregate_usage(
        [UsageRecord.model_validate(entry["usage"]) for entry in history]
    )
    session_ocr_seconds = sum(float(entry.get("rapidocr_seconds") or 0) for entry in history)
    st.markdown("**Session totals**")
    session_row = st.container(horizontal=True, border=True)
    session_row.metric("Session GPT calls", session_usage.call_count)
    session_row.metric("Session input tokens", _tokens(session_usage.input_tokens))
    session_row.metric("Session cached tokens", _tokens(session_usage.cached_input_tokens))
    session_row.metric(
        "Session cache-write tokens", _tokens(session_usage.cache_write_input_tokens)
    )
    session_row.metric("Session output tokens", _tokens(session_usage.output_tokens))
    session_row.metric("Session total tokens", _tokens(session_usage.total_tokens))
    session_row.metric(
        "Session GPT cost", _money(session_usage.total_cost_usd, session_usage.cost_status)
    )
    st.markdown("**Current run**")
    current_row = st.container(horizontal=True, border=True)
    current_row.metric("Current GPT calls", usage.call_count)
    current_row.metric("Mode", current.effective_mode.value)
    current_row.metric("GPT image pages", len(current.cloud_image_pages))
    current_row.metric("Input tokens", _tokens(usage.input_tokens))
    current_row.metric("Cached input tokens", _tokens(usage.cached_input_tokens))
    current_row.metric("Cache-write tokens", _tokens(usage.cache_write_input_tokens))
    current_row.metric("Output tokens", _tokens(usage.output_tokens))
    current_row.metric("Total tokens", _tokens(usage.total_tokens))
    cost_row = st.container(horizontal=True, border=True)
    cost_row.metric("Input cost", _money(usage.input_cost_usd, usage.cost_status))
    cost_row.metric("Output cost", _money(usage.output_cost_usd, usage.cost_status))
    cost_row.metric("Total cost", _money(usage.total_cost_usd, usage.cost_status))
    cost_row.metric("RapidOCR API cost", "$0.00")
    st.caption("Model: gpt-5.6-luna · reasoning effort: medium")
    st.caption(
        f"RapidOCR processing: {current.timings.get('ocr_seconds', 0):.3f}s · "
        f"session total: {session_ocr_seconds:.3f}s · hardware cost: unavailable"
    )
    if usage.calls:
        st.markdown("**Per-call usage**")
        st.dataframe(_call_rows(usage.calls), width="stretch", hide_index=True)
        st.markdown("**Per-page usage (estimated)**")
        st.dataframe(_page_usage_rows(usage.calls), width="stretch", hide_index=True)
    with st.expander("Session usage history"):
        st.dataframe(
            [
                {
                    "mode": entry["mode"],
                    "calls": entry["usage"]["call_count"],
                    "total_tokens": entry["usage"]["total_tokens"],
                    "cost_usd": entry["usage"]["total_cost_usd"],
                    "cost_status": entry["usage"]["cost_status"],
                    "rapidocr_seconds": entry.get("rapidocr_seconds"),
                }
                for entry in history
            ],
            width="stretch",
            hide_index=True,
        )
