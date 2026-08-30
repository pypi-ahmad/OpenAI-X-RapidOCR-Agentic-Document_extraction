"""Structured extraction results and field review page."""

import json

import streamlit as st

from agentic_extractor.artifacts import build_local_artifacts
from agentic_extractor.workflow import record_user_override

st.title(":material/data_object: Extract")
workflow = st.session_state.get("workflow")
result = st.session_state.get("result")
if workflow is None:
    st.info("Run Parse with the Extract workflow enabled to populate this page.")
elif not workflow.extracted_fields:
    st.info("No fields were produced. Define a schema on the Parse page and run again.")
else:
    st.dataframe(
        [item.model_dump(mode="json") for item in workflow.extracted_fields],
        width="stretch",
        hide_index=True,
    )
    review_fields = [field for field in workflow.extracted_fields if field.requires_review]
    if review_fields:
        with st.expander(":material/rate_review: Field review", expanded=True):
            selected_path = st.selectbox("Field", [field.path for field in review_fields])
            selected_field = next(field for field in review_fields if field.path == selected_path)
            page = next(
                (
                    item
                    for item in (result.pages if result is not None else [])
                    if item.page == selected_field.source_page
                ),
                None,
            )
            if page:
                st.image(page.image_bytes, caption=f"Source page {page.page}", width="stretch")
                raw_texts = page.raw_evidence.get("texts", [])
                st.text_area(
                    "Raw OCR evidence",
                    value=(
                        "\n".join(str(item) for item in raw_texts)
                        if isinstance(raw_texts, list)
                        else str(raw_texts)
                    ),
                    disabled=True,
                )
            with st.form("extract_field_override"):
                override = st.text_input(
                    "Accepted or corrected value", value=str(selected_field.value)
                )
                reason = st.text_input("Review reason")
                submitted = st.form_submit_button("Record reviewed value")
            if submitted:
                try:
                    value = json.loads(override)
                except json.JSONDecodeError:
                    value = override
                st.session_state.workflow = record_user_override(
                    workflow, selected_field.path, value, reason
                )
                if result is not None:
                    result.workflow_manifest = st.session_state.workflow.manifest()
                    st.session_state.artifacts = build_local_artifacts(result)
                st.rerun()
