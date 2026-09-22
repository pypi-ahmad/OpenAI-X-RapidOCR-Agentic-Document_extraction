"""Source-linked visual review; submitting a decision never invokes an engine."""

import streamlit as st

from agentic_extractor.artifacts import build_local_artifacts
from agentic_extractor.openai_refiner import CloudResult, OpenAIRefiner
from agentic_extractor.pipeline import render_result_markdown
from agentic_extractor.rich_document import record_visual_review, visual_issues, visual_status
from agentic_extractor.workflow import WorkflowState


def render_visual_review(result, workflow, include_atomic_grounding: bool) -> bool:
    cloud = CloudResult.model_validate(result.cloud_output) if result.cloud_output else None
    if cloud is None or not cloud.visual_objects:
        return False
    pages = {page.page: page for page in result.pages}
    objects = [item for item in cloud.visual_objects if item.page in pages]
    if not objects:
        return False
    with st.expander("Visual content review", expanded=True):
        st.caption("Model-verified is not human-approved. Raw OCR remains unchanged.")
        identifier = st.selectbox("Visual object", [item.id for item in objects])
        item = next(item for item in objects if item.id == identifier)
        page = pages[item.page]
        status, content = visual_status(item, page.visual_audits)
        left, right = st.columns(2)
        left.image(
            OpenAIRefiner._region_crop(page.original_image_bytes or page.image_bytes, item.bbox),
            caption=f"Source page {item.page} · {item.kind}",
            width="stretch",
        )
        right.text(content)
        right.caption(f"Status: {status}")
        with st.form(f"visual-review-{identifier}"):
            action = st.selectbox("Decision", ["approve", "correct", "reject"])
            correction = st.text_area("Corrected content", content)
            reason = st.text_input("Review reason")
            submitted = st.form_submit_button("Record visual decision")
        if submitted:
            try:
                page.visual_audits = record_visual_review(
                    item, page.visual_audits, action, reason, correction
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                result.markdown = render_result_markdown(result)
                if workflow is not None:
                    workflow.review_required = [
                        message
                        for message in workflow.review_required
                        if not message.startswith("Visual object ")
                    ]
                    workflow.review_required.extend(
                        visual_issues(
                            result.pages,
                            cloud.visual_objects,
                            [audit for source in result.pages for audit in source.visual_audits],
                        )
                    )
                    workflow.review_items = [
                        entry
                        for entry in workflow.review_items
                        if entry.message in workflow.review_required
                    ]
                    if not workflow.review_required and not workflow.errors:
                        workflow.current_state = WorkflowState.ACCEPTED
                    result.workflow_manifest = workflow.manifest()
                st.session_state.artifacts = build_local_artifacts(
                    result, include_atomic_grounding=include_atomic_grounding
                )
                return True
    return False
