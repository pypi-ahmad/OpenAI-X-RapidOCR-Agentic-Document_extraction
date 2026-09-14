"""Classification results page.

Read-only view over `st.session_state["workflow"]`. Must not call
`run_agent_workflow` or any refinement/OCR function itself — Classify runs
only as part of Parse (see streamlit_app.py). Next: `agentic_extractor.workflow`
for how `.classifications` and `.review_items` are produced.
"""

import streamlit as st

st.title(":material/category: Classify")
workflow = st.session_state.get("workflow")
if workflow is None:
    st.info("Run Parse with the Classify workflow enabled to populate this page.")
elif not workflow.classifications:
    st.info("No classifications were produced. Enable Classify on the Parse page and run again.")
else:
    st.dataframe(
        [item.model_dump(mode="json") for item in workflow.classifications],
        width="stretch",
        hide_index=True,
    )
    reviews = [item for item in workflow.review_items if item.stage == "classify"]
    if reviews:
        st.warning("Classification review is required.")
        st.dataframe(
            [item.model_dump(mode="json") for item in reviews], width="stretch", hide_index=True
        )
