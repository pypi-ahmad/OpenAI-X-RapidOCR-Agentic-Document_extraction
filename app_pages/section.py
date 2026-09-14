"""Section outline results page.

Read-only view over `st.session_state["workflow"]`. Must not call
`run_agent_workflow` or any refinement/OCR function itself — Section runs
only as part of Parse (see streamlit_app.py). Next: `agentic_extractor.workflow`
for how `.sections` and `.review_items` are produced.
"""

import streamlit as st

st.title(":material/account_tree: Section")
workflow = st.session_state.get("workflow")
if workflow is None:
    st.info("Run Parse with the Section workflow enabled to populate this page.")
elif not workflow.sections:
    st.info("No sections were produced. Enable Section on the Parse page and run again.")
else:
    st.dataframe(
        [item.model_dump(mode="json") for item in workflow.sections],
        width="stretch",
        hide_index=True,
    )
    reviews = [item for item in workflow.review_items if item.stage == "section"]
    if reviews:
        st.warning("Section review is required.")
        st.dataframe(
            [item.model_dump(mode="json") for item in reviews], width="stretch", hide_index=True
        )
