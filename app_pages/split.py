"""Document split results page.

Read-only view over `st.session_state["workflow"]`. Must not call
`run_agent_workflow` or any refinement/OCR function itself — Split runs only
as part of Parse (see streamlit_app.py). Next: `agentic_extractor.workflow`
for how `.splits` and `.review_items` are produced.
"""

import streamlit as st

st.title(":material/call_split: Split")
workflow = st.session_state.get("workflow")
if workflow is None:
    st.info("Run Parse with the Split workflow enabled to populate this page.")
elif not workflow.splits:
    st.info("No split boundaries were produced. Enable Split on the Parse page and run again.")
else:
    st.dataframe(
        [item.model_dump(mode="json") for item in workflow.splits],
        width="stretch",
        hide_index=True,
    )
    reviews = [item for item in workflow.review_items if item.stage == "split"]
    if reviews:
        st.warning("Split review is required.")
        st.dataframe(
            [item.model_dump(mode="json") for item in reviews], width="stretch", hide_index=True
        )
