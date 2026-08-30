"""Document split results page."""

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
