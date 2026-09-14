"""Streamlit entry point and page navigation.

Responsible for: page config and the top-level `st.Page` navigation table only.
Must not: run OCR, call the pipeline/workflow, or hold result state itself —
every page (including `streamlit_app.py`) reads/writes canonical results
through `st.session_state` instead, so results survive Streamlit reruns and
page switches. Next: `streamlit_app.py` (the Parse page and default entry).
"""

import streamlit as st


def reset_session() -> None:
    """Clear user-owned state only after an explicit reset action.

    Clears Streamlit session state only. It does not touch the process-memory
    caches in `agentic_extractor.cache` (rendered pages, OCR, layout, table
    results) — those live for the process lifetime and are reused by the next
    session. See docs/TECHNICAL.md#persistence-paths.
    """
    st.session_state.clear()


st.set_page_config(
    page_title="Agentic document extraction",
    page_icon=":material/document_scanner:",
    layout="wide",
)

st.session_state.setdefault("processed_documents", {})
st.session_state.setdefault("current_processed_document_id", None)
st.session_state.setdefault("chat_messages", [])
st.session_state.setdefault("chat_scope", ())
st.session_state.setdefault("chat_histories", {})

with st.sidebar:
    st.button(
        "Reset session",
        icon=":material/restart_alt:",
        width="stretch",
        on_click=reset_session,
    )

# Parse (streamlit_app.py) is the only page that calls run_agent_workflow.
# Every other page below only reads st.session_state["result"/"workflow"/...];
# adding OCR/refinement calls to one of them would silently duplicate engine
# cost and break the single-canonical-workflow contract described in
# docs/ARCHITECTURE.md.
page = st.navigation(
    [
        st.Page(
            "streamlit_app.py",
            title="Parse",
            icon=":material/document_scanner:",
            default=True,
        ),
        st.Page("app_pages/html.py", title="HTML", icon=":material/web:"),
        st.Page("app_pages/classify.py", title="Classify", icon=":material/category:"),
        st.Page("app_pages/section.py", title="Section", icon=":material/account_tree:"),
        st.Page("app_pages/split.py", title="Split", icon=":material/call_split:"),
        st.Page("app_pages/extract.py", title="Extract", icon=":material/data_object:"),
        st.Page("app_pages/chat.py", title="Chat", icon=":material/chat:"),
        st.Page(
            "app_pages/diagnostics.py",
            title="Diagnostics",
            icon=":material/monitoring:",
        ),
    ],
    position="top",
)
page.run()
