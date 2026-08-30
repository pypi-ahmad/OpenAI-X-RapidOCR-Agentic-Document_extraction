"""Streamlit entry point and page navigation."""

import streamlit as st

st.set_page_config(
    page_title="Agentic document extraction",
    page_icon=":material/document_scanner:",
    layout="wide",
)

st.session_state.setdefault("processed_documents", {})
st.session_state.setdefault("current_processed_document_id", None)
st.session_state.setdefault("chat_messages", [])
st.session_state.setdefault("chat_scope", ())

page = st.navigation(
    [
        st.Page(
            "streamlit_app.py",
            title="Parse",
            icon=":material/document_scanner:",
            default=True,
        ),
        st.Page("app_pages/classify.py", title="Classify", icon=":material/category:"),
        st.Page("app_pages/section.py", title="Section", icon=":material/account_tree:"),
        st.Page("app_pages/split.py", title="Split", icon=":material/call_split:"),
        st.Page("app_pages/extract.py", title="Extract", icon=":material/data_object:"),
        st.Page("app_pages/chat.py", title="Chat", icon=":material/chat:"),
    ],
    position="top",
)
page.run()
