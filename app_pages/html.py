"""Source-faithful coordinate-positioned HTML viewer.

Responsible for: displaying the generated coordinate-positioned HTML in an
interactive Streamlit iframe with download capabilities.

Must not: generate or modify HTML directly; reads `artifacts.html` from
`st.session_state["artifacts"]`.

Next: `agentic_extractor.layout_html` for how coordinate HTML is built, and
`streamlit_app.py` for document parsing.
"""

import streamlit as st

from agentic_extractor.ui_state import output_file_name

st.title(":material/web: HTML")
st.caption("Source page rasters with selectable RapidOCR and accepted Luna text coordinates.")

artifacts = st.session_state.get("artifacts")
result = st.session_state.get("result")
if artifacts is None or result is None:
    st.info("No processed HTML is available. Process a document on the Parse page first.")
    st.stop()

name = str(result.document_metadata.get("file_name", "Processed document"))
pages = ", ".join(str(page) for page in result.selected_pages)
with st.container(horizontal=True, border=True):
    st.text(name)
    st.caption(f"Selected pages: {pages}")
    st.download_button(
        "Download HTML",
        artifacts.html,
        output_file_name(name, ".html"),
        "text/html",
    )

st.iframe(artifacts.html.decode("utf-8"), height=900)
