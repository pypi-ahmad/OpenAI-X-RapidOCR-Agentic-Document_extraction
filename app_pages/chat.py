"""Document-only chat over generated Parse Markdown.

Responsible for: multi-document chat interaction grounded exclusively in
generated Parse Markdown excerpts, presenting questions, stream responses,
and source citations in Streamlit.

Must not: send original document bytes, images, or ungrounded content to chat.

Next: `agentic_extractor.document_chat` for retrieval and scoring logic, and
`agentic_extractor.openai_refiner` for LLM completion execution.
"""

from __future__ import annotations

import streamlit as st

from agentic_extractor.config import SETTINGS
from agentic_extractor.document_chat import (
    MarkdownExcerpt,
    recent_chat_history,
    retrieve_markdown_excerpts,
    safe_chat_text,
    validate_processed_document,
)
from agentic_extractor.openai_refiner import OpenAIConfigurationError, OpenAIRefiner


@st.cache_resource(show_spinner=False)
def get_chat_refiner() -> OpenAIRefiner:
    return OpenAIRefiner()


def show_sources(source_values: list[dict[str, object]]) -> None:
    if not source_values:
        return
    st.caption("Sources: " + ", ".join(str(item["excerpt_id"]) for item in source_values))
    with st.expander("Source excerpts"):
        for value in source_values:
            excerpt = MarkdownExcerpt.model_validate(value)
            location = excerpt.document_name
            if excerpt.page is not None:
                location += f" · page {excerpt.page}"
            if excerpt.heading:
                location += f" · {excerpt.heading}"
            st.markdown(f"**{excerpt.excerpt_id}** — {location}")
            st.code(excerpt.markdown, language="markdown", wrap_lines=True)


st.title(":material/chat: Document chat")
st.caption("Answers use generated output Markdown only. Original files are never sent to chat.")

stored = st.session_state.setdefault("processed_documents", {})
documents = {
    document_id: validate_processed_document(value) for document_id, value in stored.items()
}
st.session_state.processed_documents = {
    document_id: document.model_dump(mode="json") for document_id, document in documents.items()
}
st.session_state.setdefault("chat_messages", [])
st.session_state.setdefault("chat_scope", ())
histories = st.session_state.setdefault("chat_histories", {})

if not documents:
    st.info("No processed Markdown is available. Process a document on the Parse page first.")
    st.stop()

options = list(documents)
existing = [item for item in st.session_state.get("chat_document_ids", []) if item in documents]
if not existing:
    st.session_state.chat_document_ids = [options[-1]]

selected_ids = st.multiselect(
    "Documents in scope",
    options,
    key="chat_document_ids",
    persist_state="session",
    max_selections=12,
    help="Select up to 12 processed Markdown documents.",
    format_func=lambda document_id: (
        f"{documents[document_id].display_name} · "
        f"pages {', '.join(map(str, documents[document_id].selected_pages))}"
    ),
)
if len(selected_ids) > 12 or any(document_id not in documents for document_id in selected_ids):
    st.error("The selected document scope is invalid. Choose processed documents from the list.")
    st.stop()

scope = tuple(sorted(selected_ids))
if scope != tuple(st.session_state.chat_scope):
    previous_scope = tuple(st.session_state.chat_scope)
    if previous_scope:
        histories[previous_scope] = list(st.session_state.chat_messages)
    st.session_state.chat_scope = scope
    st.session_state.chat_messages = list(histories.get(scope, []))

selected_documents = [documents[document_id] for document_id in selected_ids]
with st.container(border=True):
    st.markdown("**Current document scope**")
    if not selected_documents:
        st.caption("No documents selected.")
    for document in selected_documents:
        pages = ", ".join(map(str, document.selected_pages)) or "not recorded"
        status = f" · {document.processing_status}"
        if document.failed_pages:
            status += f" · failed pages {', '.join(map(str, document.failed_pages))}"
        st.write(f":material/article: {document.display_name} · pages {pages}{status}")

for message in st.session_state.chat_messages:
    with st.chat_message(str(message["role"])):
        st.markdown(str(message["content"]))
        if message["role"] == "assistant":
            show_sources(list(message.get("sources", [])))
            usage = message.get("usage")
            if isinstance(usage, dict):
                st.caption(
                    f"Luna calls: {usage.get('call_count', 0)} · "
                    f"tokens: {usage.get('total_tokens') or 'unavailable'} · "
                    f"cost: "
                    + (
                        f"${usage['total_cost_usd']:.6f}"
                        if isinstance(usage.get("total_cost_usd"), int | float)
                        else "unavailable"
                    )
                )

if not SETTINGS.openai_configured:
    st.error(
        "OpenAI is not configured. Add OPENAI_API_KEY to the launcher environment, restart the "
        "app, and retry."
    )

question = st.chat_input(
    "Ask about the selected document Markdown",
    disabled=not selected_documents or not SETTINGS.openai_configured,
    submit_mode="disable",
)
if question:
    prior_messages = list(st.session_state.chat_messages)
    st.session_state.chat_messages.append({"role": "user", "content": question})
    histories[scope] = list(st.session_state.chat_messages)
    with st.chat_message("user"):
        st.markdown(question)

    prior_user_messages = [
        str(message["content"]) for message in prior_messages if message.get("role") == "user"
    ][-2:]
    retrieval_query = "\n".join([*prior_user_messages, question])
    excerpts = retrieve_markdown_excerpts(selected_documents, retrieval_query)
    history = recent_chat_history(prior_messages)
    try:
        with st.chat_message("assistant"), st.spinner("Searching generated Markdown…"):
            refiner = get_chat_refiner()
            refiner.validate_configuration()
            answer, usage = refiner.answer_document_question(
                question, selected_documents, excerpts, history
            )
            content = safe_chat_text(answer)
            cited = set(answer.citation_ids)
            sources = [excerpt for excerpt in excerpts if excerpt.excerpt_id in cited]
            st.markdown(content)
            show_sources([source.model_dump(mode="json") for source in sources])
        usage_value = usage.model_dump(mode="json")
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": content,
                "disposition": answer.disposition,
                "sources": [source.model_dump(mode="json") for source in sources],
                "usage": usage_value,
            }
        )
        histories[scope] = list(st.session_state.chat_messages)
        st.session_state.setdefault("usage_history", []).append(
            {"mode": "Document chat", "usage": usage_value, "rapidocr_seconds": 0}
        )
    except (OpenAIConfigurationError, RuntimeError) as exc:
        st.error(str(exc))
