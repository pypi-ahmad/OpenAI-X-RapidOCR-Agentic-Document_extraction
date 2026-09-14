"""Markdown-only document chat models and local retrieval.

Responsible for: session-only document representation (`ProcessedMarkdownDocument`),
lexical chunking, document scope filtering (max 12 documents), excerpt retrieval
scoring, and citation payload generation for document-grounded conversation.

Must not: accept or expose original document binaries, page rasters, raw OCR
blocks, or unparsed files to the model; must not allow conversational drift
outside the grounded Markdown excerpts.

Next: `app_pages/chat.py` for the Streamlit UI presentation of chat, and
`openai_refiner.py` for execution of the chat prompt.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

CHAT_CONTEXT_CHARACTERS = 40_000
CHAT_MAX_EXCERPTS = 12
CHAT_CHUNK_CHARACTERS = 3_000
CHAT_HISTORY_MESSAGES = 6

_PAGE_MARKER = re.compile(r"(?m)^<!--\s*page:\s*(\d+)\s*-->\s*$")
_HEADING = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
_TERMS = re.compile(r"\w+", re.UNICODE)
_BROAD_REQUEST = re.compile(
    r"\b(summar(?:y|ize|ise)|overview|compare|contrast|key\s+points?)\b", re.IGNORECASE
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "do",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "this",
    "to",
    "what",
    "which",
    "with",
}


class ProcessedMarkdownDocument(BaseModel):
    """A session-only chat source that cannot contain original document bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    markdown: str = Field(min_length=1)
    selected_pages: list[int] = Field(default_factory=list)
    processing_status: str = "Completed"
    failed_pages: list[int] = Field(default_factory=list)


def validate_processed_document(value: object) -> ProcessedMarkdownDocument:
    """Restore a document from reload-safe session data or a stale model instance."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json")
    return ProcessedMarkdownDocument.model_validate(value)


class MarkdownExcerpt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    excerpt_id: str
    document_id: str
    document_name: str
    page: int | None = None
    heading: str | None = None
    markdown: str


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str


class DocumentChatAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disposition: Literal["answered", "off_topic", "insufficient_evidence"]
    answer_markdown: str = ""
    citation_ids: list[str] = Field(default_factory=list)


def recent_chat_history(messages: Iterable[Mapping[str, object]]) -> list[ChatTurn]:
    """Keep only the last three complete turns and only their visible text."""
    turns: list[ChatTurn] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            turns.append(ChatTurn(role=cast(Literal["user", "assistant"], role), content=content))
    return turns[-CHAT_HISTORY_MESSAGES:]


def retrieve_markdown_excerpts(
    documents: list[ProcessedMarkdownDocument],
    query: str,
    *,
    max_characters: int = CHAT_CONTEXT_CHARACTERS,
    max_excerpts: int = CHAT_MAX_EXCERPTS,
) -> list[MarkdownExcerpt]:
    """Select relevant Markdown without consulting source files or OCR objects."""
    chunks = {document.document_id: _document_excerpts(document) for document in documents}
    all_chunks = [chunk for document in documents for chunk in chunks[document.document_id]]
    if sum(len(chunk.markdown) for chunk in all_chunks) <= max_characters:
        return all_chunks
    if _BROAD_REQUEST.search(query):
        candidates = _stratified_chunks(documents, chunks, max_excerpts)
    else:
        candidates = _ranked_chunks(query, documents, chunks, max_excerpts)
    return _within_budget(candidates, max_characters, max_excerpts)


def _document_excerpts(document: ProcessedMarkdownDocument) -> list[MarkdownExcerpt]:
    pages = _page_segments(document.markdown)
    excerpts: list[MarkdownExcerpt] = []
    index = 0
    for page, page_markdown in pages:
        sections = _heading_sections(page_markdown)
        for heading, section in sections:
            for chunk in _split_section(section):
                if not chunk.strip():
                    continue
                index += 1
                excerpts.append(
                    MarkdownExcerpt(
                        excerpt_id=f"{document.document_id}:p{page or 0}:s{index}",
                        document_id=document.document_id,
                        document_name=document.display_name,
                        page=page,
                        heading=heading,
                        markdown=chunk.strip(),
                    )
                )
    return excerpts


def _page_segments(markdown: str) -> list[tuple[int | None, str]]:
    markers = list(_PAGE_MARKER.finditer(markdown))
    if not markers:
        return [(None, markdown)]
    segments: list[tuple[int | None, str]] = []
    prefix = markdown[: markers[0].start()].strip()
    if prefix:
        segments.append((None, prefix))
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(markdown)
        segments.append((int(marker.group(1)), markdown[marker.end() : end].strip()))
    return segments


def _heading_sections(markdown: str) -> list[tuple[str | None, str]]:
    headings = list(_HEADING.finditer(markdown))
    if not headings:
        return [(None, markdown)]
    sections: list[tuple[str | None, str]] = []
    prefix = markdown[: headings[0].start()].strip()
    if prefix:
        sections.append((None, prefix))
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        sections.append((heading.group(2).strip(), markdown[heading.start() : end].strip()))
    return sections


def _split_section(section: str) -> list[str]:
    if len(section) <= CHAT_CHUNK_CHARACTERS:
        return [section]
    parts = re.split(r"\n{2,}", section)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}\n\n{part}".strip() if current else part
        if current and len(candidate) > CHAT_CHUNK_CHARACTERS:
            chunks.append(current)
            current = part
        else:
            current = candidate
        while len(current) > CHAT_CHUNK_CHARACTERS:
            chunks.append(current[:CHAT_CHUNK_CHARACTERS])
            current = current[CHAT_CHUNK_CHARACTERS:]
    if current:
        chunks.append(current)
    return chunks


def _stratified_chunks(
    documents: list[ProcessedMarkdownDocument],
    chunks: dict[str, list[MarkdownExcerpt]],
    limit: int,
) -> list[MarkdownExcerpt]:
    selected: list[MarkdownExcerpt] = []
    per_document = max(1, limit // max(1, len(documents)))
    for document in documents:
        source = chunks[document.document_id]
        if len(source) <= per_document:
            selected.extend(source)
            continue
        indices = {
            round(position * (len(source) - 1) / max(1, per_document - 1))
            for position in range(per_document)
        }
        selected.extend(source[index] for index in sorted(indices))
    return selected[:limit]


def _ranked_chunks(
    query: str,
    documents: list[ProcessedMarkdownDocument],
    chunks: dict[str, list[MarkdownExcerpt]],
    limit: int,
) -> list[MarkdownExcerpt]:
    query_terms = [term for term in _tokenize(query) if term not in _STOPWORDS]
    query_counts = Counter(query_terms)

    def score(chunk: MarkdownExcerpt) -> tuple[float, int]:
        # Scoring heuristics: heading matches receive 2x weight over body text,
        # exact query substring match adds +5 bonus, and shorter chunks break ties.
        text_counts = Counter(_tokenize(chunk.markdown))
        heading_counts = Counter(_tokenize(chunk.heading or ""))
        relevance = sum(
            count * (text_counts[term] + 2 * heading_counts[term])
            for term, count in query_counts.items()
        )
        if query.strip() and query.casefold() in chunk.markdown.casefold():
            relevance += 5
        return float(relevance), -len(chunk.markdown)

    ranked = sorted(
        (chunk for document in documents for chunk in chunks[document.document_id]),
        key=score,
        reverse=True,
    )
    # Fair representation: ensure each selected document contributes its top-scoring
    # chunk before filling remaining quota with global top matches.
    selected: list[MarkdownExcerpt] = []
    for document in documents:
        document_chunks = chunks[document.document_id]
        if document_chunks:
            selected.append(max(document_chunks, key=score))
    selected_ids = {chunk.excerpt_id for chunk in selected}
    selected.extend(
        chunk for chunk in ranked if chunk.excerpt_id not in selected_ids and score(chunk)[0] > 0
    )
    return selected[:limit]


def _within_budget(
    candidates: list[MarkdownExcerpt], max_characters: int, max_excerpts: int
) -> list[MarkdownExcerpt]:
    selected: list[MarkdownExcerpt] = []
    used = 0
    for candidate in candidates:
        if len(selected) >= max_excerpts:
            break
        size = len(candidate.markdown)
        if used + size <= max_characters:
            selected.append(candidate)
            used += size
    return selected


def _tokenize(value: str) -> list[str]:
    return [term.casefold() for term in _TERMS.findall(value)]


def safe_chat_text(answer: DocumentChatAnswer) -> str:
    if answer.disposition == "off_topic":
        return (
            "I can only help with questions grounded in the selected document Markdown. "
            "Please ask about those documents."
        )
    if answer.disposition == "insufficient_evidence":
        return (
            "I couldn't find enough support for that in the selected document Markdown. "
            "Try a narrower document-related question."
        )
    return answer.answer_markdown
