"""Auditable visual additions. Raw OCR blocks are never rewritten here."""

from __future__ import annotations

import hashlib
import html
import io
import math
from contextlib import closing
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from agentic_extractor.config import SETTINGS
from agentic_extractor.parse import PageParse, ParseChunk


class VisualObject(BaseModel):
    id: str
    page: int = Field(ge=1)
    kind: Literal["text", "figure", "chart", "equation"]
    bbox: list[float]
    content: str = Field(max_length=12000)
    reading_order: int = Field(ge=1)
    source_block_ids: list[str] = Field(default_factory=list)

    @field_validator("bbox")
    @classmethod
    def valid_box(cls, box: list[float]) -> list[float]:
        if (
            len(box) != 4
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in box)
            or box[0] >= box[2]
            or box[1] >= box[3]
        ):
            raise ValueError("bbox must have positive area inside the normalized page")
        return box


class DocumentLink(BaseModel):
    source_id: str
    target_id: str
    relation: Literal["continues", "caption_of", "parent_of"]


def prepare_inspection_images(
    pages: list[PageParse], source: bytes, items: list[VisualObject]
) -> None:
    """Render only inspected PDF pages at up to 300 DPI; never rerun document OCR."""
    if not source.startswith(b"%PDF-"):
        return
    import pypdfium2 as pdfium

    wanted = {item.page for item in items[:16]}
    try:
        with pdfium.PdfDocument(source) as document:
            for page in pages:
                if page.page not in wanted:
                    continue
                with closing(document[page.page - 1]) as pdf_page:
                    width, height = pdf_page.get_size()
                    scale = min(
                        300 / 72, math.sqrt(SETTINGS.max_image_pixels / (width * height)) * 0.99
                    )
                    bitmap = pdf_page.render(scale=scale)
                    try:
                        image = bitmap.to_pil().convert("RGB")
                        if image.width * image.height > SETTINGS.max_image_pixels:
                            continue
                        output = io.BytesIO()
                        image.save(output, "PNG")
                        page.inspection_image_bytes = output.getvalue()
                    finally:
                        bitmap.close()
    except Exception as exc:
        for page in pages:
            if page.page in wanted:
                page.warnings.append(
                    f"Higher-resolution inspection unavailable: {type(exc).__name__}."
                )


class VisualDecision(BaseModel):
    id: str
    supported: bool
    transcription: str
    reason: str


class VisualVerification(BaseModel):
    decisions: list[VisualDecision] = Field(default_factory=list)


def proposal_digest(item: VisualObject) -> str:
    """Bind every approval to the complete, immutable proposal, not just its ID."""
    return hashlib.sha256(item.model_dump_json().encode()).hexdigest()


def prepare_visual_objects(items: list[VisualObject]) -> list[VisualObject]:
    """Assign stable application IDs; a model cannot choose audit record identities."""
    prepared = []
    seen = set()
    for item in items:
        item = item.model_copy(deep=True)
        item.id = ""
        digest = proposal_digest(item)[:20]
        if digest in seen:
            continue
        seen.add(digest)
        item.id = f"visual-p{item.page}-{digest}"
        prepared.append(item)
    return prepared


def visual_status(item: VisualObject, audits: list[dict]) -> tuple[str, str]:
    for audit in reversed(audits):
        if audit.get("id") == item.id and audit.get("digest") == proposal_digest(item):
            return str(audit["status"]), str(audit.get("content", item.content))
    return "pending", item.content


def record_visual_review(
    item: VisualObject,
    audits: list[dict],
    action: str,
    reason: str,
    content: str | None = None,
) -> list[dict]:
    if not reason.strip():
        raise ValueError("A review reason is required.")
    if action not in {"approve", "correct", "reject"}:
        raise ValueError("Unknown review action.")
    if action == "correct" and (content is None or not content.strip()):
        raise ValueError("A correction must contain text.")
    return [
        *audits,
        {
            "id": item.id,
            "digest": proposal_digest(item),
            "status": "rejected" if action == "reject" else "human_approved",
            "action": action,
            "actor": "user",
            "content": content if action == "correct" else item.content,
            "reason": reason.strip(),
            "timestamp": datetime.now(UTC).isoformat(),
        },
    ]


def visual_issues(
    pages: list[PageParse], items: list[VisualObject], audits: list[dict]
) -> list[str]:
    page_map = {page.page: page for page in pages}
    issues = []
    ids = [item.id for item in items]
    for item in items:
        page = page_map.get(item.page)
        status, _ = visual_status(item, audits)
        if status == "rejected":
            continue
        invalid = (
            page is None
            or ids.count(item.id) != 1
            or not set(item.source_block_ids) <= {block.id for block in page.blocks}
        )
        if invalid or status == "pending":
            issues.append(
                f"Visual object {item.id!r} requires review: "
                + ("invalid source references." if invalid else "crop not verified.")
            )
    return issues


def apply_visual_objects(
    pages: list[PageParse],
    items: list[VisualObject],
    audits: list[dict],
    semantic_orders: dict[tuple[int, str], int] | None = None,
) -> list[str]:
    issues = visual_issues(pages, items, audits)
    invalid_ids = {item.id for item in items if any(repr(item.id) in issue for issue in issues)}
    for page in pages:
        # Rendering is idempotent, including after a human correction.
        page.chunks = [chunk for chunk in page.chunks if chunk.provenance != "gpt-visual"]
        if any(item.page == page.page for item in items):
            for chunk in page.chunks:
                for (number, region_id), order in (semantic_orders or {}).items():
                    if number == page.page and (
                        chunk.id == region_id or chunk.id.startswith(region_id + "-")
                    ):
                        chunk.reading_order = order
                        break
        for item in items:
            if item.page != page.page:
                continue
            status, content = visual_status(item, audits)
            if status == "rejected":
                continue
            if item.id in invalid_ids or status not in {"model_verified", "human_approved"}:
                content = "[VISUAL_CONTENT_REQUIRES_REVIEW]"
            elif item.kind in {"figure", "chart"}:
                content = (
                    f"{item.kind.title()} description (not a transcription): {html.escape(content)}"
                )
            elif item.kind == "equation":
                content = "$$\n" + content.replace("$$", "").strip() + "\n$$"
            else:
                content = html.escape(content)
            page.chunks.append(
                ParseChunk(
                    id=item.id,
                    page=item.page,
                    type=item.kind,
                    reading_order=item.reading_order,
                    text=content,
                    markdown=content,
                    source_block_ids=item.source_block_ids,
                    bbox=item.bbox,
                    raw_scores=[],
                    provenance="gpt-visual",
                    verification=status,
                )
            )
        page.chunks.sort(key=lambda chunk: chunk.reading_order)
        for order, chunk in enumerate(page.chunks, 1):
            chunk.reading_order = order
    return issues


def validate_links(pages: list[PageParse], links: list[DocumentLink]) -> list[str]:
    """Reject unknown endpoints, backwards continuation, duplicates, and directed cycles."""
    nodes = {chunk.id: (page.page, chunk.reading_order) for page in pages for chunk in page.chunks}
    edges: dict[str, set[str]] = {}
    seen = set()
    issues = []
    for link in links:
        key = (link.source_id, link.target_id, link.relation)
        source, target = nodes.get(link.source_id), nodes.get(link.target_id)
        pending = [link.target_id]
        visited = set()
        cyclic = False
        while pending:
            node = pending.pop()
            if node == link.source_id:
                cyclic = True
                break
            if node not in visited:
                visited.add(node)
                pending.extend(edges.get(node, ()))
        if (
            source is None
            or target is None
            or key in seen
            or cyclic
            or (link.relation == "continues" and source >= target)
        ):
            issues.append(f"Document link {key!r} requires review: invalid relationship.")
        else:
            edges.setdefault(link.source_id, set()).add(link.target_id)
            seen.add(key)
    return issues
