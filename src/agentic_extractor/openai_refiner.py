"""The only OpenAI boundary. Model policy is intentionally not caller-configurable.

Responsible for: the single external boundary calling OpenAI Responses API
with fixed model `gpt-5.6-luna` (`reasoning_effort="medium"`). Handles prompt
template rendering, structured output parsing, token usage collection, visual
crop routing, checkbox verification, dedicated table reviews, and document
chat turns.

Must not: allow callers to override the model or reasoning effort, expose
raw credentials to callers or log files, or accept model output without
rigorous schema validation against the untrusted cloud boundary.

Next: `pipeline.py`, which coordinates the interaction between local OCR
results and `OpenAIRefiner`.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal, cast

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, WithJsonSchema

from agentic_extractor.checkbox_vision import is_credible_checkbox_candidate
from agentic_extractor.config import SETTINGS, Settings
from agentic_extractor.costs import (
    MODEL_NAME,
    REASONING_EFFORT,
    TokenUsage,
    aggregate_usage,
    calculate_usage_cost,
    usage_and_cost_dict,
)
from agentic_extractor.document_chat import (
    ChatTurn,
    DocumentChatAnswer,
    MarkdownExcerpt,
    ProcessedMarkdownDocument,
)
from agentic_extractor.models import TableStructureEvidence, UsageRecord
from agentic_extractor.parse import PageParse, build_layout_chunks
from agentic_extractor.prompt_resources import PromptResource, load_prompt, render_prompt
from agentic_extractor.table_structure import (
    table_review_bbox,
    table_review_block_ids,
    table_word_evidence,
)
from agentic_extractor.visual_routing import block_requires_gpt_review

_CAPABILITY_PROMPTS = {
    "Parse": "capability-parse.md",
    "Classify": "capability-classify.md",
    "Section": "capability-section.md",
    "Split": "capability-split.md",
    "Extract": "capability-extract.md",
}

_BLOCK_EVIDENCE_COLUMNS = [
    "id",
    "type",
    "text",
    "confidence",
    "bbox",
    "polygon",
    "requires_gpt_review",
]
_LOCAL_SEMANTIC_REGION_COLUMNS = [
    "id",
    "label",
    "reading_order",
    "bbox",
    "confidence",
    "source_block_ids",
    "source_chunk_ids",
    "source",
]
_GROUNDING_COLUMNS = ["page", "block_id", "chunk_id", "type", "confidence", "bbox", "text"]


@dataclass(frozen=True, slots=True)
class _ContextPacket:
    text: str
    resources: list[PromptResource]
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _RefinementBatch:
    kind: Literal["compact", "full"]
    pages: list[PageParse]


type StructuredFieldValue = Annotated[
    Any,
    WithJsonSchema(
        {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "integer"},
                {"type": "boolean"},
                {"type": "null"},
            ]
        }
    ),
]


def _prompt_json(value: Any) -> str:
    """Serialize untrusted prompt data compactly without allowing delimiter injection."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _bbox_center_in(box: list[float], container: list[float]) -> bool:
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return container[0] <= x <= container[2] and container[1] <= y <= container[3]


def _local_semantic_regions(page: PageParse) -> list[list[Any]]:
    """Group every OCR block once using local layout evidence before Luna."""
    blocks = {block.id: block for block in page.blocks}
    block_order = {block.id: index for index, block in enumerate(page.blocks)}
    regions = {region.id: region for region in page.layout_regions}
    primary = {
        link.block_id: link.primary_region_id
        for link in page.layout_block_links
        if link.block_id in blocks and link.primary_region_id in regions
    }
    chunks = page.chunks or build_layout_chunks(page.blocks)
    chunk_by_block: dict[str, str] = {}
    for chunk in chunks:
        for block_id in chunk.source_block_ids:
            chunk_by_block.setdefault(block_id, chunk.id)

    pending: list[tuple[int, list[Any]]] = []
    grouped: set[str] = set()
    for region in page.layout_regions:
        source_ids = [block.id for block in page.blocks if primary.get(block.id) == region.id]
        if not source_ids:
            continue
        grouped.update(source_ids)
        pending.append(
            (
                min(block_order[block_id] for block_id in source_ids),
                [
                    region.id,
                    region.label,
                    0,
                    region.bbox,
                    region.score,
                    source_ids,
                    list(
                        dict.fromkeys(
                            chunk_by_block[block_id]
                            for block_id in source_ids
                            if block_id in chunk_by_block
                        )
                    ),
                    region.model,
                ],
            )
        )

    fallback = 0
    for chunk in chunks:
        source_ids = [
            block_id
            for block_id in chunk.source_block_ids
            if block_id in blocks and block_id not in grouped
        ]
        if not source_ids:
            continue
        fallback += 1
        grouped.update(source_ids)
        pending.append(
            (
                min(block_order[block_id] for block_id in source_ids),
                [
                    f"p{page.page}-sr-fallback-{fallback}",
                    chunk.type,
                    0,
                    _union_block_bboxes([blocks[block_id] for block_id in source_ids]),
                    None,
                    source_ids,
                    [chunk.id],
                    "geometry-fallback",
                ],
            )
        )

    for block in page.blocks:
        if block.id in grouped:
            continue
        fallback += 1
        pending.append(
            (
                block_order[block.id],
                [
                    f"p{page.page}-sr-fallback-{fallback}",
                    block.type,
                    0,
                    block.bbox,
                    None,
                    [block.id],
                    [],
                    "geometry-fallback",
                ],
            )
        )

    rows = [row for _, row in sorted(pending, key=lambda item: item[0])]
    for reading_order, row in enumerate(rows, 1):
        row[2] = reading_order
    return rows


def _union_block_bboxes(blocks: list[Any]) -> list[float] | None:
    boxes = [block.bbox for block in blocks if block.bbox]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _add_semantic_table_candidates(
    pages: list[PageParse], proposals: list[CloudSemanticRegion]
) -> None:
    """Route grounded Luna table regions through the existing table validator."""
    by_page: dict[int, list[CloudSemanticRegion]] = {}
    for proposal in proposals:
        if proposal.type == "table":
            by_page.setdefault(proposal.page, []).append(proposal)
    for page in pages:
        page_proposals = by_page.get(page.page, [])
        if not page_proposals:
            continue
        blocks = {block.id: block for block in page.blocks if block.source == "rapidocr"}
        existing_by_layout = {table.layout_region_id: table for table in page.table_structures}
        citation_counts: dict[str, int] = {}
        for proposal in page_proposals:
            for region_id in set(proposal.source_layout_region_ids) & set(existing_by_layout):
                citation_counts[region_id] = citation_counts.get(region_id, 0) + 1
        split_layout_ids = {region_id for region_id, count in citation_counts.items() if count > 1}
        retained = [
            table
            for table in page.table_structures
            if table.layout_region_id not in split_layout_ids
        ]
        occupied = {
            block_id for table in retained for block_id in table_review_block_ids(page, table)
        }
        candidate_ids = {table.id for table in retained}
        added: list[TableStructureEvidence] = []
        for proposal in page_proposals:
            source_ids = list(proposal.source_block_ids)
            source_id_set = set(source_ids)
            if (
                not source_ids
                or len(source_ids) != len(source_id_set)
                or not source_id_set <= set(blocks)
            ):
                continue
            cited_existing = set(proposal.source_layout_region_ids) & set(existing_by_layout)
            if cited_existing and not cited_existing <= split_layout_ids:
                continue
            if source_id_set & occupied:
                continue
            bbox = _union_block_bboxes([blocks[block_id] for block_id in source_ids])
            candidate_id = f"{proposal.id}-table"
            if bbox is None or candidate_id in candidate_ids:
                continue
            added.append(
                TableStructureEvidence(
                    id=candidate_id,
                    page=page.page,
                    layout_region_id=f"{proposal.id}-semantic-table",
                    bbox=bbox,
                    style="wireless",
                    classifier_score=0,
                    structure_score=0,
                    structure_model="SLANet_plus",
                    status="invalid",
                    review_required=True,
                    warnings=["Grounded semantic table candidate requires cell reconstruction."],
                )
            )
            candidate_ids.add(candidate_id)
            occupied.update(source_id_set)
        page.table_structures = [*retained, *added]


class OpenAIConfigurationError(RuntimeError):
    """OpenAI credentials or endpoint configuration cannot be used."""


class CloudEvidence(BaseModel):
    page: int = Field(ge=1)
    block_id: str | None = None
    chunk_id: str | None = None
    bbox: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Normalized [left, top, right, bottom] visual evidence coordinates.",
    )
    quote: str = Field(default="", description="Exact source substring when an ID is cited.")
    source: Literal["rapidocr", "gpt-visual"] = Field(
        default="rapidocr", description="Engine that directly supports this evidence reference."
    )


class CloudRefinement(BaseModel):
    page: int = Field(ge=1)
    block_id: str | None = Field(
        default=None,
        description="Existing OCR block ID, or a supplied local redaction candidate ID.",
    )
    corrected_text: str | None = Field(
        default=None, description="Complete replacement text for the cited block."
    )
    block_type: Literal["heading", "paragraph", "list_item", "table_row", "key_value"] | None = None
    reading_order: int | None = Field(default=None, ge=1)
    evidence: list[CloudEvidence] = Field(
        default_factory=list, description="Evidence that directly supports this refinement."
    )
    verified: bool = Field(default=False, description="True only with valid supporting evidence.")
    abstained: bool = Field(
        default=False, description="True when evidence cannot support a change."
    )
    warning: str | None = None


class CloudTableCell(BaseModel):
    row: int = Field(ge=1)
    column: int = Field(ge=1)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    tag: Literal["th", "td"] = "td"
    text: str = ""
    bbox: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description=(
            "Page-normalized visual bounds. Required only for a genuinely blank cell with no "
            "RapidOCR word or block IDs."
        ),
    )
    source_block_ids: list[str] = Field(default_factory=list)
    source_word_ids: list[str] = Field(default_factory=list)


class CloudTableReview(BaseModel):
    page: int = Field(ge=1)
    table_id: str
    outcome: Literal["confirmed", "corrected", "not_table", "abstained"]
    visible_cell_count: int | None = Field(
        default=None,
        ge=0,
        description="Count of visible HTML cells, including blank and spanning cells.",
    )
    cells: list[CloudTableCell] = Field(default_factory=list)
    excluded_source_block_ids: list[str] = Field(default_factory=list)
    excluded_source_word_ids: list[str] = Field(default_factory=list)
    evidence: list[CloudEvidence] = Field(default_factory=list)
    warning: str | None = None


class TableReviewResult(BaseModel):
    reviews: list[CloudTableReview] = Field(default_factory=list)


class CloudClassification(BaseModel):
    label: str
    scope: Literal["page", "document"] | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    evidence: list[CloudEvidence] = Field(
        default_factory=list, description="Evidence within the classified page range."
    )


class CloudSection(BaseModel):
    title: str
    level: int = Field(ge=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    source_block_ids: list[str] = Field(
        default_factory=list, description="Existing RapidOCR block IDs supporting this section."
    )
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="Existing layout chunk IDs supporting this section."
    )


class CloudSplit(BaseModel):
    name: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    label: str = "unknown"
    evidence: list[CloudEvidence] = Field(
        default_factory=list, description="Evidence at or immediately before the split boundary."
    )


class ExtractedField(BaseModel):
    path: str
    value: StructuredFieldValue = None
    status: str = Field(
        default="uncertain", description="Use verified only for an unambiguous grounded value."
    )
    confidence: float | None = Field(default=None, ge=0, le=1)
    abstention_reason: str | None = None
    evidence: list[CloudEvidence] = Field(
        default_factory=list, description="Evidence that directly supports the field value."
    )


class CloudCheckbox(BaseModel):
    id: str
    page: int = Field(ge=1)
    label: str = ""
    state: Literal["CHECKED", "UNCHECKED", "INDETERMINATE", "CROSSED_OUT", "NOT_DETERMINABLE"]
    control_bbox: list[float] = Field(
        min_length=4,
        max_length=4,
        description="Normalized checkbox-control coordinates, excluding its label.",
    )
    label_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    label_block_id: str | None = None
    label_chunk_id: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None
    evidence: list[CloudEvidence] = Field(default_factory=list)


class CloudSemanticRegion(BaseModel):
    """A grounded grouping proposal; displayed text always comes from its source blocks."""

    id: str
    page: int = Field(ge=1)
    type: Literal[
        "text",
        "heading",
        "list",
        "form",
        "table",
        "marginalia",
        "logo",
        "attestation",
        "figure",
        "scan_code",
    ]
    reading_order: int = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    source_layout_region_ids: list[str] = Field(default_factory=list)
    evidence: list[CloudEvidence] = Field(
        default_factory=list,
        description=(
            "Visual evidence required when PP-DocLayoutV3 cannot express the semantic type."
        ),
    )
    join_style: Literal["space", "line_break"] = "line_break"
    heading_level: int = Field(default=2, ge=1, le=6)


class CheckboxVerification(BaseModel):
    id: str
    page: int = Field(ge=1)
    control_status: Literal["checkbox", "not_checkbox", "uncertain"] = Field(
        description="Whether the crop visibly depicts a checkbox control."
    )
    state: Literal["CHECKED", "UNCHECKED", "INDETERMINATE", "CROSSED_OUT", "NOT_DETERMINABLE"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None


class CheckboxVerificationResult(BaseModel):
    verifications: list[CheckboxVerification] = Field(default_factory=list)


class CloudResult(BaseModel):
    refined_markdown: str
    reviewed_pages: list[int]
    refinements: list[CloudRefinement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    classifications: list[CloudClassification] = Field(default_factory=list)
    sections: list[CloudSection] = Field(default_factory=list)
    splits: list[CloudSplit] = Field(default_factory=list)
    extracted_fields: list[ExtractedField] = Field(default_factory=list)
    checkboxes: list[CloudCheckbox] = Field(default_factory=list)
    table_reviews: list[CloudTableReview] = Field(default_factory=list)
    semantic_regions: list[CloudSemanticRegion] = Field(default_factory=list)


class MarkdownWorkflowResult(BaseModel):
    reviewed_pages: list[int]
    warnings: list[str] = Field(default_factory=list)
    classifications: list[CloudClassification] = Field(default_factory=list)
    sections: list[CloudSection] = Field(default_factory=list)
    splits: list[CloudSplit] = Field(default_factory=list)
    extracted_fields: list[ExtractedField] = Field(default_factory=list)


class OpenAIRefiner:
    def __init__(self, client: Any | None = None, settings: Settings = SETTINGS) -> None:
        self.settings = settings
        try:
            self.client = client or OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                max_retries=2,
                timeout=120,
            )
        except RuntimeError as exc:
            raise OpenAIConfigurationError(
                "OpenAI is not configured. Add OPENAI_API_KEY to the launcher environment, "
                "restart the application, and retry."
            ) from exc
        self._configuration_validated = False

    def validate_configuration(self) -> None:
        """Validate credentials without sending document content."""
        if self._configuration_validated:
            return
        try:
            self.client.models.retrieve(MODEL_NAME)
        except Exception as exc:
            raise OpenAIConfigurationError(
                "OpenAI configuration could not access gpt-5.6-luna. Configure a valid "
                "OPENAI_API_KEY and, only when required, OPENAI_BASE_URL, then retry."
            ) from exc
        self._configuration_validated = True

    def answer_document_question(
        self,
        question: str,
        documents: list[ProcessedMarkdownDocument],
        excerpts: list[MarkdownExcerpt],
        history: list[ChatTurn],
    ) -> tuple[DocumentChatAnswer, UsageRecord]:
        """Answer from generated Markdown excerpts without accepting source-file objects."""
        prompt, prompt_resource = render_prompt(
            "document-chat-request.md",
            document_scope=_prompt_json(
                [
                    {
                        "document_id": document.document_id,
                        "display_name": document.display_name,
                        "selected_pages": document.selected_pages,
                        "processing_status": document.processing_status,
                        "failed_pages": document.failed_pages,
                    }
                    for document in documents
                ]
            ),
            source_excerpts=_prompt_json([excerpt.model_dump(mode="json") for excerpt in excerpts]),
            conversation_history=_prompt_json([turn.model_dump(mode="json") for turn in history]),
            user_question=_prompt_json(question),
        )
        started = time.perf_counter()
        response = self._request(
            [{"type": "input_text", "text": prompt}],
            DocumentChatAnswer,
            policy_name="document-chat-system.md",
        )
        elapsed = time.perf_counter() - started
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no structured document-chat response.")
        supplied = {excerpt.excerpt_id for excerpt in excerpts}
        cited = set(parsed.citation_ids)
        if not cited <= supplied or (
            parsed.disposition == "answered"
            and (not parsed.answer_markdown.strip() or not parsed.citation_ids)
        ):
            parsed = DocumentChatAnswer(disposition="insufficient_evidence")
        elif parsed.disposition != "answered":
            parsed.answer_markdown = ""
            parsed.citation_ids = []
        usage = self._read_usage(
            response,
            [],
            set(),
            elapsed,
            [prompt_resource],
            purpose="document_chat",
            context={
                "kind": "document_chat",
                "prompt_characters": len(prompt),
                "evidence_characters": sum(len(excerpt.markdown) for excerpt in excerpts),
                "source_text_characters": sum(len(excerpt.markdown) for excerpt in excerpts),
                "available_markdown_characters": sum(
                    len(document.markdown) for document in documents
                ),
                "block_count": len(excerpts),
                "document_ids": [document.document_id for document in documents],
                "compact_pages": [],
                "full_context_pages": [],
            },
            policy_name="document-chat-system.md",
        )
        return parsed, usage

    def refine(
        self,
        pages: list[PageParse],
        image_pages: set[int],
        capabilities: set[str],
        allowed_classes: list[str],
        extraction_schema: dict[str, Any] | None,
    ) -> tuple[CloudResult, UsageRecord]:
        full_context_pages = image_pages
        aggregate = CloudResult(refined_markdown="", reviewed_pages=[])
        call_usages: list[UsageRecord] = []
        batches = self._batches(
            pages,
            full_context_pages,
            capabilities,
            allowed_classes,
            extraction_schema,
        )
        for batch_index, planned_batch in enumerate(batches, start=1):
            batch = planned_batch.pages
            packet = self._prompt_packet(
                batch,
                full_context_pages,
                capabilities,
                allowed_classes,
                extraction_schema,
            )
            content: list[dict[str, Any]] = [
                {
                    "type": "input_text",
                    "text": packet.text,
                }
            ]
            visual_resources: list[PromptResource] = []
            overview_pages: list[int] = []
            high_resolution_regions: list[dict[str, Any]] = []
            for page in batch:
                regions = page.visual_review_regions
                page_image = self._bounded_image(page.original_image_bytes or page.image_bytes)
                visual_label, visual_resource = render_prompt(
                    "visual-page.md", page_number=page.page
                )
                content.extend(
                    [
                        {"type": "input_text", "text": visual_label},
                        {
                            "type": "input_image",
                            "image_url": "data:image/jpeg;base64,"
                            + base64.b64encode(page_image).decode("ascii"),
                            "detail": "high",
                        },
                    ]
                )
                visual_resources.append(visual_resource)
                overview_pages.append(page.page)
                for region in regions:
                    if region.page_wide:
                        continue
                    region_label, region_resource = render_prompt(
                        "visual-region.md",
                        region_id=region.id,
                        page_number=page.page,
                        region_bbox=_prompt_json(region.bbox),
                        reason_codes=_prompt_json(region.reason_codes),
                        source_block_ids=_prompt_json(region.source_block_ids),
                        source_checkbox_ids=_prompt_json(region.source_checkbox_ids),
                        source_redaction_ids=_prompt_json(region.source_redaction_ids),
                    )
                    crop = self._region_crop(
                        page.original_image_bytes or page.image_bytes, region.bbox
                    )
                    content.extend(
                        [
                            {"type": "input_text", "text": region_label},
                            {
                                "type": "input_image",
                                "image_url": "data:image/jpeg;base64,"
                                + base64.b64encode(crop).decode("ascii"),
                                "detail": "high",
                            },
                        ]
                    )
                    visual_resources.append(region_resource)
                    high_resolution_regions.append(region.model_dump(mode="json"))
            started = time.perf_counter()
            try:
                response = self._request(content)
            except Exception as exc:
                page_numbers = [page.page for page in batch]
                raise RuntimeError(
                    f"GPT-5.6-luna {planned_batch.kind} batch "
                    f"{batch_index}/{len(batches)} for pages {page_numbers} failed: {exc}"
                ) from exc
            elapsed = time.perf_counter() - started
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("OpenAI returned no structured result.")
            expected_pages = [page.page for page in batch]
            if sorted(parsed.reviewed_pages) != expected_pages:
                raise RuntimeError("GPT-5.6-luna must review every requested page exactly once.")
            redaction_pages = {
                candidate.id: page.page
                for page in batch
                for candidate in page.local_redaction_candidates
            }
            expected_redactions = sorted(redaction_pages)
            redaction_ids = set(expected_redactions)
            reviewed_redactions = [
                refinement.block_id
                for refinement in parsed.refinements
                if refinement.block_id in redaction_ids
            ]
            if len(reviewed_redactions) != len(set(reviewed_redactions)):
                raise RuntimeError(
                    "GPT-5.6-luna returned duplicate outcomes for a local redaction candidate."
                )
            missing_redactions = sorted(redaction_ids - set(reviewed_redactions))
            for candidate_id in missing_redactions:
                parsed.refinements.append(
                    CloudRefinement(
                        page=redaction_pages[candidate_id],
                        block_id=candidate_id,
                        abstained=True,
                        warning="GPT-5.6-luna omitted this local redaction candidate.",
                    )
                )
            if missing_redactions:
                parsed.warnings.append(
                    "GPT-5.6-luna omitted local redaction candidates; they were recorded as "
                    "abstentions and were not published: " + ", ".join(missing_redactions)
                )
            separator = "\n\n" if aggregate.refined_markdown else ""
            aggregate.refined_markdown += separator + parsed.refined_markdown
            aggregate.reviewed_pages.extend(parsed.reviewed_pages)
            aggregate.refinements.extend(parsed.refinements)
            aggregate.warnings.extend(parsed.warnings)
            aggregate.classifications.extend(parsed.classifications)
            aggregate.sections.extend(parsed.sections)
            aggregate.splits.extend(parsed.splits)
            aggregate.extracted_fields.extend(parsed.extracted_fields)
            aggregate.checkboxes.extend(parsed.checkboxes)
            aggregate.table_reviews.extend(parsed.table_reviews)
            aggregate.semantic_regions.extend(parsed.semantic_regions)
            resources = list(packet.resources)
            resources.extend(visual_resources)
            context = dict(packet.metrics)
            context["prompt_characters"] = sum(
                len(item["text"]) for item in content if item["type"] == "input_text"
            )
            context["overview_pages"] = overview_pages
            context["high_resolution_regions"] = high_resolution_regions
            context["high_resolution_region_count"] = len(high_resolution_regions)
            context["batch_kind"] = planned_batch.kind
            context["batch_index"] = batch_index
            context["batch_count"] = len(batches)
            call = self._read_usage(
                response,
                batch,
                {page.page for page in batch},
                elapsed,
                resources,
                purpose="refinement",
                context=context,
            )
            call_usages.append(call)
        _add_semantic_table_candidates(pages, aggregate.semantic_regions)
        if any(page.table_structures for page in pages):
            table_result, table_usage = self.review_tables(pages)
            table_pages = {review.page for review in table_result.reviews}
            aggregate.table_reviews = [
                review for review in aggregate.table_reviews if review.page not in table_pages
            ]
            aggregate.table_reviews.extend(table_result.reviews)
            call_usages.append(table_usage)
        expected_pages = [page.page for page in pages]
        if sorted(aggregate.reviewed_pages) != sorted(expected_pages):
            raise RuntimeError("GPT-5.6-luna must review every requested page exactly once.")
        aggregate.reviewed_pages = expected_pages
        if len(batches) > 1 and capabilities - {"Parse"}:
            reconciled, usage = self._reconcile(
                pages, capabilities, allowed_classes, extraction_schema, aggregate
            )
            aggregate.classifications = reconciled.classifications
            aggregate.sections = reconciled.sections
            aggregate.splits = reconciled.splits
            aggregate.extracted_fields = reconciled.extracted_fields
            aggregate.warnings.extend(reconciled.warnings)
            call_usages.append(usage)
        return aggregate, aggregate_usage(call_usages)

    def review_tables(self, pages: list[PageParse]) -> tuple[TableReviewResult, UsageRecord]:
        """Run a bounded visual review covering every local table candidate exactly once."""
        page_by_number = {page.page: page for page in pages}
        candidates: list[dict[str, Any]] = []
        expected: dict[str, int] = {}
        for page in pages:
            for table in page.table_structures:
                source_ids = table_review_block_ids(page, table)
                review_bbox = table_review_bbox(page, table)
                source_blocks = [block for block in page.blocks if block.id in source_ids]
                expected[table.id] = page.page
                candidates.append(
                    {
                        "table": table.model_dump(mode="json"),
                        "review_bbox": review_bbox,
                        "expected_source_blocks": [
                            {
                                "id": block.id,
                                "text": block.text,
                                "bbox": block.bbox,
                                "ocr_score": block.ocr_score,
                                "words": table_word_evidence(page, {block.id}),
                            }
                            for block in source_blocks
                        ],
                    }
                )
        if not candidates:
            return TableReviewResult(), UsageRecord()
        review_page_numbers = set(expected.values())
        review_pages = [page for page in pages if page.page in review_page_numbers]
        evidence = _prompt_json(candidates)
        prompt, prompt_resource = render_prompt("table-review.md", table_candidates=evidence)
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        visual_resources: list[PromptResource] = []
        for page in review_pages:
            label, resource = render_prompt("visual-page.md", page_number=page.page)
            overview = self._bounded_image(page.image_bytes, max_dimension=512, quality=75)
            content.extend(
                [
                    {"type": "input_text", "text": label},
                    {
                        "type": "input_image",
                        "image_url": "data:image/jpeg;base64,"
                        + base64.b64encode(overview).decode("ascii"),
                        "detail": "low",
                    },
                ]
            )
            visual_resources.append(resource)
        for candidate in candidates:
            table = candidate["table"]
            page = page_by_number[int(table["page"])]
            review_bbox = candidate["review_bbox"]
            label, resource = render_prompt(
                "visual-region.md",
                region_id=table["id"],
                page_number=table["page"],
                region_bbox=_prompt_json(review_bbox),
                reason_codes=_prompt_json(["table_semantic_validation", "detector_edge_recovery"]),
                source_block_ids=_prompt_json(
                    [item["id"] for item in candidate["expected_source_blocks"]]
                ),
                source_checkbox_ids="[]",
                source_redaction_ids="[]",
            )
            crop = self._region_crop(page.original_image_bytes or page.image_bytes, review_bbox)
            content.extend(
                [
                    {"type": "input_text", "text": label},
                    {
                        "type": "input_image",
                        "image_url": "data:image/jpeg;base64,"
                        + base64.b64encode(crop).decode("ascii"),
                        "detail": "high",
                    },
                ]
            )
            visual_resources.append(resource)
        started = time.perf_counter()
        for attempt in range(2):
            response = self._request(content, TableReviewResult)
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("OpenAI returned no structured table-review result.")
            actual = [review.table_id for review in parsed.reviews]
            coverage_is_exact = len(actual) == len(set(actual)) and set(actual) == set(expected)
            if coverage_is_exact:
                break
            if attempt:
                raise RuntimeError("GPT-5.6-luna must review every table candidate exactly once.")
        elapsed = time.perf_counter() - started
        if any(review.page != expected[review.table_id] for review in parsed.reviews):
            raise RuntimeError("GPT-5.6-luna table review referenced an unexpected page.")
        usage = self._read_usage(
            response,
            review_pages,
            review_page_numbers,
            elapsed,
            [prompt_resource, *visual_resources],
            purpose="table_review",
            context={
                "kind": "table_review",
                "prompt_characters": len(prompt),
                "evidence_characters": len(evidence),
                "source_text_characters": sum(
                    len(item["text"])
                    for candidate in candidates
                    for item in candidate["expected_source_blocks"]
                ),
                "block_count": sum(
                    len(candidate["expected_source_blocks"]) for candidate in candidates
                ),
                "table_ids": list(expected),
                "compact_pages": [],
                "full_context_pages": [page.page for page in review_pages],
            },
        )
        return parsed, usage

    def refine_markdown(
        self,
        markdown: str,
        pages: list[PageParse],
        capabilities: set[str],
        allowed_classes: list[str],
        extraction_schema: dict[str, Any] | None,
        *,
        issues: list[dict[str, Any]] | None = None,
        prior: CloudResult | None = None,
    ) -> tuple[CloudResult, UsageRecord]:
        """Run optional workflows from canonical Markdown without page images."""
        requested = capabilities - {"Parse"}
        if not requested:
            return CloudResult(refined_markdown=markdown, reviewed_pages=[]), UsageRecord()
        capability_instructions, capability_resources = self._capability_instructions(requested)
        grounding = []
        for page in pages:
            chunks = page.chunks or build_layout_chunks(page.blocks)
            for block in page.blocks:
                grounding.append(
                    [
                        page.page,
                        block.id,
                        next(
                            (chunk.id for chunk in chunks if block.id in chunk.source_block_ids),
                            None,
                        ),
                        block.type,
                        block.ocr_score,
                        block.bbox,
                        block.text,
                    ]
                )
        grounding_json = _prompt_json(grounding)
        issues_json = _prompt_json(issues or [])
        target_ids = sorted(
            {
                str(identifier)
                for issue in issues or []
                for identifier in issue.get("source_ids", [])
            }
        )
        if issues and not target_ids:
            raise ValueError("Object repair requires at least one identified validation object.")
        prior_json = _prompt_json(prior.model_dump(mode="json") if prior else None)
        prompt, prompt_resource = render_prompt(
            "markdown-workflow.md",
            capabilities=_prompt_json(sorted(requested)),
            allowed_classes=_prompt_json([*allowed_classes, "unknown"]),
            extraction_schema=_prompt_json(extraction_schema),
            capability_instructions=capability_instructions,
            validation_issues=issues_json,
            prior_proposal=prior_json,
            grounding_columns=_prompt_json(_GROUNDING_COLUMNS),
            grounding_rows=grounding_json,
            document_markdown=markdown,
        )
        started = time.perf_counter()
        response = self._request([{"type": "input_text", "text": prompt}], MarkdownWorkflowResult)
        elapsed = time.perf_counter() - started
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no structured Markdown workflow result.")
        expected_pages = sorted(page.page for page in pages)
        if sorted(parsed.reviewed_pages) != expected_pages:
            raise RuntimeError("GPT-5.6-luna Markdown workflow must cover every selected page.")
        cloud = CloudResult(
            refined_markdown=markdown,
            reviewed_pages=parsed.reviewed_pages,
            warnings=parsed.warnings,
            classifications=parsed.classifications,
            sections=parsed.sections,
            splits=parsed.splits,
            extracted_fields=parsed.extracted_fields,
        )
        usage = self._read_usage(
            response,
            pages,
            set(),
            elapsed,
            [prompt_resource, *capability_resources],
            purpose="object_repair" if issues else "markdown_workflow",
            context={
                "kind": "object_repair" if issues else "grounded_markdown",
                "prompt_characters": len(prompt),
                "evidence_characters": (
                    len(markdown) + len(grounding_json) + len(issues_json) + len(prior_json)
                ),
                "source_text_characters": sum(
                    len(block.text) for page in pages for block in page.blocks
                ),
                "block_count": len(grounding),
                "compact_pages": [],
                "full_context_pages": [],
                "target_ids": target_ids,
            },
        )
        return cloud, usage

    def verify_checkboxes(
        self, pages: list[PageParse], candidates: list[CloudCheckbox]
    ) -> tuple[CheckboxVerificationResult, UsageRecord]:
        """Run one independent crop-based verification call for risky controls."""
        if not candidates:
            return CheckboxVerificationResult(), UsageRecord()
        page_map = {page.page: page for page in pages}
        evidence = [candidate.model_dump(mode="json") for candidate in candidates]
        prompt, prompt_resource = render_prompt(
            "checkbox-verification.md",
            checkbox_evidence=_prompt_json(evidence),
        )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        visual_resource = load_prompt("checkbox-crop.md")
        for candidate in candidates:
            page = page_map.get(candidate.page)
            if page is None:
                continue
            label, _ = render_prompt(
                "checkbox-crop.md",
                page_number=candidate.page,
                checkbox_id=candidate.id,
            )
            content.extend(
                [
                    {"type": "input_text", "text": label},
                    {
                        "type": "input_image",
                        "image_url": "data:image/jpeg;base64,"
                        + base64.b64encode(
                            self._checkbox_crop(
                                page.original_image_bytes or page.image_bytes,
                                candidate.control_bbox,
                            )
                        ).decode("ascii"),
                        "detail": "high",
                    },
                ]
            )
        started = time.perf_counter()
        response = self._request(content, text_format=CheckboxVerificationResult)
        elapsed = time.perf_counter() - started
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no structured checkbox verification result.")
        expected = sorted(candidate.id for candidate in candidates)
        actual = sorted(item.id for item in parsed.verifications)
        if actual != expected:
            raise RuntimeError("GPT-5.6-luna must verify every requested checkbox exactly once.")
        verification_pages = [
            page_map[number]
            for number in dict.fromkeys(candidate.page for candidate in candidates)
            if number in page_map
        ]
        usage = self._read_usage(
            response,
            verification_pages,
            {candidate.page for candidate in candidates},
            elapsed,
            [prompt_resource, visual_resource],
            purpose="checkbox_verification",
            checkbox_ids=expected,
            context={
                "kind": "checkbox_verification",
                "prompt_characters": sum(
                    len(item["text"]) for item in content if item["type"] == "input_text"
                ),
                "evidence_characters": len(_prompt_json(evidence)),
                "source_text_characters": sum(
                    len(item.quote) for candidate in candidates for item in candidate.evidence
                ),
                "block_count": len(candidates),
                "compact_pages": [],
                "full_context_pages": [],
            },
        )
        return parsed, usage

    def _reconcile(
        self,
        pages: list[PageParse],
        capabilities: set[str],
        allowed_classes: list[str],
        extraction_schema: dict[str, Any] | None,
        candidates: CloudResult,
    ) -> tuple[CloudResult, UsageRecord]:
        summaries: list[str] = []
        context_resources: list[PromptResource] = []
        represented_blocks = 0
        represented_source_characters = 0
        for page in pages:
            blocks = page.blocks[:2] + page.blocks[-2:] if len(page.blocks) > 4 else page.blocks
            represented_blocks += len(blocks)
            represented_source_characters += sum(len(block.text) for block in blocks)
            block_summaries: list[str] = []
            for block in blocks:
                summary, resource = render_prompt(
                    "reconciliation-block.md",
                    block_json=_prompt_json({"id": block.id, "text": block.text[:500]}),
                )
                block_summaries.append(summary)
                context_resources.append(resource)
            summary, resource = render_prompt(
                "reconciliation-page.md",
                page_number=page.page,
                blocks="\n".join(block_summaries),
            )
            summaries.append(summary)
            context_resources.append(resource)
        capability_instructions, capability_resources = self._capability_instructions(
            capabilities - {"Parse"}
        )
        candidates_json = _prompt_json(candidates.model_dump(mode="json"))
        summaries_text = "\n".join(summaries)
        prompt, prompt_resource = render_prompt(
            "reconciliation.md",
            capabilities=_prompt_json(sorted(capabilities)),
            allowed_classes=_prompt_json([*allowed_classes, "unknown"]),
            extraction_schema=_prompt_json(extraction_schema),
            capability_instructions=capability_instructions,
            batch_candidates=candidates_json,
            document_summaries=summaries_text,
        )
        started = time.perf_counter()
        response = self._request([{"type": "input_text", "text": prompt}])
        elapsed = time.perf_counter() - started
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no structured reconciliation result.")
        expected_pages = sorted(page.page for page in pages)
        if sorted(parsed.reviewed_pages) != expected_pages:
            raise RuntimeError("GPT-5.6-luna reconciliation must cover every selected page.")
        return parsed, self._read_usage(
            response,
            pages,
            set(),
            elapsed,
            [prompt_resource, *capability_resources, *context_resources],
            purpose="reconciliation",
            context={
                "kind": "reconciliation",
                "prompt_characters": len(prompt),
                "evidence_characters": len(candidates_json) + len(summaries_text),
                "source_text_characters": represented_source_characters,
                "block_count": represented_blocks,
                "compact_pages": [],
                "full_context_pages": [],
            },
        )

    def repair(
        self,
        pages: list[PageParse],
        image_pages: set[int],
        capabilities: set[str],
        allowed_classes: list[str],
        extraction_schema: dict[str, Any] | None,
        issues: list[dict[str, Any]],
        prior: CloudResult,
    ) -> tuple[CloudResult, UsageRecord]:
        """Run one application-bounded correction pass over unresolved proposals."""
        packet = self._prompt_packet(
            pages, image_pages, capabilities, allowed_classes, extraction_schema
        )
        issues_json = _prompt_json(issues)
        prior_json = _prompt_json(prior.model_dump(mode="json"))
        prompt, repair_resource = render_prompt(
            "repair.md",
            base_prompt=packet.text,
            validation_issues=issues_json,
            prior_proposal=prior_json,
        )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        visual_resources: list[PromptResource] = []
        for page in pages:
            if page.page not in image_pages:
                continue
            if not any(region.page_wide for region in page.visual_review_regions):
                visual_label, visual_resource = render_prompt(
                    "visual-page.md", page_number=page.page
                )
                overview = self._bounded_image(page.image_bytes, max_dimension=512, quality=75)
                content.extend(
                    [
                        {"type": "input_text", "text": visual_label},
                        {
                            "type": "input_image",
                            "image_url": "data:image/jpeg;base64,"
                            + base64.b64encode(overview).decode("ascii"),
                            "detail": "low",
                        },
                    ]
                )
                visual_resources.append(visual_resource)
            for region in page.visual_review_regions:
                region_label, region_resource = render_prompt(
                    "visual-region.md",
                    region_id=region.id,
                    page_number=page.page,
                    region_bbox=_prompt_json(region.bbox),
                    reason_codes=_prompt_json(region.reason_codes),
                    source_block_ids=_prompt_json(region.source_block_ids),
                    source_checkbox_ids=_prompt_json(region.source_checkbox_ids),
                    source_redaction_ids=_prompt_json(region.source_redaction_ids),
                )
                crop = self._region_crop(page.original_image_bytes or page.image_bytes, region.bbox)
                content.extend(
                    [
                        {"type": "input_text", "text": region_label},
                        {
                            "type": "input_image",
                            "image_url": "data:image/jpeg;base64,"
                            + base64.b64encode(crop).decode("ascii"),
                            "detail": "high",
                        },
                    ]
                )
                visual_resources.append(region_resource)
        started = time.perf_counter()
        response = self._request(content)
        elapsed = time.perf_counter() - started
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no structured repair result.")
        expected_pages = sorted(page.page for page in pages)
        if sorted(parsed.reviewed_pages) != expected_pages:
            raise RuntimeError("GPT-5.6-luna repair must review every requested page exactly once.")
        resources = [*packet.resources, repair_resource]
        resources.extend(visual_resources)
        context = dict(packet.metrics)
        context["kind"] = "repair"
        context["prompt_characters"] = sum(
            len(item["text"]) for item in content if item["type"] == "input_text"
        )
        context["evidence_characters"] += len(issues_json) + len(prior_json)
        context["overview_pages"] = [
            page.page
            for page in pages
            if page.page in image_pages
            and not any(region.page_wide for region in page.visual_review_regions)
        ]
        context["high_resolution_regions"] = [
            region.model_dump(mode="json")
            for page in pages
            if page.page in image_pages
            for region in page.visual_review_regions
        ]
        context["high_resolution_region_count"] = len(context["high_resolution_regions"])
        return parsed, self._read_usage(
            response,
            pages,
            image_pages,
            elapsed,
            resources,
            purpose="repair",
            context=context,
        )

    def _batches(
        self,
        pages: list[PageParse],
        full_context_pages: set[int],
        capabilities: set[str],
        allowed_classes: list[str],
        extraction_schema: dict[str, Any] | None,
    ) -> list[_RefinementBatch]:
        compact_pages = [page for page in pages if page.page not in full_context_pages]
        compact_page_numbers = {page.page for page in compact_pages}
        full_pages = [page for page in pages if page.page not in compact_page_numbers]
        batches: list[_RefinementBatch] = []
        current: list[PageParse] = []
        for page in compact_pages:
            candidate = [*current, page]
            candidate_size = self._prompt_packet(
                candidate,
                full_context_pages,
                capabilities,
                allowed_classes,
                extraction_schema,
            ).metrics["evidence_characters"]
            if current and candidate_size > self.settings.cloud_batch_characters:
                batches.append(_RefinementBatch(kind="compact", pages=current))
                current = [page]
            else:
                current = candidate
        if current:
            batches.append(_RefinementBatch(kind="compact", pages=current))
        batches.extend(_RefinementBatch(kind="full", pages=[page]) for page in full_pages)
        return batches

    @staticmethod
    def _bounded_image(data: bytes, max_dimension: int = 2048, quality: int = 85) -> bytes:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.thumbnail((max_dimension, max_dimension))
        output = io.BytesIO()
        image.save(output, "JPEG", quality=quality, optimize=True)
        return output.getvalue()

    @staticmethod
    def _region_crop(data: bytes, bbox: list[float]) -> bytes:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        left, top, right, bottom = bbox
        x0 = min(image.width - 1, max(0, int(left * image.width)))
        y0 = min(image.height - 1, max(0, int(top * image.height)))
        x1 = min(image.width, max(x0 + 1, int(right * image.width)))
        y1 = min(image.height, max(y0 + 1, int(bottom * image.height)))
        crop = image.crop((x0, y0, x1, y1))
        output = io.BytesIO()
        crop.thumbnail((2048, 2048))
        crop.save(output, "JPEG", quality=90, optimize=True)
        return output.getvalue()

    @staticmethod
    def _checkbox_crop(data: bytes, bbox: list[float]) -> bytes:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        left, top, right, bottom = bbox
        width = max(right - left, 0.01)
        height = max(bottom - top, 0.01)
        pad_x = max(width * 2, 0.02)
        pad_y = max(height * 2, 0.02)
        crop = image.crop(
            (
                max(0, int((left - pad_x) * image.width)),
                max(0, int((top - pad_y) * image.height)),
                min(image.width, int((right + pad_x) * image.width)),
                min(image.height, int((bottom + pad_y) * image.height)),
            )
        )
        longest_side = max(crop.size)
        if longest_side < 512:
            scale = 512 / longest_side
            crop = crop.resize(
                (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                Image.Resampling.LANCZOS,
            )
        output = io.BytesIO()
        crop.save(output, "JPEG", quality=95, optimize=True)
        return output.getvalue()

    def _request(
        self,
        content: list[dict[str, Any]],
        text_format: type[BaseModel] = CloudResult,
        *,
        policy_name: Literal["policy.md", "document-chat-system.md"] = "policy.md",
    ) -> Any:
        policy = load_prompt(policy_name)
        request = {
            "model": MODEL_NAME,
            "reasoning": {"effort": REASONING_EFFORT},
            "store": False,
            "tools": [],
            "instructions": policy.text,
            "input": cast(Any, [{"role": "user", "content": content}]),
            "text_format": text_format,
        }
        try:
            return self.client.responses.parse(**request)
        except ValidationError:
            # The SDK can receive an otherwise successful response whose JSON
            # text was cut off before Pydantic parsing. Retry that idempotent,
            # non-stored request once; a second invalid response still fails.
            return self.client.responses.parse(**request)

    @staticmethod
    def _read_usage(
        response: Any,
        pages: list[PageParse],
        image_pages: set[int],
        elapsed: float,
        prompt_resources: list[PromptResource] | None = None,
        purpose: str = "refinement",
        checkbox_ids: list[str] | None = None,
        context: dict[str, Any] | None = None,
        policy_name: Literal["policy.md", "document-chat-system.md"] = "policy.md",
    ) -> UsageRecord:
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        token_usage = TokenUsage(
            input_tokens=input_tokens,
            cached_input_tokens=getattr(input_details, "cached_tokens", None),
            cache_write_input_tokens=getattr(input_details, "cache_write_tokens", None),
            output_tokens=output_tokens,
            reasoning_tokens=getattr(output_details, "reasoning_tokens", None),
            total_tokens=total_tokens,
        )
        cost = calculate_usage_cost(token_usage)
        page_numbers = [page.page for page in pages]
        call = usage_and_cost_dict(token_usage)
        call["pages"] = page_numbers
        call["image_pages"] = [page for page in page_numbers if page in image_pages]
        call["model"] = MODEL_NAME
        call["reasoning_effort"] = REASONING_EFFORT
        call["elapsed_seconds"] = elapsed
        call["purpose"] = purpose
        resources = [load_prompt(policy_name), *(prompt_resources or [])]
        call["prompts"] = [
            {"name": item.name, "version": item.version, "sha256": item.sha256}
            for item in dict.fromkeys(resources)
        ]
        call["checkbox_ids"] = checkbox_ids or []
        if context is not None:
            call["context"] = context
        page_count = len(page_numbers)
        call["per_page_usage_estimate"] = {
            "input_tokens": (
                input_tokens / page_count if input_tokens is not None and page_count else None
            ),
            "cached_input_tokens": (
                token_usage.cached_input_tokens / page_count
                if token_usage.cached_input_tokens is not None and page_count
                else None
            ),
            "cache_write_input_tokens": (
                token_usage.cache_write_input_tokens / page_count
                if token_usage.cache_write_input_tokens is not None and page_count
                else None
            ),
            "output_tokens": (
                output_tokens / page_count if output_tokens is not None and page_count else None
            ),
            "total_tokens": (
                total_tokens / page_count if total_tokens is not None and page_count else None
            ),
        }
        call["per_page_usage_basis"] = "equal allocation across pages in this API call"
        call["per_page_cost_usd_estimate"] = (
            cost.total_cost_usd / len(page_numbers)
            if cost.total_cost_usd is not None and page_numbers
            else None
        )
        call["per_page_cost_basis"] = "equal allocation across pages in this API call"
        return UsageRecord(
            call_count=1,
            input_tokens=input_tokens,
            cached_input_tokens=token_usage.cached_input_tokens,
            cache_write_input_tokens=token_usage.cache_write_input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=token_usage.reasoning_tokens,
            total_tokens=total_tokens,
            cloud_seconds=elapsed,
            input_cost_usd=cost.input_cost_usd,
            output_cost_usd=cost.output_cost_usd,
            total_cost_usd=cost.total_cost_usd,
            cost_status=cost.status,
            calls=[call],
        )

    def _prompt(
        self,
        pages: list[PageParse],
        full_context_pages: set[int],
        capabilities: set[str],
        allowed_classes: list[str],
        extraction_schema: dict[str, Any] | None,
    ) -> tuple[str, list[PromptResource]]:
        packet = self._prompt_packet(
            pages,
            full_context_pages,
            capabilities,
            allowed_classes,
            extraction_schema,
        )
        return packet.text, packet.resources

    def _prompt_packet(
        self,
        pages: list[PageParse],
        full_context_pages: set[int],
        capabilities: set[str],
        allowed_classes: list[str],
        extraction_schema: dict[str, Any] | None,
    ) -> _ContextPacket:
        lines: list[str] = []
        context_resources: list[PromptResource] = []
        capability_instructions, capability_resources = self._capability_instructions(capabilities)
        local_checkbox_evidence = [
            item.model_dump(mode="json")
            for page in pages
            for item in page.local_checkbox_candidates
            if is_credible_checkbox_candidate(page, item)
        ]
        checkbox_candidate_pages = sorted({item["page"] for item in local_checkbox_evidence})
        checkbox_instructions = ""
        checkbox_resources: list[PromptResource] = []
        if local_checkbox_evidence:
            checkbox_instructions, checkbox_resource = render_prompt(
                "checkbox-discovery.md",
                candidate_pages=_prompt_json(checkbox_candidate_pages),
                local_checkbox_evidence=_prompt_json(local_checkbox_evidence),
            )
            checkbox_resources.append(checkbox_resource)
        semantic_region_count = 0
        for page in pages:
            full_context = page.page in full_context_pages
            semantic_regions = _local_semantic_regions(page)
            semantic_region_count += len(semantic_regions)
            blocks: list[str] = []
            for block in page.blocks:
                block_context, resource = self._block_context(block, full_context)
                blocks.append(block_context)
                context_resources.append(resource)
            template = "page-context-full.md" if full_context else "page-context-compact.md"
            layout_regions = [
                region.model_dump(mode="json")
                if full_context
                else region.model_dump(mode="json", exclude={"coordinate", "polygon", "raw_order"})
                for region in page.layout_regions
            ]
            page_context, resource = render_prompt(
                template,
                page_number=page.page,
                page_metadata=_prompt_json(
                    {
                        "layout_signals": page.layout_signals,
                        "status": page.status,
                        "warnings": page.warnings,
                    }
                ),
                semantic_region_columns=_prompt_json(_LOCAL_SEMANTIC_REGION_COLUMNS),
                semantic_regions=_prompt_json(semantic_regions),
                block_columns=_prompt_json(_BLOCK_EVIDENCE_COLUMNS),
                blocks="\n".join(blocks),
                layout_regions=_prompt_json(layout_regions),
                layout_links=_prompt_json(
                    [link.model_dump(mode="json") for link in page.layout_block_links]
                ),
                reading_order=_prompt_json(
                    page.reading_order_evidence.model_dump(mode="json")
                    if page.reading_order_evidence
                    else None
                ),
                table_structures=_prompt_json(
                    [
                        table.model_dump(
                            mode="json",
                            exclude=({"cells", "markdown"} if table.status != "valid" else None),
                        )
                        for table in page.table_structures
                    ]
                ),
            )
            lines.append(page_context)
            context_resources.append(resource)
        ocr = "\n".join(lines)
        rendered, resource = render_prompt(
            "refinement.md",
            capabilities=_prompt_json(sorted(capabilities)),
            allowed_classes=_prompt_json([*allowed_classes, "unknown"]),
            extraction_schema=_prompt_json(extraction_schema),
            capability_instructions=capability_instructions,
            checkbox_instructions=checkbox_instructions,
            document_context=ocr,
        )
        return _ContextPacket(
            text=rendered,
            resources=[
                resource,
                *capability_resources,
                *checkbox_resources,
                *context_resources,
            ],
            metrics={
                "kind": "parse",
                "prompt_characters": len(rendered),
                "evidence_characters": len(ocr),
                "source_text_characters": sum(
                    len(block.text) for page in pages for block in page.blocks
                ),
                "block_count": sum(len(page.blocks) for page in pages),
                "semantic_region_count": semantic_region_count,
                "checkbox_candidate_pages": checkbox_candidate_pages,
                "checkbox_candidate_count": len(local_checkbox_evidence),
                "compact_pages": [
                    page.page for page in pages if page.page not in full_context_pages
                ],
                "full_context_pages": [
                    page.page for page in pages if page.page in full_context_pages
                ],
            },
        )

    @staticmethod
    def _capability_instructions(
        capabilities: set[str],
    ) -> tuple[str, list[PromptResource]]:
        rendered: list[str] = []
        resources: list[PromptResource] = []
        for capability, name in _CAPABILITY_PROMPTS.items():
            if capability not in capabilities:
                continue
            text, resource = render_prompt(name)
            rendered.append(text)
            resources.append(resource)
        return "\n\n".join(rendered), resources

    @staticmethod
    def _block_context(block: Any, full_context: bool) -> tuple[str, PromptResource]:
        requires_gpt_review = block_requires_gpt_review(block)
        return render_prompt(
            "block-context-full.md" if full_context else "block-context-compact.md",
            block_json=_prompt_json(
                [
                    block.id,
                    block.type,
                    block.text,
                    block.ocr_score,
                    block.bbox,
                    block.polygon,
                    requires_gpt_review,
                ]
            ),
        )
