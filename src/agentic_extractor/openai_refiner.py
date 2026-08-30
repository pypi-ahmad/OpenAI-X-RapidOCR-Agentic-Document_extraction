"""The only OpenAI boundary. Model policy is intentionally not caller-configurable."""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Annotated, Any, Literal, cast

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field, WithJsonSchema

from agentic_extractor.config import SETTINGS, Settings
from agentic_extractor.costs import (
    MODEL_NAME,
    REASONING_EFFORT,
    TokenUsage,
    aggregate_usage,
    calculate_usage_cost,
    usage_and_cost_dict,
)
from agentic_extractor.models import UsageRecord
from agentic_extractor.parse import LOW_CONFIDENCE_THRESHOLD, PageParse, build_layout_chunks
from agentic_extractor.prompt_resources import PromptResource, load_prompt, render_prompt

_CAPABILITY_PROMPTS = {
    "Parse": "capability-parse.md",
    "Classify": "capability-classify.md",
    "Section": "capability-section.md",
    "Split": "capability-split.md",
    "Extract": "capability-extract.md",
}

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
    block_id: str | None = None
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


class CheckboxVerification(BaseModel):
    id: str
    page: int = Field(ge=1)
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

    def refine(
        self,
        pages: list[PageParse],
        image_pages: set[int],
        capabilities: set[str],
        allowed_classes: list[str],
        extraction_schema: dict[str, Any] | None,
    ) -> tuple[CloudResult, UsageRecord]:
        aggregate = CloudResult(refined_markdown="", reviewed_pages=[])
        call_usages: list[UsageRecord] = []
        batches = self._batches(pages)
        for batch in batches:
            prompt, prompt_resources = self._prompt(
                batch,
                image_pages,
                capabilities,
                allowed_classes,
                extraction_schema,
            )
            content: list[dict[str, Any]] = [
                {
                    "type": "input_text",
                    "text": prompt,
                }
            ]
            for page in batch:
                encoded = base64.b64encode(self._bounded_image(page.image_bytes)).decode("ascii")
                visual_label, visual_resource = render_prompt(
                    "visual-page.md", page_number=page.page
                )
                content.extend(
                    [
                        {"type": "input_text", "text": visual_label},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{encoded}",
                            "detail": "high",
                        },
                    ]
                )
            started = time.perf_counter()
            response = self._request(content)
            elapsed = time.perf_counter() - started
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("OpenAI returned no structured result.")
            expected_pages = [page.page for page in batch]
            if sorted(parsed.reviewed_pages) != expected_pages:
                raise RuntimeError("GPT-5.6-luna must review every requested page exactly once.")
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
            resources = list(prompt_resources)
            resources.append(visual_resource)
            call = self._read_usage(
                response,
                batch,
                {page.page for page in batch},
                elapsed,
                resources,
                purpose="refinement",
            )
            call_usages.append(call)
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
        grounding = [
            {
                "page": page.page,
                "blocks": [
                    {
                        "id": block.id,
                        "chunk_id": next(
                            (
                                chunk.id
                                for chunk in (page.chunks or build_layout_chunks(page.blocks))
                                if block.id in chunk.source_block_ids
                            ),
                            None,
                        ),
                        "bbox": block.bbox,
                        "confidence": block.ocr_score,
                        "text": block.text,
                        "type": block.type,
                    }
                    for block in page.blocks
                ],
            }
            for page in pages
        ]
        prompt, prompt_resource = render_prompt(
            "markdown-workflow.md",
            capabilities=_prompt_json(sorted(requested)),
            allowed_classes=_prompt_json([*allowed_classes, "unknown"]),
            extraction_schema=_prompt_json(extraction_schema),
            capability_instructions=capability_instructions,
            validation_issues=_prompt_json(issues or []),
            prior_proposal=_prompt_json(prior.model_dump(mode="json") if prior else None),
            grounding_index=_prompt_json(grounding),
            document_markdown=markdown,
        )
        started = time.perf_counter()
        response = self._request(
            [{"type": "input_text", "text": prompt}], MarkdownWorkflowResult
        )
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
            purpose="markdown_workflow_repair" if issues else "markdown_workflow",
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
        for page in pages:
            blocks = page.blocks[:2] + page.blocks[-2:] if len(page.blocks) > 4 else page.blocks
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
        prompt, prompt_resource = render_prompt(
            "reconciliation.md",
            capabilities=_prompt_json(sorted(capabilities)),
            allowed_classes=_prompt_json([*allowed_classes, "unknown"]),
            extraction_schema=_prompt_json(extraction_schema),
            capability_instructions=capability_instructions,
            batch_candidates=_prompt_json(candidates.model_dump(mode="json")),
            document_summaries="\n".join(summaries),
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
        base_prompt, base_resources = self._prompt(
            pages, image_pages, capabilities, allowed_classes, extraction_schema
        )
        prompt, repair_resource = render_prompt(
            "repair.md",
            base_prompt=base_prompt,
            validation_issues=_prompt_json(issues),
            prior_proposal=_prompt_json(prior.model_dump(mode="json")),
        )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for page in pages:
            if page.page not in image_pages:
                continue
            encoded = base64.b64encode(self._bounded_image(page.image_bytes)).decode("ascii")
            visual_label, visual_resource = render_prompt("visual-page.md", page_number=page.page)
            content.extend(
                [
                    {"type": "input_text", "text": visual_label},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded}",
                        "detail": "high",
                    },
                ]
            )
        started = time.perf_counter()
        response = self._request(content)
        elapsed = time.perf_counter() - started
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no structured repair result.")
        expected_pages = sorted(page.page for page in pages)
        if sorted(parsed.reviewed_pages) != expected_pages:
            raise RuntimeError("GPT-5.6-luna repair must review every requested page exactly once.")
        resources = [*base_resources, repair_resource]
        if image_pages:
            resources.append(visual_resource)
        return parsed, self._read_usage(
            response, pages, image_pages, elapsed, resources, purpose="repair"
        )

    def _batches(self, pages: list[PageParse]) -> list[list[PageParse]]:
        batches: list[list[PageParse]] = []
        current: list[PageParse] = []
        size = 0
        for page in pages:
            page_size = sum(len(block.text) for block in page.blocks)
            if current and size + page_size > self.settings.cloud_batch_characters:
                batches.append(current)
                current, size = [], 0
            current.append(page)
            size += page_size
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _bounded_image(data: bytes, max_dimension: int = 2048) -> bytes:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.thumbnail((max_dimension, max_dimension))
        output = io.BytesIO()
        image.save(output, "JPEG", quality=85, optimize=True)
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
        output = io.BytesIO()
        crop.save(output, "JPEG", quality=95, optimize=True)
        return output.getvalue()

    def _request(
        self, content: list[dict[str, Any]], text_format: type[BaseModel] = CloudResult
    ) -> Any:
        policy = load_prompt("policy.md")
        return self.client.responses.parse(
            model=MODEL_NAME,
            reasoning={"effort": REASONING_EFFORT},
            store=False,
            tools=[],
            instructions=policy.text,
            input=cast(Any, [{"role": "user", "content": content}]),
            text_format=text_format,
        )

    @staticmethod
    def _read_usage(
        response: Any,
        pages: list[PageParse],
        image_pages: set[int],
        elapsed: float,
        prompt_resources: list[PromptResource] | None = None,
        purpose: str = "refinement",
        checkbox_ids: list[str] | None = None,
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
        resources = [load_prompt("policy.md"), *(prompt_resources or [])]
        call["prompts"] = [
            {"name": item.name, "version": item.version, "sha256": item.sha256}
            for item in dict.fromkeys(resources)
        ]
        call["checkbox_ids"] = checkbox_ids or []
        page_count = len(page_numbers)
        call["per_page_usage_estimate"] = {
            "input_tokens": input_tokens / page_count if input_tokens is not None else None,
            "cached_input_tokens": (
                token_usage.cached_input_tokens / page_count
                if token_usage.cached_input_tokens is not None
                else None
            ),
            "cache_write_input_tokens": (
                token_usage.cache_write_input_tokens / page_count
                if token_usage.cache_write_input_tokens is not None
                else None
            ),
            "output_tokens": output_tokens / page_count if output_tokens is not None else None,
            "total_tokens": total_tokens / page_count if total_tokens is not None else None,
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
        lines: list[str] = []
        context_resources: list[PromptResource] = []
        capability_instructions, capability_resources = self._capability_instructions(capabilities)
        checkbox_instructions, checkbox_resource = render_prompt("checkbox-discovery.md")
        for page in pages:
            full_context = page.page in full_context_pages
            blocks: list[str] = []
            for block in page.blocks:
                block_context, resource = self._block_context(block, full_context)
                blocks.append(block_context)
                context_resources.append(resource)
            template = "page-context-full.md" if full_context else "page-context-compact.md"
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
                blocks="\n".join(blocks),
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
        return rendered, [
            resource,
            *capability_resources,
            checkbox_resource,
            *context_resources,
        ]

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
        requires_gpt_review = (
            block.ocr_score is not None and block.ocr_score < LOW_CONFIDENCE_THRESHOLD
        )
        if full_context:
            return render_prompt(
                "block-context-full.md",
                block_json=_prompt_json(
                    {
                        "bbox": block.bbox,
                        "confidence": block.ocr_score,
                        "id": block.id,
                        "polygon": block.polygon,
                        "requires_gpt_review": requires_gpt_review,
                        "text": block.text,
                        "type": block.type,
                    }
                ),
            )
        bbox = [round(value, 4) for value in block.bbox] if block.bbox else None
        confidence = round(block.ocr_score, 3) if block.ocr_score is not None else None
        return render_prompt(
            "block-context-compact.md",
            block_json=_prompt_json(
                {
                    "bbox": bbox,
                    "confidence": confidence,
                    "id": block.id,
                    "requires_gpt_review": requires_gpt_review,
                    "text": block.text,
                    "type": block.type,
                }
            ),
        )
