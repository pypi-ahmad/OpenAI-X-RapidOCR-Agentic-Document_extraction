"""End-to-end document processing state machine."""

from __future__ import annotations

import copy
import importlib.metadata
from typing import Any, Protocol

from agentic_extractor.capabilities import validate_cloud_result
from agentic_extractor.costs import aggregate_usage, rate_assumptions
from agentic_extractor.ingest import load_document
from agentic_extractor.models import (
    CheckboxRecord,
    DocumentRequest,
    DocumentResult,
    EvidenceRef,
    ProcessingMode,
    RefinementRecord,
)
from agentic_extractor.ocr import (
    LocalParseResult,
    OCRResource,
    create_rapidocr_engine,
    encode_jpeg,
    ocr_page,
    parse_document_local,
)
from agentic_extractor.openai_refiner import (
    CloudEvidence,
    CloudRefinement,
    CloudResult,
    OpenAIRefiner,
)
from agentic_extractor.parse import (
    LOW_CONFIDENCE_THRESHOLD,
    build_layout_chunks,
    document_markdown,
    document_markdown_with_checkboxes,
    routing_reasons,
    should_send_image,
)
from agentic_extractor.quality import analyze_page, preprocess_if_improved


class Refiner(Protocol):
    def validate_configuration(self) -> None: ...

    def refine(
        self,
        pages: list[Any],
        image_pages: set[int],
        capabilities: set[str],
        allowed_classes: list[str],
        extraction_schema: dict[str, Any] | None,
    ) -> tuple[CloudResult, Any]: ...


class GPTRefinementError(RuntimeError):
    """Required GPT refinement did not complete successfully."""


def _process_local_document(
    request: DocumentRequest, *, ocr_resource: OCRResource | None = None
) -> LocalParseResult:
    """Run the offline RapidOCR Parse phase without constructing a cloud client."""
    document = load_document(request.file_name, request.file_bytes)
    selected = request.selected_pages or {page.number for page in document.pages}
    diagnostics: list[dict[str, object]] = []
    originals: dict[int, Any] = {}
    for page in document.pages:
        if page.number not in selected:
            continue
        diagnostic = analyze_page(
            page.number,
            page.image,
            dpi=page.source_dpi,
            source_format=page.source_format,
            orientation_correction_degrees=page.orientation_correction_degrees,
        )
        if page.orientation_correction_degrees:
            diagnostic.preprocessing_actions.append(
                f"EXIF orientation normalized by {page.orientation_correction_degrees}°"
            )
        if request.enable_preprocessing:
            originals[page.number] = page.image.copy()
            page.image, actions = preprocess_if_improved(page.number, page.image, diagnostic)
            diagnostic.preprocessing_actions.extend(actions)
        diagnostics.append(diagnostic.model_dump(mode="json"))
    result = parse_document_local(document, selected, ocr_resource or create_rapidocr_engine())
    result.quality_diagnostics = diagnostics
    for parsed in result.pages:
        if original := originals.get(parsed.page):
            parsed.original_image_bytes = encode_jpeg(original)
    result.adaptive_processing = {
        "strategy": "one-page bounded RapidOCR calls",
        "initial_batch_size": 1,
        "final_batch_size": 1,
        "memory_retries": 0,
        "preprocessing": [
            {"page": item["page"], "actions": item["preprocessing_actions"]} for item in diagnostics
        ],
    }
    return result


def process_hybrid_document(
    request: DocumentRequest,
    *,
    ocr_resource: OCRResource | None = None,
    refiner: Refiner | None = None,
) -> LocalParseResult:
    """Run required RapidOCR Parse followed by required GPT refinement."""
    cloud_refiner = refiner or OpenAIRefiner()
    cloud_refiner.validate_configuration()
    local = _process_local_document(request, ocr_resource=ocr_resource)
    return refine_local_parse(local, request, cloud_refiner)


def refine_local_parse(
    local: LocalParseResult, request: DocumentRequest, refiner: Refiner | None = None
) -> LocalParseResult:
    """Route and merge only evidence-linked GPT refinements into a local Parse result."""
    local.requested_mode = request.mode
    local.effective_mode = request.mode
    local.routing = {
        page.page: routing_reasons(page, request.mode, request.forced_cloud_pages)
        for page in local.pages
    }
    local.cloud_pages = [page.page for page in local.pages]
    full_context_pages = {page for page, reasons in local.routing.items() if reasons}
    # Checkbox discovery is visual: every selected page image must reach GPT. Balanced
    # still limits full OCR/layout context to pages selected by routing heuristics.
    local.cloud_image_pages = [page.page for page in local.pages]
    cloud_refiner = refiner or OpenAIRefiner()
    requested_capabilities = {capability.value for capability in request.capabilities}
    try:
        cloud, usage = cloud_refiner.refine(
            local.pages,
            full_context_pages,
            {"Parse"},
            [],
            None,
        )
    except Exception as exc:
        raise GPTRefinementError(
            f"GPT-5.6-luna refinement failed; no extraction was completed: {exc}"
        ) from exc
    local.usage = aggregate_usage([local.usage, usage]) if local.usage.call_count else usage
    if usage.call_count < 1:
        raise GPTRefinementError(
            "GPT-5.6-luna returned no API call record; no extraction was completed."
        )
    local.cloud_attempts.append(
        {
            "attempt": len(local.cloud_attempts) + 1,
            "purpose": "parse_refinement",
            "output": cloud.model_dump(mode="json"),
        }
    )
    local.warnings.extend(cloud.warnings)
    local.markdown, local.refinements, refinement_warnings = _apply_refinements(
        local.pages, cloud, set(local.cloud_image_pages)
    )
    local.warnings.extend(refinement_warnings)
    _record_low_confidence_reviews(local, request)
    if requested_capabilities - {"Parse"}:
        local, cloud = refine_markdown_workflows(local, request, cloud_refiner, cloud)
    local.cloud_output = cloud.model_dump(mode="json")
    local.document_metadata["gpt_rate_assumptions"] = rate_assumptions()
    return local


def _record_low_confidence_reviews(local: LocalParseResult, request: DocumentRequest) -> None:
    if request.mode is not ProcessingMode.HIGH_ACCURACY:
        return
    records_by_block: dict[str, list[RefinementRecord]] = {}
    for record in local.refinements:
        if record.block_id:
            records_by_block.setdefault(record.block_id, []).append(record)
    reviews: list[dict[str, object]] = []
    for page in local.pages:
        for block in page.blocks:
            if block.ocr_score is None or block.ocr_score >= LOW_CONFIDENCE_THRESHOLD:
                continue
            records = records_by_block.get(block.id, [])
            status = next(
                (
                    candidate
                    for candidate in ("accepted", "abstained", "rejected")
                    if any(record.status == candidate for record in records)
                ),
                "missing",
            )
            review: dict[str, object] = {
                "block_id": block.id,
                "page": page.page,
                "ocr_score": block.ocr_score,
                "status": status,
            }
            reason = next((record.reason for record in records if record.reason), None)
            if reason:
                review["reason"] = reason
            reviews.append(review)
    local.document_metadata["low_confidence_block_reviews"] = reviews


def refine_markdown_workflows(
    local: LocalParseResult,
    request: DocumentRequest,
    refiner: Refiner,
    cloud: CloudResult | None = None,
) -> tuple[LocalParseResult, CloudResult]:
    """Run optional workflows from an existing refined Markdown Parse result."""
    optional_capabilities = {capability.value for capability in request.capabilities} - {"Parse"}
    existing = cloud or (
        CloudResult.model_validate(local.cloud_output) if local.cloud_output else None
    )
    if existing is None:
        raise GPTRefinementError("A completed GPT-refined Parse result is required.")
    markdown_refine = getattr(refiner, "refine_markdown", None)
    if not callable(markdown_refine):
        # Preserve injected adapter compatibility; the production OpenAI adapter always
        # implements the Markdown-only boundary.
        return local, existing
    try:
        downstream, downstream_usage = markdown_refine(
            local.markdown,
            local.pages,
            optional_capabilities,
            request.allowed_classes,
            request.extraction_schema,
        )
    except Exception as exc:
        raise GPTRefinementError(
            f"GPT-5.6-luna Markdown workflow failed; no workflow result was completed: {exc}"
        ) from exc
    existing.classifications = downstream.classifications
    existing.sections = downstream.sections
    existing.splits = downstream.splits
    existing.extracted_fields = downstream.extracted_fields
    existing.warnings.extend(downstream.warnings)
    local.usage = aggregate_usage([local.usage, downstream_usage])
    local.cloud_attempts.append(
        {
            "attempt": len(local.cloud_attempts) + 1,
            "purpose": "markdown_workflow",
            "output": downstream.model_dump(mode="json"),
        }
    )
    local.warnings.extend(downstream.warnings)
    local.cloud_output = existing.model_dump(mode="json")
    return local, existing


def _refinement_record(
    refinement: CloudRefinement,
    evidence: list[CloudEvidence],
    status: str,
    reason: str | None = None,
) -> RefinementRecord:
    return RefinementRecord.model_validate(
        {
            "page": refinement.page,
            "block_id": refinement.block_id,
            "status": status,
            "proposed_text": refinement.corrected_text,
            "proposed_type": refinement.block_type,
            "proposed_reading_order": refinement.reading_order,
            "evidence": [
                EvidenceRef.model_validate(item.model_dump(exclude={"chunk_id"}))
                for item in evidence
            ],
            "reason": reason,
        }
    )


def _apply_refinements(
    raw_pages: list[Any],
    cloud: CloudResult,
    visual_pages: set[int],
    checkboxes: list[CheckboxRecord] | None = None,
) -> tuple[str, list[RefinementRecord], list[str]]:
    pages = {page.page: page for page in raw_pages}
    blocks = {block.id: block for page in raw_pages for block in page.blocks}
    refined_pages = copy.deepcopy(raw_pages)
    refined_blocks = {block.id: block for page in refined_pages for block in page.blocks}
    requested_order: dict[str, int] = {}
    records: list[RefinementRecord] = []
    warnings: list[str] = []
    for refinement in cloud.refinements:
        evidence = [
            item
            for item in refinement.evidence
            if _valid_refinement_evidence(item, pages, blocks, visual_pages)
        ]

        if refinement.abstained:
            records.append(
                _refinement_record(
                    refinement,
                    evidence,
                    "abstained",
                    refinement.warning or "GPT abstained from this refinement.",
                )
            )
            if refinement.warning:
                warnings.append(refinement.warning)
            continue
        block = blocks.get(refinement.block_id or "")
        if refinement.page not in pages or block is None or block.page != refinement.page:
            reason = "GPT refinement referencing an unknown page or block was rejected."
            warnings.append(reason)
            records.append(_refinement_record(refinement, evidence, "rejected", reason))
            continue
        if not refinement.verified or not evidence:
            reason = f"Unsupported GPT refinement for {block.id} was rejected."
            warnings.append(reason)
            records.append(_refinement_record(refinement, evidence, "rejected", reason))
            continue
        records.append(_refinement_record(refinement, evidence, "accepted"))
        refined_block = refined_blocks[block.id]
        if refinement.corrected_text is not None:
            refined_block.text = refinement.corrected_text
        if refinement.block_type is not None:
            refined_block.type = refinement.block_type
        if refinement.reading_order is not None:
            requested_order[refined_block.id] = refinement.reading_order
    for page in refined_pages:
        page_orders = {
            block.id: requested_order[block.id]
            for block in page.blocks
            if block.id in requested_order
        }
        if page_orders:
            original = {block.id: index for index, block in enumerate(page.blocks)}
            page.blocks.sort(
                key=lambda block: (page_orders.get(block.id, 1_000_000), original[block.id])
            )
            page.layout_signals["reading_order"] = "gpt-evidence-linked"
        page.chunks = build_layout_chunks(page.blocks)
    markdown = (
        document_markdown_with_checkboxes(refined_pages, checkboxes)
        if checkboxes is not None
        else document_markdown(refined_pages)
    )
    return markdown, records, warnings


def render_result_markdown(local: LocalParseResult) -> str:
    """Rebuild derived Markdown from immutable OCR, GPT refinements, and checkboxes."""
    if not local.cloud_output:
        return document_markdown_with_checkboxes(local.pages, local.checkboxes)
    cloud = CloudResult.model_validate(local.cloud_output)
    markdown, _, _ = _apply_refinements(
        local.pages, cloud, set(local.cloud_image_pages), local.checkboxes
    )
    return markdown


def _valid_refinement_evidence(
    evidence: CloudEvidence,
    pages: dict[int, Any],
    blocks: dict[str, Any],
    cloud_pages: set[int],
) -> bool:
    if evidence.page not in pages:
        return False
    if evidence.block_id:
        block = blocks.get(evidence.block_id)
        return bool(
            block
            and block.page == evidence.page
            and (not evidence.quote or evidence.quote in block.text)
        )
    return bool(
        evidence.source == "gpt-visual"
        and evidence.page in cloud_pages
        and evidence.bbox
        and len(evidence.bbox) == 4
        and all(0 <= value <= 1 for value in evidence.bbox)
    )


def process_document(
    request: DocumentRequest,
    *,
    ocr_resource: OCRResource | None = None,
    refiner: Refiner | None = None,
) -> DocumentResult:
    """Run both required engines; preserve local diagnostics if cloud processing fails."""
    cloud_refiner = refiner or OpenAIRefiner()
    cloud_refiner.validate_configuration()
    document = load_document(request.file_name, request.file_bytes)
    selected = request.selected_pages or {page.number for page in document.pages}
    invalid = selected - {page.number for page in document.pages}
    if invalid:
        raise ValueError(f"Selected pages do not exist: {sorted(invalid)}")
    pages = [page for page in document.pages if page.number in selected]
    resource = ocr_resource or create_rapidocr_engine()
    parsed = [ocr_page(resource, page.number, page.image) for page in pages]
    local_markdown = document_markdown(parsed)
    image_pages = {
        page.page
        for page in parsed
        if should_send_image(page, request.mode, request.forced_cloud_pages)
    }
    warnings = [resource.warning] if resource.warning else []
    metadata: dict[str, Any] = {
        "file_name": request.file_name,
        "mime_type": document.mime_type,
        "page_count": len(document.pages),
        "processed_pages": sorted(selected),
        "cloud_image_pages": sorted(image_pages),
        "mode": request.mode.value,
        "ocr_device": resource.device,
        "ocr_seconds": sum(page.ocr_seconds for page in parsed),
        "rapidocr_version": _version("rapidocr"),
        "openai_model": "gpt-5.6-luna",
        "reasoning_effort": "medium",
        "balanced_thresholds_calibrated": False,
    }
    try:
        cloud, usage = cloud_refiner.refine(
            parsed,
            image_pages,
            {capability.value for capability in request.capabilities},
            request.allowed_classes,
            request.extraction_schema,
        )
        if usage.call_count < 1:
            raise RuntimeError("GPT-5.6-luna returned no API call record.")
        refined_markdown, refinements, refinement_warnings = _apply_refinements(
            parsed, cloud, image_pages
        )
        classifications, sections, splits, extraction, validation_warnings = validate_cloud_result(
            cloud, parsed, request.allowed_classes, request.extraction_schema
        )
        warnings.extend(refinement_warnings)
        warnings.extend(validation_warnings)
        return DocumentResult(
            markdown=refined_markdown or local_markdown,
            blocks=[block for page in parsed for block in page.blocks],
            classifications=classifications,
            sections=sections,
            splits=splits,
            extraction=extraction,
            metadata=metadata,
            usage=usage,
            warnings=warnings,
            refinements=refinements,
        )
    except Exception as exc:
        raise GPTRefinementError(
            f"GPT-5.6-luna refinement failed; no extraction was completed: {exc}"
        ) from exc


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
