"""End-to-end document processing state machine.

Responsible for: orchestrating the canonical hybrid extraction pipeline
(`process_document` and `process_hybrid_document`) across ingestion,
RapidOCR, PP-DocLayoutV3, table structure analysis, visual crop routing,
and OpenAI `gpt-6-sol` refinement.

Must not: bypass any of the three required engines, mutate raw OCR blocks
(immutable evidence boundary), or allow single-engine fallback.

Next: `workflow.py`, which wraps this pipeline in an auditable agent
state machine with deterministic validation and repair gates.
"""

from __future__ import annotations

import copy
import importlib.metadata
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from agentic_extractor.capabilities import validate_cloud_result
from agentic_extractor.checkbox_vision import (
    detect_page_checkboxes,
    is_credible_checkbox_candidate,
)
from agentic_extractor.config import SETTINGS
from agentic_extractor.costs import aggregate_usage, rate_assumptions
from agentic_extractor.ingest import load_document
from agentic_extractor.layout import (
    PPDocLayoutResource,
    apply_document_layout,
    create_pp_doclayout_engine,
)
from agentic_extractor.models import (
    Block,
    CheckboxRecord,
    DocumentRequest,
    DocumentResult,
    EvidenceRef,
    ProcessingMode,
    RefinementRecord,
)
from agentic_extractor.ocr import (
    EngineProvenance,
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
    CloudSemanticRegion,
    OpenAIRefiner,
)
from agentic_extractor.parse import (
    LOW_CONFIDENCE_THRESHOLD,
    ParseChunk,
    document_markdown,
    document_markdown_with_checkboxes,
    routing_reasons,
)
from agentic_extractor.quality import analyze_page, preprocess_if_improved
from agentic_extractor.redaction_vision import detect_page_redactions
from agentic_extractor.table_structure import apply_table_reviews, build_chunks_with_tables
from agentic_extractor.timing import record_stage_timing
from agentic_extractor.visual_routing import plan_visual_review_regions


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


_SIGNATURE_MARKDOWN = "[SIGNED]\n[ILLEGIBLE_SIGNATURE]"
_ELECTRONIC_ATTESTATION = re.compile(r"^Electronically\s+signed\s+by:\s*\S", re.IGNORECASE)


def _process_local_document(
    request: DocumentRequest,
    *,
    ocr_resource: OCRResource | None = None,
) -> LocalParseResult:
    """Run the immutable RapidOCR first pass without constructing a cloud client."""
    ingest_started = time.perf_counter()
    document = load_document(request.file_name, request.file_bytes)
    ingest_seconds = time.perf_counter() - ingest_started
    selected = request.selected_pages or {page.number for page in document.pages}
    source_pages = [page for page in document.pages if page.number in selected]
    quality_started = time.perf_counter()
    diagnostics: list[dict[str, object]] = []
    originals: dict[int, Any] = {}

    def prepare_page(page: Any) -> tuple[Any, Any, Any | None, Any]:
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
        original = None
        prepared = page.image
        if request.enable_preprocessing:
            original = page.image.copy()
            prepared, actions = preprocess_if_improved(page.number, page.image, diagnostic)
            diagnostic.preprocessing_actions.extend(actions)
        return page, diagnostic, original, prepared

    quality_workers = min(len(source_pages), SETTINGS.ocr_max_workers, max(1, os.cpu_count() or 1))
    if quality_workers > 1:
        with ThreadPoolExecutor(
            max_workers=quality_workers, thread_name_prefix="page-preparation"
        ) as pool:
            prepared_pages = list(pool.map(prepare_page, source_pages))
    else:
        prepared_pages = [prepare_page(page) for page in source_pages]
    for page, diagnostic, original, prepared in prepared_pages:
        page.image = prepared
        if original is not None:
            originals[page.number] = original
        diagnostics.append(diagnostic.model_dump(mode="json"))
    quality_seconds = time.perf_counter() - quality_started
    ocr_started = time.perf_counter()
    result = parse_document_local(document, selected, ocr_resource or create_rapidocr_engine())
    record_stage_timing(result.timings, "document_ingest_seconds", ingest_seconds)
    record_stage_timing(result.timings, "page_preparation_seconds", quality_seconds)
    record_stage_timing(result.timings, "ocr_wall_seconds", time.perf_counter() - ocr_started)
    result.timings["quality_seconds"] = quality_seconds
    result.quality_diagnostics = diagnostics
    for parsed in result.pages:
        if original := originals.get(parsed.page):
            parsed.original_image_bytes = encode_jpeg(original)
    result.adaptive_processing.update(
        {
            "memory_retries": 0,
            "page_preparation_workers": quality_workers,
            "render_cache": {
                "hits": document.render_cache_hits,
                "misses": document.render_cache_misses,
            },
            "preprocessing": [
                {"page": item["page"], "actions": item["preprocessing_actions"]}
                for item in diagnostics
            ],
        }
    )
    return result


def process_hybrid_document(
    request: DocumentRequest,
    *,
    ocr_resource: OCRResource | None = None,
    layout_resource: PPDocLayoutResource | None = None,
    refiner: Refiner | None = None,
) -> LocalParseResult:
    """Run required RapidOCR Parse followed by required GPT refinement."""
    started = time.perf_counter()
    cloud_refiner = refiner or OpenAIRefiner()
    configuration_started = time.perf_counter()
    cloud_refiner.validate_configuration()
    configuration_seconds = time.perf_counter() - configuration_started
    local = _process_local_document(request, ocr_resource=ocr_resource)
    record_stage_timing(local.timings, "configuration_seconds", configuration_seconds)
    layout_initialization_started = time.perf_counter()
    active_layout = layout_resource or create_pp_doclayout_engine()
    record_stage_timing(
        local.timings,
        "layout_initialization_seconds",
        time.perf_counter() - layout_initialization_started,
    )
    apply_document_layout(
        local,
        None,
        active_layout,
    )
    result = refine_local_parse(local, request, cloud_refiner)
    result.timings["total_seconds"] = time.perf_counter() - started
    return result


def refine_local_parse(
    local: LocalParseResult,
    request: DocumentRequest,
    refiner: Refiner | None = None,
    *,
    defer_markdown_workflows: bool = False,
) -> LocalParseResult:
    """Route and merge only evidence-linked GPT refinements into a local Parse result."""
    routing_started = time.perf_counter()
    local.requested_mode = request.mode
    local.effective_mode = request.mode
    local.routing = {
        page.page: routing_reasons(page, request.mode, request.forced_cloud_pages)
        for page in local.pages
    }
    local.cloud_pages = [page.page for page in local.pages]
    # Every selected page receives a high-detail image in both modes.
    local.cloud_image_pages = [page.page for page in local.pages]
    record_stage_timing(local.timings, "routing_seconds", time.perf_counter() - routing_started)
    _detect_local_checkboxes(local)
    _detect_local_redactions(local)
    visual_routing_started = time.perf_counter()
    _plan_visual_review(local, request)
    record_stage_timing(
        local.timings,
        "visual_routing_seconds",
        time.perf_counter() - visual_routing_started,
    )
    full_context_pages = (
        {page.page for page in local.pages}
        if request.mode is ProcessingMode.HIGH_ACCURACY
        else {
            page.page
            for page in local.pages
            if any(region.page_wide for region in page.visual_review_regions)
        }
    )
    cloud_refiner = refiner or OpenAIRefiner()
    requested_capabilities = {capability.value for capability in request.capabilities}
    refinement_started = time.perf_counter()
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
            f"gpt-6-sol refinement failed; no extraction was completed: {exc}"
        ) from exc
    record_stage_timing(
        local.timings, "gpt_refinement_seconds", time.perf_counter() - refinement_started
    )
    local.usage = aggregate_usage([local.usage, usage]) if local.usage.call_count else usage
    if usage.call_count < 1:
        raise GPTRefinementError(
            "gpt-6-sol returned no API call record; no extraction was completed."
        )
    local.cloud_attempts.append(
        {
            "attempt": len(local.cloud_attempts) + 1,
            "purpose": "parse_refinement",
            "output": cloud.model_dump(mode="json"),
        }
    )
    local.warnings.extend(cloud.warnings)
    merge_started = time.perf_counter()
    (
        local.markdown,
        local.refinements,
        refinement_warnings,
        table_reviews,
    ) = _apply_refinements(local.pages, cloud, set(local.cloud_image_pages))
    record_stage_timing(
        local.timings, "refinement_merge_seconds", time.perf_counter() - merge_started
    )
    local.document_metadata["table_reviews"] = table_reviews
    local.warnings.extend(refinement_warnings)
    _record_low_confidence_reviews(local, request)
    if requested_capabilities - {"Parse"} and not defer_markdown_workflows:
        local, cloud = refine_markdown_workflows(local, request, cloud_refiner, cloud)
    local.cloud_output = cloud.model_dump(mode="json")
    local.document_metadata["gpt_rate_assumptions"] = rate_assumptions()
    local.warnings = list(dict.fromkeys(local.warnings))
    return local


def _detect_local_checkboxes(local: LocalParseResult) -> None:
    """Attach additive pixel proposals without changing immutable OCR evidence."""
    started = time.perf_counter()
    failed_pages: list[int] = []
    for page in local.pages:
        page_started = time.perf_counter()
        try:
            page.local_checkbox_candidates = detect_page_checkboxes(page)
        except Exception as exc:
            page.local_checkbox_candidates = []
            failed_pages.append(page.page)
            page.warnings.append(
                "OpenCV checkbox detection failed; checkbox acceptance is disabled for this "
                f"page ({type(exc).__name__})."
            )
        page.stage_timings["checkbox_detection_seconds"] = time.perf_counter() - page_started
    local.timings["checkbox_detection_seconds"] = time.perf_counter() - started
    local.adaptive_processing["checkbox_detection"] = {
        "engine": "OpenCV",
        "device": "CPU",
        "confidence_calibrated": False,
        "candidate_count": sum(len(page.local_checkbox_candidates) for page in local.pages),
        "credible_candidate_count": sum(
            is_credible_checkbox_candidate(page, candidate)
            for page in local.pages
            for candidate in page.local_checkbox_candidates
        ),
        "candidate_pages": [
            page.page
            for page in local.pages
            if any(
                is_credible_checkbox_candidate(page, candidate)
                for candidate in page.local_checkbox_candidates
            )
        ],
        "skipped_pages": [
            page.page
            for page in local.pages
            if not any(
                is_credible_checkbox_candidate(page, candidate)
                for candidate in page.local_checkbox_candidates
            )
        ],
        "failed_pages": failed_pages,
    }


def _detect_local_redactions(local: LocalParseResult) -> None:
    """Attach non-publishable pixel-mask proposals for bounded visual confirmation."""
    started = time.perf_counter()
    failed_pages: list[int] = []
    for page in local.pages:
        page_started = time.perf_counter()
        try:
            page.local_redaction_candidates = detect_page_redactions(page)
        except Exception as exc:
            page.local_redaction_candidates = []
            failed_pages.append(page.page)
            page.warnings.append(
                "OpenCV redaction-mask detection failed; no inferred placeholders were "
                f"published for this page ({type(exc).__name__})."
            )
        page.stage_timings["redaction_detection_seconds"] = time.perf_counter() - page_started
    local.timings["redaction_detection_seconds"] = time.perf_counter() - started
    local.adaptive_processing["redaction_detection"] = {
        "engine": "OpenCV",
        "device": "CPU",
        "confidence_calibrated": False,
        "candidate_count": sum(len(page.local_redaction_candidates) for page in local.pages),
        "failed_pages": failed_pages,
        "publication_rule": "local mask candidate plus Sol visual confirmation",
    }


def _plan_visual_review(local: LocalParseResult, request: DocumentRequest) -> None:
    quality_by_page = {
        item["page"]: item
        for item in local.quality_diagnostics
        if isinstance(item.get("page"), int)
    }
    for page in local.pages:
        page.visual_review_regions = plan_visual_review_regions(
            page,
            quality=quality_by_page.get(page.page),
            forced=page.page in request.forced_cloud_pages,
        )
    regions = [
        region.model_dump(mode="json")
        for page in local.pages
        for region in page.visual_review_regions
    ]
    page_area = {
        page.page: min(
            1.0,
            sum(
                (region.bbox[2] - region.bbox[0]) * (region.bbox[3] - region.bbox[1])
                for region in page.visual_review_regions
            ),
        )
        for page in local.pages
    }
    local.visual_routing = {
        "overview_pages": [
            page.page
            for page in local.pages
            if not any(region.page_wide for region in page.visual_review_regions)
        ],
        "high_resolution_regions": regions,
        "high_resolution_region_count": len(regions),
        "high_resolution_area_ratio_by_page": page_area,
        "max_regions_per_page": 12,
        "region_limit_is_soft": True,
        "max_localized_region_area_ratio": 0.15,
    }
    local.adaptive_processing["visual_routing"] = local.visual_routing


def _record_low_confidence_reviews(local: LocalParseResult, request: DocumentRequest) -> None:
    # High Accuracy audit invariant: every block below LOW_CONFIDENCE_THRESHOLD (0.85)
    # must have an explicit review status (accepted, abstained, rejected, or missing) in metadata.
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
    workflow_started = time.perf_counter()
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
            f"gpt-6-sol Markdown workflow failed; no workflow result was completed: {exc}"
        ) from exc
    record_stage_timing(
        local.timings, "markdown_workflow_seconds", time.perf_counter() - workflow_started
    )
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
) -> tuple[str, list[RefinementRecord], list[str], list[dict[str, Any]]]:
    pages = {page.page: page for page in raw_pages}
    blocks = {block.id: block for page in raw_pages for block in page.blocks}
    redactions = {
        candidate.id: (page, candidate)
        for page in raw_pages
        for candidate in page.local_redaction_candidates
    }
    refined_pages = copy.deepcopy(raw_pages)
    refined_pages_by_number = {page.page: page for page in refined_pages}
    refined_blocks = {block.id: block for page in refined_pages for block in page.blocks}
    requested_order: dict[str, int] = {}
    records: list[RefinementRecord] = []
    warnings: list[str] = []
    reviewed_redactions: set[str] = set()
    derived_redaction_ids: set[str] = set()
    derived_redaction_labels: dict[str, str | None] = {}
    for refinement in cloud.refinements:
        evidence = [
            item
            for item in refinement.evidence
            if _valid_refinement_evidence(item, pages, blocks, visual_pages)
        ]
        block = blocks.get(refinement.block_id or "")
        attestation = _signature_attestation(pages.get(refinement.page), block)
        signature_evidence = (
            _signature_visual_evidence(block, evidence) if attestation is not None else []
        )

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
            if refinement.block_id is not None and refinement.block_id in redactions:
                reviewed_redactions.add(refinement.block_id)
            if block is not None and attestation is not None and signature_evidence:
                _apply_signature_semantics(
                    refined_blocks,
                    block,
                    attestation,
                    signature_evidence,
                    records,
                    add_signature_record=True,
                )
            continue
        if refinement.block_id is not None and refinement.block_id in redactions:
            reviewed_redactions.add(refinement.block_id)
            source_page, candidate = redactions[refinement.block_id]
            visual_evidence = [
                item
                for item in evidence
                if item.source == "gpt-visual"
                and item.bbox
                and _candidate_evidence_overlap(candidate.bbox, item.bbox) >= 0.50
            ]
            if (
                refinement.page != source_page.page
                or refinement.corrected_text != "[REDACTED]"
                or not refinement.verified
                or not visual_evidence
            ):
                reason = (
                    f"Unsupported redaction confirmation for {candidate.id} was rejected; "
                    "the placeholder was not published."
                )
                warnings.append(reason)
                records.append(_refinement_record(refinement, evidence, "rejected", reason))
                continue
            records.append(_refinement_record(refinement, visual_evidence, "accepted"))
            block_id = f"{candidate.id}-block"
            derived_redaction_ids.add(block_id)
            derived_redaction_labels[block_id] = candidate.label_block_id
            if block_id not in refined_blocks:
                block = _redaction_block(source_page, candidate, block_id)
                _insert_redaction_block(
                    refined_pages_by_number[source_page.page], block, candidate.label_block_id
                )
                refined_blocks[block_id] = block
            continue
        if refinement.page not in pages or block is None or block.page != refinement.page:
            reason = "GPT refinement referencing an unknown page or block was rejected."
            warnings.append(reason)
            records.append(_refinement_record(refinement, evidence, "rejected", reason))
            continue
        if attestation is not None and (
            refinement.corrected_text != _SIGNATURE_MARKDOWN or not signature_evidence
        ):
            reason = (
                f"Signature refinement for {block.id} was rejected because it attempted a "
                "transcription or lacked matching signature pixels."
            )
            warnings.append(reason)
            records.append(_refinement_record(refinement, evidence, "rejected", reason))
            continue
        if not refinement.verified or not evidence:
            reason = f"Unsupported GPT refinement for {block.id} was rejected."
            warnings.append(reason)
            records.append(_refinement_record(refinement, evidence, "rejected", reason))
            continue
        if (
            refinement.corrected_text is not None
            and _critical_token_change(block.text, refinement.corrected_text)
            and not any(item.source == "gpt-visual" for item in evidence)
        ):
            reason = f"Critical-token refinement for {block.id} lacked visual evidence."
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
        if (
            refinement.corrected_text == _SIGNATURE_MARKDOWN
            and attestation is not None
            and signature_evidence
        ):
            refined_block.type = "paragraph"
            _apply_signature_semantics(
                refined_blocks,
                block,
                attestation,
                signature_evidence,
                records,
                add_signature_record=False,
            )
    for candidate_id in sorted(set(redactions) - reviewed_redactions):
        warnings.append(
            f"Local redaction candidate {candidate_id} received no Sol outcome; "
            "the placeholder was not published."
        )
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
    table_audits, table_warnings = apply_table_reviews(refined_pages, cloud.table_reviews)
    warnings.extend(table_warnings)
    _apply_semantic_regions(refined_pages, cloud.semantic_regions, warnings)
    from agentic_extractor.rich_document import apply_visual_objects, validate_links

    audits = [audit for page in raw_pages for audit in page.visual_audits]
    warnings.extend(
        apply_visual_objects(
            refined_pages,
            cloud.visual_objects,
            audits,
            {(region.page, region.id): region.reading_order for region in cloud.semantic_regions},
        )
    )
    warnings.extend(validate_links(refined_pages, cloud.document_links))
    refined_by_page = {page.page: page for page in refined_pages}
    for raw_page in raw_pages:
        refined_page = refined_by_page[raw_page.page]
        raw_page.table_structures = copy.deepcopy(refined_page.table_structures)
        existing = {block.id for block in raw_page.blocks}
        for block in refined_page.blocks:
            if block.id not in derived_redaction_ids or block.id in existing:
                continue
            _insert_redaction_block(
                raw_page, copy.deepcopy(block), derived_redaction_labels.get(block.id)
            )
            existing.add(block.id)
        raw_page.chunks = copy.deepcopy(refined_page.chunks)
    markdown = (
        document_markdown_with_checkboxes(refined_pages, checkboxes)
        if checkboxes is not None
        else document_markdown(refined_pages)
    )
    return markdown, records, warnings, table_audits


def _apply_semantic_regions(
    pages: list[Any], proposals: list[CloudSemanticRegion], warnings: list[str]
) -> None:
    proposals_by_page: dict[int, list[CloudSemanticRegion]] = {}
    for proposal in proposals:
        proposals_by_page.setdefault(proposal.page, []).append(proposal)
    for page in pages:
        page_proposals = proposals_by_page.get(page.page, [])
        if not page_proposals:
            page.chunks = build_chunks_with_tables(page)
            continue
        try:
            page.chunks = _validated_semantic_chunks(page, page_proposals)
        except ValueError as exc:
            page.chunks = build_chunks_with_tables(page)
            warnings.append(
                f"Page {page.page} semantic reconstruction was rejected; grounded fallback "
                f"was used ({exc})."
            )


def _validated_semantic_chunks(page: Any, proposals: list[CloudSemanticRegion]) -> list[ParseChunk]:
    blocks = {block.id: block for block in page.blocks if block.source == "rapidocr"}
    layout_ids = {region.id for region in page.layout_regions}
    proposal_ids = [proposal.id for proposal in proposals]
    if len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError("semantic region IDs must be unique")
    fallback_tables = [chunk for chunk in build_chunks_with_tables(page) if chunk.type == "table"]
    table_block_ids = set().union(
        *(set(chunk.source_block_ids) for chunk in fallback_tables), set()
    )
    seen: set[str] = set()
    chunks: list[ParseChunk] = []
    for proposal in sorted(proposals, key=lambda item: item.reading_order):
        if proposal.page != page.page:
            raise ValueError("semantic region references the wrong page")
        if not all(
            region_id in layout_ids or _is_page_semantic_fallback_id(region_id, page.page)
            for region_id in proposal.source_layout_region_ids
        ):
            raise ValueError(f"semantic region {proposal.id} references an unknown layout region")
        if len(proposal.source_block_ids) != len(set(proposal.source_block_ids)):
            raise ValueError(f"semantic region {proposal.id} duplicates a source block")
        unknown = set(proposal.source_block_ids) - set(blocks)
        duplicate = set(proposal.source_block_ids) & seen
        if unknown or duplicate:
            raise ValueError(f"semantic region {proposal.id} has unknown or reused source blocks")
        source_blocks = [blocks[block_id] for block_id in proposal.source_block_ids]
        seen.update(proposal.source_block_ids)
        semantic_source_blocks = [
            block for block in source_blocks if block.id not in table_block_ids
        ]
        if not semantic_source_blocks:
            continue
        validated_type = _validated_semantic_type(page, proposal, semantic_source_blocks)
        chunk_type = validated_type
        markdown = _semantic_markdown(proposal, semantic_source_blocks)
        if validated_type == "table":
            chunk_type = "text"
        semantic_groups = _semantic_source_groups(chunk_type, semantic_source_blocks)
        for group_index, (group_type, group_blocks) in enumerate(semantic_groups, 1):
            chunks.append(
                ParseChunk(
                    id=(
                        proposal.id if len(semantic_groups) == 1 else f"{proposal.id}-{group_index}"
                    ),
                    page=page.page,
                    type=group_type,
                    reading_order=proposal.reading_order,
                    text="\n".join(block.text.strip() for block in group_blocks),
                    markdown=(
                        markdown
                        if len(semantic_groups) == 1
                        else _semantic_markdown(proposal, group_blocks)
                    ),
                    source_block_ids=[block.id for block in group_blocks],
                    bbox=_blocks_bbox(group_blocks),
                    raw_scores=[block.ocr_score for block in group_blocks],
                )
            )
    missing_blocks = [block for block in blocks.values() if block.id not in seen]
    for block in missing_blocks:
        chunks.append(
            ParseChunk(
                id=f"p{page.page}-semantic-fallback-{block.id}",
                page=page.page,
                type="text",
                reading_order=len(proposals) + len(chunks) + 1,
                text=block.text,
                markdown=block.text,
                source_block_ids=[block.id],
                bbox=block.bbox,
                raw_scores=[block.ocr_score],
            )
        )
    chunks.extend(fallback_tables)
    derived_chunks = []
    for block in page.blocks:
        if block.source == "rapidocr":
            continue
        derived_chunks.append(
            ParseChunk(
                id=f"p{page.page}-derived-{len(derived_chunks) + 1}",
                page=page.page,
                type=block.type,
                reading_order=len(chunks) + len(derived_chunks) + 1,
                text=block.text,
                markdown=block.text,
                source_block_ids=[block.id],
                bbox=block.bbox,
                raw_scores=[block.ocr_score],
            )
        )
    chunks.extend(derived_chunks)
    chunks.sort(key=lambda chunk: chunk.reading_order)
    for index, chunk in enumerate(chunks, 1):
        chunk.reading_order = index
    return chunks


def _is_page_semantic_fallback_id(region_id: str, page: int) -> bool:
    """Accept only IDs produced by the local semantic grouping fallback for this page."""
    return re.fullmatch(rf"p{page}-sr-fallback-[1-9]\d*", region_id) is not None


def _validated_semantic_type(
    page: Any, proposal: CloudSemanticRegion, source_blocks: list[Any]
) -> str:
    if proposal.type not in {"marginalia", "logo", "attestation", "figure", "scan_code"}:
        return proposal.type
    region_by_id = {region.id: region.label.lower() for region in page.layout_regions}
    block_regions = {
        link.block_id: region_by_id.get(link.primary_region_id, "")
        for link in page.layout_block_links
    }
    labels = {block_regions.get(block.id, "") for block in source_blocks}
    if proposal.type == "marginalia" and not labels & {
        "header",
        "footer",
        "footnote",
        "number",
        "vision_footnote",
    }:
        return "text"
    if proposal.type == "logo":
        visual_bbox = _blocks_bbox(source_blocks)
        visual_evidence = (
            [
                evidence
                for evidence in proposal.evidence
                if evidence.page == page.page
                and evidence.source == "gpt-visual"
                and evidence.bbox
                and max(
                    _candidate_evidence_overlap(visual_bbox, evidence.bbox),
                    _candidate_evidence_overlap(evidence.bbox, visual_bbox),
                )
                >= 0.40
            ]
            if visual_bbox
            else []
        )
        visually_grounded = bool(visual_evidence)
        if not visually_grounded and not any(
            label in {"header_image", "footer_image", "image"} for label in labels
        ):
            return "text"
        if (
            visually_grounded
            and not any(label in {"header_image", "footer_image", "image"} for label in labels)
            and len(source_blocks) == 1
            and max(evidence.bbox[3] - evidence.bbox[1] for evidence in visual_evidence) < 0.04
        ):
            # A small, single OCR word without independent image-layout support
            # is ordinary running-header text, even if Sol recognizes the brand.
            return "text"
    if proposal.type == "attestation":
        text = "\n".join(block.text for block in source_blocks)
        visual_bbox = _blocks_bbox(source_blocks)
        visually_grounded = bool(
            visual_bbox
            and any(
                evidence.page == page.page
                and evidence.source == "gpt-visual"
                and evidence.bbox
                and max(
                    _candidate_evidence_overlap(visual_bbox, evidence.bbox),
                    _candidate_evidence_overlap(evidence.bbox, visual_bbox),
                )
                >= 0.40
                for evidence in proposal.evidence
            )
        )
        if not visually_grounded and not re.search(
            r"\b(?:attest(?:ation)?|electronically\s+(?:signed|authenticated)|"
            r"signature|signed\s+by)\b|\[(?:E-)?SIGNED\]|\[ILLEGIBLE_SIGNATURE\]",
            text,
            re.IGNORECASE,
        ):
            return "text"
    if proposal.type in {"figure", "scan_code"}:
        visual_bbox = _blocks_bbox(source_blocks)
        if not visual_bbox or not any(
            evidence.page == page.page
            and evidence.source == "gpt-visual"
            and evidence.bbox
            and max(
                _candidate_evidence_overlap(visual_bbox, evidence.bbox),
                _candidate_evidence_overlap(evidence.bbox, visual_bbox),
            )
            >= 0.40
            for evidence in proposal.evidence
        ):
            return "text"
    return proposal.type


def _semantic_source_groups(
    chunk_type: str, source_blocks: list[Any]
) -> list[tuple[str, list[Any]]]:
    """Keep independently grounded attestations as independently addressable objects."""
    if chunk_type != "attestation":
        return [(chunk_type, source_blocks)]
    attestation_blocks = [
        block
        for block in source_blocks
        if any(
            marker in block.text.upper()
            for marker in ("[E-SIGNED]", "[SIGNED]", "[ILLEGIBLE_SIGNATURE]")
        )
    ]
    if len(attestation_blocks) > 1 and len(attestation_blocks) == len(source_blocks):
        return [("attestation", [block]) for block in attestation_blocks]
    return [(chunk_type, source_blocks)]


def _semantic_markdown(proposal: CloudSemanticRegion, source_blocks: list[Any]) -> str:
    texts = [block.text.strip() for block in source_blocks if block.text.strip()]
    if proposal.type == "heading":
        return f"{'#' * proposal.heading_level} {' '.join(texts)}"
    if proposal.type == "list":
        return "\n".join(
            text if text.startswith(("- ", "* ")) else f"- {text.lstrip('• ')}" for text in texts
        )
    separator = " " if proposal.join_style == "space" else "\n"
    return separator.join(texts)


def _blocks_bbox(blocks: list[Any]) -> list[float] | None:
    boxes = [block.bbox for block in blocks if block.bbox]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _candidate_evidence_overlap(candidate: list[float], evidence: list[float]) -> float:
    intersection = max(0.0, min(candidate[2], evidence[2]) - max(candidate[0], evidence[0])) * max(
        0.0, min(candidate[3], evidence[3]) - max(candidate[1], evidence[1])
    )
    area = max(1e-9, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
    return intersection / area


def _redaction_block(page: Any, candidate: Any, block_id: str) -> Block:
    left, top, right, bottom = candidate.bbox
    polygon = [
        [left * page.width, top * page.height],
        [right * page.width, top * page.height],
        [right * page.width, bottom * page.height],
        [left * page.width, bottom * page.height],
    ]
    return Block(
        id=block_id,
        page=page.page,
        type="key_value" if candidate.label_block_id else "paragraph",
        text="[REDACTED]",
        bbox=list(candidate.bbox),
        polygon=polygon,
        source="gpt-visual",
    )


def _insert_redaction_block(page: Any, block: Block, label_block_id: str | None) -> None:
    if label_block_id:
        for index, existing in enumerate(page.blocks):
            if existing.id == label_block_id:
                page.blocks.insert(index + 1, block)
                return
    page.blocks.append(block)


def _signature_attestation(page: Any | None, block: Any | None) -> Any | None:
    if (
        page is None
        or block is None
        or block.bbox is None
        or block.ocr_score is None
        or block.ocr_score >= LOW_CONFIDENCE_THRESHOLD
    ):
        return None
    center_x = (block.bbox[0] + block.bbox[2]) / 2
    center_y = (block.bbox[1] + block.bbox[3]) / 2
    if not any(
        region.label.lower() == "image"
        and region.bbox[0] <= center_x <= region.bbox[2]
        and region.bbox[1] <= center_y <= region.bbox[3]
        for region in page.layout_regions
    ):
        return None
    candidates = []
    for candidate in page.blocks:
        if candidate.bbox is None or not _ELECTRONIC_ATTESTATION.search(candidate.text.strip()):
            continue
        gap = block.bbox[1] - candidate.bbox[3]
        if -0.005 <= gap <= 0.05 and abs(block.bbox[0] - candidate.bbox[0]) <= 0.08:
            candidates.append((abs(gap), candidate))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _signature_visual_evidence(block: Any, evidence: list[CloudEvidence]) -> list[CloudEvidence]:
    if block is None or block.bbox is None:
        return []
    return [
        item
        for item in evidence
        if item.source == "gpt-visual"
        and item.bbox
        and _candidate_evidence_overlap(block.bbox, item.bbox) >= 0.40
    ]


def _apply_signature_semantics(
    refined_blocks: dict[str, Block],
    signature_block: Block,
    attestation_block: Block,
    visual_evidence: list[CloudEvidence],
    records: list[RefinementRecord],
    *,
    add_signature_record: bool,
) -> None:
    refined_signature = refined_blocks[signature_block.id]
    refined_signature.text = _SIGNATURE_MARKDOWN
    refined_signature.type = "paragraph"
    evidence_refs = [
        EvidenceRef.model_validate(item.model_dump(exclude={"chunk_id"}))
        for item in visual_evidence
    ]
    if add_signature_record:
        records.append(
            RefinementRecord(
                page=signature_block.page,
                block_id=signature_block.id,
                status="accepted",
                proposed_text=_SIGNATURE_MARKDOWN,
                proposed_type="paragraph",
                evidence=evidence_refs,
                reason=(
                    "Semantic signature markers preserve a locally grounded signature image "
                    "without transcribing illegible handwriting."
                ),
            )
        )
    refined_attestation = refined_blocks[attestation_block.id]
    proposed_attestation = refined_attestation.text
    if not proposed_attestation.startswith("[E-SIGNED]"):
        proposed_attestation = f"[E-SIGNED]\n{proposed_attestation}"
        refined_attestation.text = proposed_attestation
        records.append(
            RefinementRecord(
                page=attestation_block.page,
                block_id=attestation_block.id,
                status="accepted",
                proposed_text=proposed_attestation,
                proposed_type=refined_attestation.type,
                evidence=[
                    EvidenceRef(
                        page=attestation_block.page,
                        block_id=attestation_block.id,
                        quote=attestation_block.text,
                        source="rapidocr",
                    ),
                    *evidence_refs,
                ],
                reason=(
                    "The exact electronic-attestation line is adjacent to the visually reviewed "
                    "signature image."
                ),
            )
        )


def _critical_token_change(original: str, proposed: str) -> bool:
    if original == proposed:
        return False
    critical = re.compile(
        r"(?:\d|[<>]=?|\[(?:E-)?SIGNED\]|ILLEGIBLE_SIGNATURE|REDACTED)", re.IGNORECASE
    )
    return bool(critical.search(original) or critical.search(proposed))


def render_result_markdown(local: LocalParseResult) -> str:
    """Rebuild derived Markdown from immutable OCR, GPT refinements, and checkboxes."""
    if not local.cloud_output:
        return document_markdown_with_checkboxes(local.pages, local.checkboxes)
    cloud = CloudResult.model_validate(local.cloud_output)
    markdown, _, _, _ = _apply_refinements(
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
    if evidence.source == "rapidocr" and evidence.block_id:
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
    layout_resource: PPDocLayoutResource | None = None,
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
    local = LocalParseResult(
        document_metadata={},
        selected_pages=sorted(selected),
        pages=parsed,
        markdown="",
        engine=EngineProvenance(
            name="RapidOCR", version=_version("rapidocr"), device=resource.device
        ),
        timings={"ocr_seconds": sum(page.ocr_seconds for page in parsed)},
    )
    apply_document_layout(local, pages, layout_resource or create_pp_doclayout_engine())
    _detect_local_redactions(local)
    parsed = local.pages
    for page in parsed:
        page.visual_review_regions = plan_visual_review_regions(
            page, forced=page.page in request.forced_cloud_pages
        )
    local_markdown = document_markdown(parsed)
    full_context_pages = (
        {page.page for page in parsed}
        if request.mode is ProcessingMode.HIGH_ACCURACY
        else {
            page.page
            for page in parsed
            if any(region.page_wide for region in page.visual_review_regions)
        }
    )
    warnings = [resource.warning] if resource.warning else []
    metadata: dict[str, Any] = {
        "file_name": request.file_name,
        "mime_type": document.mime_type,
        "page_count": len(document.pages),
        "processed_pages": sorted(selected),
        "cloud_image_pages": sorted(page.page for page in parsed),
        "mode": request.mode.value,
        "ocr_device": resource.device,
        "ocr_seconds": sum(page.ocr_seconds for page in parsed),
        "rapidocr_version": _version("rapidocr"),
        "openai_model": "gpt-6-sol",
        "reasoning_effort": "medium",
        "balanced_thresholds_calibrated": False,
    }
    try:
        cloud, usage = cloud_refiner.refine(
            parsed,
            full_context_pages,
            {capability.value for capability in request.capabilities},
            request.allowed_classes,
            request.extraction_schema,
        )
        if usage.call_count < 1:
            raise RuntimeError("gpt-6-sol returned no API call record.")
        refined_markdown, refinements, refinement_warnings, _ = _apply_refinements(
            parsed, cloud, {page.page for page in parsed}
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
            f"gpt-6-sol refinement failed; no extraction was completed: {exc}"
        ) from exc


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
