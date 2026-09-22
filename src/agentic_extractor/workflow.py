"""Constrained, evidence-gated document workflow.

Responsible for the ADE-style state machine (VALIDATED -> ... -> ACCEPTED |
REVIEW_REQUIRED | FAILED), turning raw GPT ("cloud") proposals into
`AgentWorkflowResult` only after grounding every proposal against immutable
RapidOCR/PP-DocLayoutV3 evidence, and the bounded, targeted GPT repair path
for REVIEW_REQUIRED output. Must not accept a classification, section, split,
extracted field, or checkbox that cannot be traced back to a real block,
chunk, or reviewed image region (see `_evidence`, `_checkbox_risks`) - GPT
output is treated as untrusted until grounded here. Must not mutate raw OCR
`Block`/`PageParse` evidence in place; corrections are recorded as separate
audit objects (`FieldCorrection`, `CheckboxCorrection`).
Callers: `api.py` (HTTP job execution) and the Streamlit `app_pages/`
extraction pages, both via `run_agent_workflow` / `run_workflow_from_parse`.
Consumers of this module's output: `artifacts.py` (renders `AgentWorkflowResult`
into the manifest/ZIP) and `pipeline.py` (supplies the `LocalParseResult` this
module refines).
"""

from __future__ import annotations

import copy
import io
import json
import re
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from jsonschema import Draft202012Validator, FormatChecker
from PIL import Image
from pydantic import BaseModel, Field

from agentic_extractor.checkbox_vision import (
    is_credible_checkbox_candidate,
    is_visual_checkbox_candidate,
)
from agentic_extractor.costs import MODEL_NAME, aggregate_usage
from agentic_extractor.layout import (
    PPDocLayoutResource,
    PPDocLayoutSetupError,
    apply_document_layout,
    create_pp_doclayout_engine,
)
from agentic_extractor.models import (
    Capability,
    CheckboxCorrection,
    CheckboxRecord,
    CheckboxState,
    DocumentRequest,
    LocalCheckboxCandidate,
    VisualReviewRegion,
)
from agentic_extractor.ocr import (
    LocalParseResult,
    OCRResource,
    create_rapidocr_engine,
    ocr_page,
)
from agentic_extractor.openai_refiner import (
    CheckboxVerification,
    CloudCheckbox,
    CloudEvidence,
    CloudResult,
    OpenAIRefiner,
)
from agentic_extractor.parse import build_layout_chunks, document_markdown
from agentic_extractor.pipeline import (
    Refiner,
    _apply_refinements,
    _process_local_document,
    refine_local_parse,
    refine_markdown_workflows,
    render_result_markdown,
)
from agentic_extractor.table_structure import table_review_bbox, table_review_block_ids
from agentic_extractor.timing import record_stage_timing, stage_timing_summary


class WorkflowState(StrEnum):
    VALIDATED = "VALIDATED"
    NORMALIZED = "NORMALIZED"
    PARSED = "PARSED"
    CLASSIFIED = "CLASSIFIED"
    SECTIONED = "SECTIONED"
    SPLIT = "SPLIT"
    EXTRACTED = "EXTRACTED"
    ACCEPTED = "ACCEPTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"


class RapidOCRSetupError(RuntimeError):
    """RapidOCR could not produce a local first pass."""


_MIN_CHECKBOX_VERIFICATION_CONFIDENCE = 0.90


# Append-only audit record for one state transition. `evaluate_workflow` emits
# one of these per stage (see its nested `event()` helper) regardless of
# outcome, so `AgentWorkflowResult.events` is a complete trace even for a
# FAILED or REVIEW_REQUIRED run, not just successful ones.
class WorkflowEvent(BaseModel):
    state: WorkflowState
    action: str
    provider: Literal["system", "RapidOCR", "PP-DocLayoutV3", "gpt-6-sol"] = "system"
    reason: str
    elapsed_seconds: float = 0
    token_impact: int = 0
    cost_impact_usd: float | None = 0.0
    warnings: list[str] = Field(default_factory=list)


class GroundedClassification(BaseModel):
    label: str
    scope: Literal["page", "document"]
    page_start: int
    page_end: int
    evidence: list[CloudEvidence]
    status: Literal["verified", "abstained"] = "verified"
    review_reason: str | None = None


class GroundedSection(BaseModel):
    section_id: str
    parent_section_id: str | None = None
    title: str
    level: int
    page_start: int
    page_end: int
    source_block_ids: list[str]
    source_chunk_ids: list[str]


class GroundedSplit(BaseModel):
    name: str
    page_start: int
    page_end: int
    source_pages: list[int]
    label: str = "unknown"
    overridden: bool = False
    override_reason: str | None = None
    evidence: list[CloudEvidence] = Field(default_factory=list)


class ConfidenceRecord(BaseModel):
    engine: Literal["RapidOCR", "gpt-6-sol"]
    value: float | None = None
    kind: Literal["recognition", "model_asserted"]
    calibrated: bool = False


class ReviewItem(BaseModel):
    id: str
    stage: Literal["parse", "classify", "section", "split", "extract", "checkbox", "validate"]
    code: str
    message: str
    retryable: bool = True
    pages: list[int] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    attempt: int = 1


class ValidatedField(BaseModel):
    path: str
    value: Any = None
    normalized_value: Any = None
    source_text: str | None = None
    engine_provenance: list[str] = Field(default_factory=list)
    status: Literal["verified", "uncertain", "abstained", "invalid"]
    confidence: float | None = None
    source_page: int | None = None
    source_block_id: str | None = None
    source_chunk_id: str | None = None
    bbox: list[float] | None = None
    polygon: list[list[float]] | None = None
    validation_errors: list[str] = Field(default_factory=list)
    validation_outcome: Literal["passed", "failed", "not_validated"] = "not_validated"
    review_reason: str | None = None
    abstention_reason: str | None = None
    evidence: list[CloudEvidence] = Field(default_factory=list)
    confidence_by_engine: list[ConfidenceRecord] = Field(default_factory=list)
    requires_review: bool = False


class FieldCorrection(BaseModel):
    path: str
    value: Any
    previous_value: Any = None
    reason: str
    timestamp: str
    actor: Literal["user"] = "user"
    action: Literal["accept", "correct"]


class AgentWorkflowResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    current_state: WorkflowState
    events: list[WorkflowEvent]
    classifications: list[GroundedClassification] = Field(default_factory=list)
    sections: list[GroundedSection] = Field(default_factory=list)
    splits: list[GroundedSplit] = Field(default_factory=list)
    extracted_fields: list[ValidatedField] = Field(default_factory=list)
    review_required: list[str] = Field(default_factory=list)
    review_items: list[ReviewItem] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    schema_version: str | None = None
    corrections: list[FieldCorrection] = Field(default_factory=list)
    checkboxes: list[CheckboxRecord] = Field(default_factory=list)
    checkbox_corrections: list[CheckboxCorrection] = Field(default_factory=list)
    business_rules: list[dict[str, Any]] = Field(default_factory=list)

    def manifest(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def run_agent_workflow(
    request: DocumentRequest,
    *,
    ocr_resource: OCRResource | None = None,
    layout_resource: PPDocLayoutResource | None = None,
    refiner: Refiner | None = None,
    initialization_timings: dict[str, float] | None = None,
) -> tuple[LocalParseResult, AgentWorkflowResult]:
    """Run required local parsing and GPT refinement, then deterministic adjudication.

    This is the canonical FULL entry point: it always initializes RapidOCR and
    PP-DocLayoutV3 and re-parses `request` from scratch. To refine/adjudicate a
    `LocalParseResult` that has already been parsed (e.g. the API's schema-extraction
    endpoint), call `run_workflow_from_parse` directly instead - it must not
    reach this function, or OCR/layout would silently run a second time.
    """
    started = time.perf_counter()
    cloud_refiner = refiner or OpenAIRefiner()
    configuration_started = time.perf_counter()
    cloud_refiner.validate_configuration()
    configuration_seconds = time.perf_counter() - configuration_started
    rapidocr_initialization_started = time.perf_counter()
    try:
        resource = ocr_resource or create_rapidocr_engine()
    except Exception as exc:
        raise RapidOCRSetupError(
            "RapidOCR could not initialize. Run 'uv sync --all-groups' and verify ONNX "
            "Runtime setup."
        ) from exc
    rapidocr_initialization_seconds = time.perf_counter() - rapidocr_initialization_started
    local = _process_local_document(request, ocr_resource=resource)
    for key, elapsed in (initialization_timings or {}).items():
        record_stage_timing(local.timings, key, elapsed)
    record_stage_timing(local.timings, "configuration_seconds", configuration_seconds)
    record_stage_timing(
        local.timings, "rapidocr_initialization_seconds", rapidocr_initialization_seconds
    )
    local.ocr_attempts = [_ocr_attempt(page, 1) for page in local.pages]
    retry_started = time.perf_counter()
    _retry_failed_pages_once(local, resource)
    record_stage_timing(local.timings, "ocr_retry_seconds", time.perf_counter() - retry_started)
    if local.failed_pages and len(local.failed_pages) == len(local.selected_pages):
        raise RapidOCRSetupError(
            "RapidOCR failed on every selected page. Verify models and ONNX Runtime setup."
        )
    try:
        layout_initialization_started = time.perf_counter()
        active_layout = layout_resource or create_pp_doclayout_engine()
        record_stage_timing(
            local.timings,
            "layout_initialization_seconds",
            time.perf_counter() - layout_initialization_started,
        )
        apply_document_layout(local, None, active_layout)
    except PPDocLayoutSetupError:
        raise
    except Exception as exc:
        raise PPDocLayoutSetupError(
            "PP-DocLayoutV3 could not process every selected page. Run "
            "'uv sync --project tools/pp_doclayout --locked', verify the Paddle runtime, "
            "and retry."
        ) from exc
    workflow_started = time.perf_counter()
    result, workflow = run_workflow_from_parse(
        local, request, cloud_refiner, validate_configuration=False
    )
    result.timings["workflow_seconds"] = time.perf_counter() - workflow_started
    result.timings["total_seconds"] = time.perf_counter() - started
    result.adaptive_processing["stage_timing"] = stage_timing_summary(result.timings)
    return result, workflow


def run_workflow_from_parse(
    local: LocalParseResult,
    request: DocumentRequest,
    refiner: Refiner | None = None,
    *,
    validate_configuration: bool = True,
) -> tuple[LocalParseResult, AgentWorkflowResult]:
    """Run GPT analysis and adjudication on an existing canonical Parse result.

    Must NOT re-run RapidOCR or PP-DocLayoutV3: `local` is trusted as already
    parsed. This is what lets schema extraction (`api.py`'s extract endpoint)
    reuse a prior Parse without paying for OCR again - accidentally calling
    `_process_local_document` or similar from here would double engine cost
    and break that reuse contract.
    """
    started = time.perf_counter()
    cloud_refiner = refiner or OpenAIRefiner()
    if validate_configuration:
        configuration_started = time.perf_counter()
        cloud_refiner.validate_configuration()
        record_stage_timing(
            local.timings,
            "configuration_seconds",
            time.perf_counter() - configuration_started,
        )
    reused_parse = bool(local.cloud_output)
    ledger_start = len(getattr(cloud_refiner, "request_usage", []))
    prior_usage = local.usage if local.usage.call_count else None
    if local.cloud_output and request.capabilities - {Capability.PARSE}:
        local, _ = refine_markdown_workflows(local, request, cloud_refiner)
    elif not local.cloud_output:
        local = refine_local_parse(local, request, cloud_refiner, defer_markdown_workflows=True)
    cloud = CloudResult.model_validate(local.cloud_output) if local.cloud_output else None
    if cloud is not None:
        # Only send a checkbox candidate for the paid Sol crop-verification call
        # if the *page* already has independent local (OpenCV+OCR) corroboration;
        # this bounds verification cost and keeps ungrounded Sol-only guesses out
        # of the crop-review request entirely (they still surface as review items
        # via `_checkbox_risks`, they just never reach `verify_checkboxes`).
        credible_pages = {
            page.page
            for page in local.pages
            if any(
                is_credible_checkbox_candidate(page, candidate)
                for candidate in page.local_checkbox_candidates
            )
        }
        candidates = [
            candidate
            for candidate in _checkbox_candidates(local, cloud)
            if candidate.page in credible_pages
        ]
        verify = getattr(cloud_refiner, "verify_checkboxes", None)
        if candidates and callable(verify):
            verification_started = time.perf_counter()
            page_map = {page.page: page for page in local.pages}
            # One bounded Sol call per page (not one big batch): keeps each crop-
            # verification request small, and a single page's failure (caught below)
            # only removes that page's checkboxes from automation - it does not
            # abort verification for the rest of the document.
            for page_number in dict.fromkeys(candidate.page for candidate in candidates):
                page_candidates = [
                    candidate for candidate in candidates if candidate.page == page_number
                ]
                try:
                    verified, verification_usage = verify([page_map[page_number]], page_candidates)
                    local.usage = aggregate_usage([local.usage, verification_usage])
                    batch_verifications = [
                        item.model_dump(mode="json") for item in verified.verifications
                    ]
                    local.checkbox_verifications.extend(batch_verifications)
                    local.cloud_attempts.append(
                        {
                            "attempt": len(local.cloud_attempts) + 1,
                            "purpose": "checkbox_verification",
                            "pages": [page_number],
                            "output": {"verifications": batch_verifications},
                        }
                    )
                except Exception as exc:
                    local.warnings.append(
                        f"Bounded checkbox verification failed for page {page_number}; human "
                        f"review is required: {type(exc).__name__}."
                    )
            record_stage_timing(
                local.timings,
                "checkbox_verification_seconds",
                time.perf_counter() - verification_started,
            )
    workflow = evaluate_workflow(local, request, cloud)
    inspected: set[str] = set()
    downstream_pending = not reused_parse and bool(request.capabilities - {Capability.PARSE})
    if cloud is not None and not reused_parse and cloud.visual_objects:
        from agentic_extractor.rich_document import prepare_inspection_images

        prepare_inspection_images(local.pages, request.file_bytes, cloud.visual_objects)
    for repair_round in range(1, 3):
        before = tuple(workflow.review_required)
        if cloud is not None and not reused_parse:
            _verify_visual_round(local, cloud, cloud_refiner, inspected, repair_round)
            local.markdown = render_result_markdown(local)
            workflow = evaluate_workflow(local, request, cloud)
            if (
                repair_round == 1
                and tuple(workflow.review_required) != before
                and any(item.id not in inspected for item in cloud.visual_objects)
            ):
                continue
        if downstream_pending and cloud is not None:
            local, cloud = refine_markdown_workflows(local, request, cloud_refiner, cloud)
            workflow = evaluate_workflow(local, request, cloud)
            downstream_pending = False
        retryable = [
            item
            for item in workflow.review_items
            if item.retryable and item.stage != "checkbox" and item.source_ids
        ]
        optional_capabilities = {capability.value for capability in request.capabilities} - {
            "Parse"
        }
        markdown_repair = getattr(cloud_refiner, "refine_markdown", None)
        repair = (
            markdown_repair
            if optional_capabilities and callable(markdown_repair)
            else getattr(cloud_refiner, "repair", None)
        )
        repair_scope = (
            _object_repair_scope(local, request, cloud, retryable) if cloud is not None else None
        )
        if (
            workflow.current_state is WorkflowState.REVIEW_REQUIRED
            and retryable
            and cloud is not None
            and callable(repair)
            and repair_scope is not None
        ):
            repair_started = time.perf_counter()
            repair_markdown, repair_pages, repair_capabilities, repair_schema, repair_prior = (
                repair_scope
            )
            try:
                issues = [item.model_dump(mode="json") for item in retryable]
                if repair is markdown_repair:
                    repaired, repair_usage = repair(
                        repair_markdown,
                        repair_pages,
                        repair_capabilities,
                        request.allowed_classes,
                        repair_schema,
                        issues=issues,
                        prior=repair_prior,
                    )
                else:
                    image_pages = {page.page for page in repair_pages if page.visual_review_regions}
                    repaired, repair_usage = repair(
                        repair_pages,
                        image_pages,
                        repair_capabilities,
                        request.allowed_classes,
                        repair_schema,
                        issues,
                        repair_prior,
                    )
                merged = _merge_repair(
                    cloud, repaired, request.capabilities, repair_items=retryable
                )
                local.usage = aggregate_usage([local.usage, repair_usage])
                local.cloud_output = merged.model_dump(mode="json")
                local.cloud_attempts.append(
                    {
                        "attempt": len(local.cloud_attempts) + 1,
                        "purpose": "repair",
                        "output": local.cloud_output,
                    }
                )
                (
                    local.markdown,
                    local.refinements,
                    refinement_warnings,
                    table_reviews,
                ) = _apply_refinements(local.pages, merged, set(local.cloud_image_pages))
                local.document_metadata["table_reviews"] = table_reviews
                local.warnings.extend(refinement_warnings)
                repaired_workflow = evaluate_workflow(local, request, merged)
                repaired_workflow.events = [
                    *workflow.events,
                    WorkflowEvent(
                        state=WorkflowState.REVIEW_REQUIRED,
                        action="request_deeper_gpt_refinement",
                        provider=MODEL_NAME,
                        reason=f"Repair round {repair_round} targeted {len(retryable)} object(s).",
                        token_impact=repair_usage.total_tokens or 0,
                        cost_impact_usd=repair_usage.total_cost_usd,
                    ),
                    *repaired_workflow.events,
                ]
                workflow = repaired_workflow
                cloud = merged
            except Exception as exc:
                message = (
                    f"Bounded GPT repair failed; human review is required: {type(exc).__name__}."
                )
                workflow.review_required.append(message)
                workflow.review_items.append(
                    ReviewItem(
                        id=f"review-{len(workflow.review_items) + 1}",
                        stage="validate",
                        code="REPAIR_FAILED",
                        message=message,
                        retryable=False,
                        attempt=repair_round + 1,
                    )
                )
                workflow.events.append(
                    WorkflowEvent(
                        state=WorkflowState.REVIEW_REQUIRED,
                        action="deeper_gpt_refinement_failed",
                        provider=MODEL_NAME,
                        reason=message,
                    )
                )
            record_stage_timing(
                local.timings, "object_repair_seconds", time.perf_counter() - repair_started
            )
        if workflow.current_state is not WorkflowState.REVIEW_REQUIRED:
            break
        if tuple(workflow.review_required) == before:
            break
    finalization_started = time.perf_counter()
    ledger = getattr(cloud_refiner, "request_usage", [])[ledger_start:]
    if ledger:
        recorded_count = sum(item.call_count for item in ledger)
        if (
            recorded_count + (prior_usage.call_count if prior_usage else 0)
            != local.usage.call_count
        ):
            local.usage = aggregate_usage(([prior_usage] if prior_usage else []) + ledger)
    local.workflow_manifest = workflow.manifest()
    local.checkboxes = workflow.checkboxes
    local.checkbox_corrections = workflow.checkbox_corrections
    local.markdown = render_result_markdown(local)
    local.warnings = list(dict.fromkeys(local.warnings))
    record_stage_timing(
        local.timings,
        "result_finalization_seconds",
        time.perf_counter() - finalization_started,
    )
    local.timings["workflow_seconds"] = time.perf_counter() - started
    local.adaptive_processing["stage_timing"] = stage_timing_summary(local.timings)
    return local, workflow


def _verify_visual_round(local, cloud, refiner, inspected: set[str], repair_round: int) -> None:
    from agentic_extractor.rich_document import proposal_digest, visual_status

    verify = getattr(refiner, "verify_visual_objects", None)
    if not callable(verify):
        return
    pages = {page.page: page for page in local.pages}
    audits = [audit for page in local.pages for audit in page.visual_audits]
    objects = [
        item
        for item in cloud.visual_objects
        if item.id not in inspected
        and item.page in pages
        and visual_status(item, audits)[0] == "pending"
    ][:8]
    if not objects:
        return
    inspected.update(item.id for item in objects)
    try:
        result, usage = verify(local.pages, objects)
        local.usage = aggregate_usage([local.usage, usage])
        ids = [item.id for item in result.decisions]
        for item in objects:
            decision = next((value for value in result.decisions if value.id == item.id), None)
            supported = (
                decision is not None
                and ids.count(item.id) == 1
                and decision.supported
                and decision.transcription.strip() == item.content.strip()
            )
            pages[item.page].visual_audits.append(
                {
                    "id": item.id,
                    "digest": proposal_digest(item),
                    "status": "model_verified" if supported else "pending",
                    "content": item.content,
                    "reason": decision.reason if decision else "Missing crop verification.",
                    "round": repair_round,
                    "provider": MODEL_NAME,
                }
            )
        local.cloud_attempts.append(
            {
                "attempt": len(local.cloud_attempts) + 1,
                "purpose": "visual_verification",
                "round": repair_round,
                "output": result.model_dump(mode="json"),
            }
        )
    except Exception as exc:
        local.warnings.append(f"Visual verification requires human review: {type(exc).__name__}.")


def _merge_repair(
    original: CloudResult,
    repaired: CloudResult,
    capabilities: set[Capability],
    *,
    repair_items: list[ReviewItem] | None = None,
) -> CloudResult:
    """Replace only identified repair objects; raw OCR remains outside this layer."""
    # Targeted repair invariant: only items explicitly targeted by review_items
    # are replaced; all unflagged extractions and accepted structures are preserved.
    merged = original.model_copy(deep=True)
    targets = _repair_targets(repair_items or [])
    merged.reviewed_pages = list(
        dict.fromkeys([*original.reviewed_pages, *repaired.reviewed_pages])
    )
    merged.warnings.extend(repaired.warnings)
    if repaired.refined_markdown and not repair_items:
        merged.refined_markdown = repaired.refined_markdown
    if repaired.refinements:
        repaired_refinements = {(item.page, item.block_id): item for item in repaired.refinements}
        merged.refinements = [
            repaired_refinements.pop((item.page, item.block_id), item)
            for item in original.refinements
        ]
        merged.refinements.extend(repaired_refinements.values())
    if repaired.table_reviews:
        repaired_tables = {
            (review.page, review.table_id): review for review in repaired.table_reviews
        }
        merged.table_reviews = [
            repaired_tables.pop((review.page, review.table_id), review)
            for review in original.table_reviews
        ]
        merged.table_reviews.extend(repaired_tables.values())
    if repaired.semantic_regions:
        repaired_pages = {item.page for item in repaired.semantic_regions}
        merged.semantic_regions = [
            item for item in original.semantic_regions if item.page not in repaired_pages
        ]
        merged.semantic_regions.extend(repaired.semantic_regions)
    if Capability.CLASSIFY in capabilities and targets.get("classify"):
        repaired_items = (
            repaired.classifications
            if len(repaired.classifications) <= len(targets["classify"])
            else []
        )
        merged.classifications = [
            item for item in original.classifications if item.label not in targets["classify"]
        ]
        merged.classifications.extend(repaired_items)
    if Capability.SECTION in capabilities and targets.get("section"):
        repaired_items = (
            repaired.sections if len(repaired.sections) <= len(targets["section"]) else []
        )
        merged.sections = [
            item for item in original.sections if item.title not in targets["section"]
        ]
        merged.sections.extend(repaired_items)
    if Capability.SPLIT in capabilities and targets.get("split"):
        repaired_items = repaired.splits if len(repaired.splits) <= len(targets["split"]) else []
        merged.splits = [item for item in original.splits if item.name not in targets["split"]]
        merged.splits.extend(repaired_items)
    if Capability.EXTRACT in capabilities:
        field_targets = targets.get("extract", set())
        repaired_by_path = {
            item.path: item for item in repaired.extracted_fields if item.path in field_targets
        }
        merged.extracted_fields = [
            repaired_by_path.get(item.path, item) for item in original.extracted_fields
        ]
        merged.extracted_fields.extend(
            item
            for path, item in repaired_by_path.items()
            if path not in {field.path for field in original.extracted_fields}
        )
    return merged


def _repair_targets(items: list[ReviewItem]) -> dict[str, set[str]]:
    targets: dict[str, set[str]] = {}
    for item in items:
        if item.source_ids:
            targets.setdefault(item.stage, set()).update(item.source_ids)
    return targets


def _cloud_for_objects(cloud: CloudResult, targets: dict[str, set[str]]) -> CloudResult:
    """Project prior derived proposals to exact validation objects."""
    parse_ids = targets.get("parse", set())
    return CloudResult(
        refined_markdown="",
        reviewed_pages=[],
        refinements=[item for item in cloud.refinements if item.block_id in parse_ids],
        warnings=[],
        classifications=[
            item for item in cloud.classifications if item.label in targets.get("classify", set())
        ],
        sections=[item for item in cloud.sections if item.title in targets.get("section", set())],
        splits=[item for item in cloud.splits if item.name in targets.get("split", set())],
        extracted_fields=[
            item for item in cloud.extracted_fields if item.path in targets.get("extract", set())
        ],
        table_reviews=[
            item for item in cloud.table_reviews if item.table_id in targets.get("parse", set())
        ],
        semantic_regions=[
            item
            for item in cloud.semantic_regions
            if item.id in parse_ids or bool(set(item.source_block_ids) & parse_ids)
        ],
    )


def _object_repair_scope(
    local: LocalParseResult,
    request: DocumentRequest,
    cloud: CloudResult,
    items: list[ReviewItem],
) -> tuple[str, list[Any], set[str], dict[str, Any] | None, CloudResult] | None:
    """Build a Markdown/evidence packet containing only identified repair objects."""
    targets = _repair_targets(items)
    if not targets:
        return None
    prior = _cloud_for_objects(cloud, targets)
    direct_blocks: set[str] = set(targets.get("parse", set()))
    direct_chunks: set[str] = set()
    targeted_tables: dict[int, list[Any]] = {}
    table_ids = targets.get("parse", set())
    for page in local.pages:
        for table in page.table_structures:
            if table.id not in table_ids:
                continue
            targeted_tables.setdefault(page.page, []).append(table)
            direct_blocks.update(table_review_block_ids(page, table))
    for evidence in _cloud_evidence(prior):
        if evidence.block_id:
            direct_blocks.add(evidence.block_id)
        if evidence.chunk_id:
            direct_chunks.add(evidence.chunk_id)
    for section in prior.sections:
        direct_blocks.update(section.source_block_ids)
        direct_chunks.update(section.source_chunk_ids)

    terms = _repair_query_terms(items, request, targets)
    candidates: list[tuple[int, Any, Any]] = []
    for page in local.pages:
        for chunk in page.chunks or build_layout_chunks(page.blocks):
            score = (
                100
                if chunk.id in direct_chunks or direct_blocks & set(chunk.source_block_ids)
                else 0
            )
            lowered = chunk.text.casefold()
            score += sum(lowered.count(term) for term in terms)
            candidates.append((score, page, chunk))
    selected: list[tuple[Any, Any]] = []
    used_ids: set[str] = set()
    used_characters = 0
    for score, page, chunk in sorted(candidates, key=lambda item: item[0], reverse=True):
        if score <= 0 or chunk.id in used_ids or len(selected) >= 8:
            continue
        size = len(chunk.markdown or chunk.text)
        if selected and used_characters + size > 12_000:
            continue
        selected.append((page, chunk))
        used_ids.add(chunk.id)
        used_characters += size
    if not selected:
        return None

    scoped_pages: list[Any] = []
    markdown_parts: list[str] = []
    for page in local.pages:
        page_chunks = [
            chunk for candidate_page, chunk in selected if candidate_page.page == page.page
        ]
        if not page_chunks:
            continue
        scoped = copy.deepcopy(page)
        scoped.chunks = page_chunks
        block_ids = {block_id for chunk in page_chunks for block_id in chunk.source_block_ids}
        scoped.blocks = [block for block in scoped.blocks if block.id in block_ids]
        scoped.layout_block_links = [
            link for link in scoped.layout_block_links if link.block_id in block_ids
        ]
        layout_ids = {link.primary_region_id for link in scoped.layout_block_links}
        scoped.layout_regions = [
            region for region in scoped.layout_regions if region.id in layout_ids
        ]
        scoped.table_structures = [
            table
            for table in scoped.table_structures
            if table.id in targets.get("parse", set())
            or any(
                cell.source_block_ids and block_ids & set(cell.source_block_ids)
                for cell in table.cells
            )
        ]
        scoped.local_checkbox_candidates = [
            item
            for item in scoped.local_checkbox_candidates
            if item.id in targets.get("checkbox", set())
        ]
        scoped.local_redaction_candidates = [
            item
            for item in scoped.local_redaction_candidates
            if item.id in targets.get("parse", set())
        ]
        scoped.visual_review_regions = [
            region
            for region in scoped.visual_review_regions
            if set(region.source_block_ids) & block_ids
            or set(region.source_redaction_ids) & targets.get("parse", set())
        ]
        for table in targeted_tables.get(page.page, []):
            source_ids = sorted(table_review_block_ids(page, table))
            scoped.visual_review_regions.append(
                VisualReviewRegion(
                    id=f"{table.id}-repair",
                    page=page.page,
                    bbox=table_review_bbox(page, table),
                    reason_codes=["table_structure_review"],
                    source_block_ids=source_ids,
                    source_layout_region_ids=[table.layout_region_id],
                )
            )
        scoped.reading_order_evidence = None
        scoped.raw_evidence = {}
        scoped.layout_raw_evidence = {}
        scoped_pages.append(scoped)
        markdown_parts.append(
            f"<!-- page: {page.page} -->\n\n"
            + "\n\n".join(chunk.markdown or chunk.text for chunk in page_chunks)
        )
    capabilities = {
        {
            "classify": "Classify",
            "section": "Section",
            "split": "Split",
            "extract": "Extract",
            "parse": "Parse",
        }[stage]
        for stage in targets
        if stage in {"classify", "section", "split", "extract", "parse"}
    }
    return (
        "\n\n".join(markdown_parts),
        scoped_pages,
        capabilities,
        _repair_schema(request.extraction_schema, targets.get("extract", set())),
        prior,
    )


def _cloud_evidence(cloud: CloudResult) -> list[CloudEvidence]:
    evidence = [item for value in cloud.classifications for item in value.evidence]
    evidence.extend(item for value in cloud.splits for item in value.evidence)
    evidence.extend(item for value in cloud.extracted_fields for item in value.evidence)
    evidence.extend(item for value in cloud.refinements for item in value.evidence)
    evidence.extend(item for value in cloud.table_reviews for item in value.evidence)
    return evidence


def _repair_query_terms(
    items: list[ReviewItem], request: DocumentRequest, targets: dict[str, set[str]]
) -> set[str]:
    values = [item.message for item in items]
    properties = (request.extraction_schema or {}).get("properties", {})
    for path in targets.get("extract", set()):
        values.append(path.replace("_", " "))
        definition = properties.get(path, {})
        values.extend(str(definition.get(key, "")) for key in ("title", "description"))
    ignored = {"field", "required", "review", "uncertain", "invalid", "abstained", "schema"}
    return {
        term
        for value in values
        for term in re.findall(r"[a-z0-9]+", value.casefold().replace("_", " "))
        if len(term) > 2 and term not in ignored
    }


def _repair_schema(schema: dict[str, Any] | None, field_targets: set[str]) -> dict[str, Any] | None:
    if not schema or not field_targets:
        return None
    scoped = copy.deepcopy(schema)
    properties = scoped.get("properties", {})
    scoped["properties"] = {
        name: definition for name, definition in properties.items() if name in field_targets
    }
    scoped["required"] = [name for name in scoped.get("required", []) if name in field_targets]
    return scoped


def _cloud_for_pages(cloud: CloudResult, pages: set[int]) -> CloudResult:
    """Project prior proposals to the pages in one bounded repair request."""

    def overlaps(page_start: int, page_end: int) -> bool:
        return any(page_start <= page <= page_end for page in pages)

    return CloudResult(
        refined_markdown="",
        reviewed_pages=[page for page in cloud.reviewed_pages if page in pages],
        refinements=[item for item in cloud.refinements if item.page in pages],
        warnings=[],
        classifications=[
            item for item in cloud.classifications if overlaps(item.page_start, item.page_end)
        ],
        sections=[item for item in cloud.sections if overlaps(item.page_start, item.page_end)],
        splits=[item for item in cloud.splits if overlaps(item.page_start, item.page_end)],
        extracted_fields=[
            item
            for item in cloud.extracted_fields
            if any(evidence.page in pages for evidence in item.evidence)
        ],
        checkboxes=[item for item in cloud.checkboxes if item.page in pages],
        table_reviews=[item for item in cloud.table_reviews if item.page in pages],
        semantic_regions=[item for item in cloud.semantic_regions if item.page in pages],
    )


def evaluate_workflow(
    local: LocalParseResult, request: DocumentRequest, cloud: CloudResult | None
) -> AgentWorkflowResult:
    """Validate proposals without mutating original OCR evidence."""
    started = time.perf_counter()
    events: list[WorkflowEvent] = []
    review: list[str] = []
    errors: list[str] = []
    selected = sorted(local.selected_pages)
    pages = {page.page: page for page in local.pages}
    blocks = {block.id: block for page in local.pages for block in page.blocks}
    chunks = {
        chunk.id: chunk
        for page in local.pages
        for chunk in (page.chunks or build_layout_chunks(page.blocks))
    }
    block_chunks = {
        block_id: chunk.id for chunk in chunks.values() for block_id in chunk.source_block_ids
    }
    visual_pages = set(local.cloud_image_pages)

    def event(
        state: WorkflowState,
        action: str,
        reason: str,
        provider: Literal["system", "RapidOCR", "PP-DocLayoutV3", "gpt-6-sol"] = "system",
    ) -> None:
        events.append(
            WorkflowEvent(
                state=state,
                action=action,
                reason=reason,
                provider=provider,
                elapsed_seconds=time.perf_counter() - started,
                token_impact=(local.usage.total_tokens or 0) if provider == MODEL_NAME else 0,
                cost_impact_usd=local.usage.total_cost_usd if provider == MODEL_NAME else 0.0,
            )
        )

    event(WorkflowState.VALIDATED, "validate", "Input and selected page range accepted.")
    event(
        WorkflowState.NORMALIZED, "normalize", "Source pages normalized without changing evidence."
    )
    event(WorkflowState.PARSED, "parse", "RapidOCR evidence preserved.", "RapidOCR")
    event(
        WorkflowState.PARSED,
        "analyze_layout",
        "PP-DocLayoutV3 region and reading-order evidence preserved.",
        "PP-DocLayoutV3",
    )
    event(
        WorkflowState.PARSED,
        "detect_table_structure",
        "Detected table cells were grounded in immutable RapidOCR blocks before GPT refinement.",
        "PP-DocLayoutV3",
    )
    if retries := local.document_metadata.get("local_retry_pages"):
        event(
            WorkflowState.PARSED,
            "retry_local_parsing",
            f"One bounded local retry was attempted for source pages {retries}.",
            "RapidOCR",
        )
    provider: Literal["system", "gpt-6-sol"] = "gpt-6-sol" if cloud else "system"
    if cloud and local.usage.call_count >= 1:
        event(
            WorkflowState.PARSED,
            "request_gpt_refinement",
            "Required bounded GPT semantic refinement completed.",
            MODEL_NAME,
        )
    else:
        errors.append("gpt-6-sol participation is required for a successful extraction.")

    if Capability.EXTRACT in request.capabilities:
        try:
            Draft202012Validator.check_schema(request.extraction_schema or {})
        except Exception as exc:
            errors.append(f"Invalid extraction schema: {exc}")

    classifications: list[GroundedClassification] = []
    if Capability.CLASSIFY in request.capabilities:
        for item in cloud.classifications if cloud else []:
            valid_evidence = [
                evidence
                for evidence in _evidence(item.evidence, pages, blocks, chunks, visual_pages)
                if item.page_start <= evidence.page <= item.page_end
            ]
            valid_range = (
                item.page_start <= item.page_end
                and item.page_start in pages
                and item.page_end in pages
            )
            valid_scope = item.scope != "page" or item.page_start == item.page_end
            if (
                item.label not in {*request.allowed_classes, "unknown"}
                or not valid_range
                or not valid_scope
                or not valid_evidence
            ):
                review.append(f"Classification {item.label!r} lacks allowed, grounded evidence.")
                continue
            data = item.model_dump(exclude={"evidence", "scope"})
            scope = item.scope or ("page" if item.page_start == item.page_end else "document")
            if item.label == "unknown":
                reason = "Classification abstained because no allowlisted class was supported."
                review.append(reason)
                classifications.append(
                    GroundedClassification(
                        **data,
                        scope=scope,
                        evidence=valid_evidence,
                        status="abstained",
                        review_reason=reason,
                    )
                )
                continue
            classifications.append(
                GroundedClassification(
                    **data,
                    scope=scope,
                    evidence=valid_evidence,
                )
            )
        if not classifications:
            review.append("Classification requires human review: no supported class proposal.")
    event(
        WorkflowState.CLASSIFIED, "classify", "Allowlist and evidence checks completed.", provider
    )

    sections: list[GroundedSection] = []
    if Capability.SECTION in request.capabilities:
        for item in cloud.sections if cloud else []:
            supplied_chunks = [
                chunk_id
                for chunk_id in item.source_chunk_ids
                if chunk_id in chunks and item.page_start <= chunks[chunk_id].page <= item.page_end
            ]
            ids = [
                block_id
                for block_id in item.source_block_ids
                if block_id in blocks and item.page_start <= blocks[block_id].page <= item.page_end
            ]
            if supplied_chunks:
                ids = list(
                    dict.fromkeys(
                        block_id
                        for chunk_id in supplied_chunks
                        for block_id in chunks[chunk_id].source_block_ids
                    )
                )
            if (
                item.level < 1
                or item.page_start > item.page_end
                or item.page_start not in pages
                or item.page_end not in pages
                or not ids
            ):
                review.append(f"Section {item.title!r} lacks valid source grounding.")
                continue
            data = item.model_dump(exclude={"source_block_ids", "source_chunk_ids"})
            parent = next(
                (
                    section.section_id
                    for section in reversed(sections)
                    if section.level < item.level
                ),
                None,
            )
            if item.level > 1 and parent is None:
                review.append(f"Section {item.title!r} has no grounded parent section.")
                continue
            sections.append(
                GroundedSection(
                    **data,
                    section_id=f"section-{len(sections) + 1}",
                    parent_section_id=parent,
                    source_block_ids=ids,
                    source_chunk_ids=(
                        supplied_chunks
                        or list(dict.fromkeys(block_chunks[block_id] for block_id in ids))
                    ),
                )
            )
        if not sections:
            review.append("Section outline requires human review.")
    event(WorkflowState.SECTIONED, "section", "Outline source references checked.", provider)

    splits = (
        _splits(
            selected,
            request.split_boundaries,
            request.split_override_reason,
            cloud,
            pages,
            blocks,
            chunks,
            visual_pages,
            review,
        )
        if Capability.SPLIT in request.capabilities
        else []
    )
    event(WorkflowState.SPLIT, "split", "Original source page references preserved.", provider)

    fields = _extract(request, cloud, pages, blocks, chunks, block_chunks, visual_pages, review)
    event(
        WorkflowState.EXTRACTED,
        "extract",
        "Schema, evidence, and business rules checked.",
        provider,
    )
    verifications = {
        item.id: item
        for item in (
            CheckboxVerification.model_validate(value) for value in local.checkbox_verifications
        )
    }
    checkboxes = _checkboxes(
        local, cloud, pages, blocks, chunks, visual_pages, verifications, review
    )
    checkbox_rule_errors = _checkbox_rule_errors(checkboxes, request.business_rules)
    if checkbox_rule_errors:
        for checkbox in checkboxes:
            if checkbox.decision_status == "automated":
                checkbox.decision_status = "review_required"
                checkbox.review_reason = "Checkbox business rule conflict."
                review.append(f"Checkbox {checkbox.id!r} requires review: business rule conflict.")
        review.extend(checkbox_rule_errors)
    if local.failed_pages:
        review.append(f"Failed pages require review: {local.failed_pages}")
    table_reviews = local.document_metadata.get("table_reviews", [])
    if isinstance(table_reviews, list):
        for item in table_reviews:
            if isinstance(item, dict) and item.get("status") != "accepted":
                review.append(f"Table {item.get('table_id')} requires review: {item.get('reason')}")
    block_reviews = local.document_metadata.get("low_confidence_block_reviews", [])
    if isinstance(block_reviews, list):
        for item in block_reviews:
            if isinstance(item, dict) and item.get("status") != "accepted":
                review.append(
                    "High Accuracy review required for low-confidence block "
                    f"{item.get('block_id')}: GPT outcome was {item.get('status')}."
                )
    if cloud is not None:
        from agentic_extractor.rich_document import validate_links, visual_issues

        review.extend(
            visual_issues(
                local.pages,
                cloud.visual_objects,
                [audit for page in local.pages for audit in page.visual_audits],
            )
        )
        review.extend(validate_links(local.pages, cloud.document_links))
    event(WorkflowState.VALIDATED, "validate_output", "Deterministic output validation completed.")
    final = (
        WorkflowState.FAILED
        if errors
        else (WorkflowState.REVIEW_REQUIRED if review else WorkflowState.ACCEPTED)
    )
    action = (
        "fail"
        if final is WorkflowState.FAILED
        else ("require_human_review" if final is WorkflowState.REVIEW_REQUIRED else "accept")
    )
    event(final, action, errors[0] if errors else (review[0] if review else "All checks passed."))
    result = AgentWorkflowResult(
        current_state=final,
        events=events,
        classifications=classifications,
        sections=sections,
        splits=splits,
        extracted_fields=fields,
        checkboxes=checkboxes,
        review_required=review,
        review_items=_review_items(review),
        errors=errors,
        schema_version=_schema_version(request.extraction_schema),
        business_rules=copy.deepcopy(request.business_rules),
    )
    record_stage_timing(local.timings, "workflow_validation_seconds", time.perf_counter() - started)
    return result


def record_user_override(
    workflow: AgentWorkflowResult, path: str, value: Any, reason: str
) -> AgentWorkflowResult:
    """Return a new audit result; raw extraction and OCR evidence remain unchanged."""
    if not reason.strip():
        raise ValueError("A reason is required for a user override.")
    updated = workflow.model_copy(deep=True)
    field = next((item for item in updated.extracted_fields if item.path == path), None)
    if field is None:
        raise ValueError(f"Unknown extracted field: {path}")
    updated.corrections.append(
        FieldCorrection(
            path=path,
            value=value,
            previous_value=field.normalized_value,
            reason=reason.strip(),
            timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            action=("accept" if value == field.normalized_value else "correct"),
        )
    )
    updated.events.append(
        WorkflowEvent(
            state=WorkflowState.REVIEW_REQUIRED,
            action="user_override",
            reason=f"User supplied an audited correction for {path!r}.",
        )
    )
    return updated


def record_checkbox_override(
    workflow: AgentWorkflowResult, checkbox_id: str, state: CheckboxState, reason: str
) -> AgentWorkflowResult:
    """Apply an audited human checkbox decision without altering GPT or OCR evidence."""
    if not reason.strip():
        raise ValueError("A reason is required for a checkbox review decision.")
    updated = workflow.model_copy(deep=True)
    checkbox = next((item for item in updated.checkboxes if item.id == checkbox_id), None)
    if checkbox is None:
        raise ValueError(f"Unknown checkbox: {checkbox_id}")
    previous = checkbox.state
    checkbox.state = state
    checkbox.decision_status = "user_verified"
    checkbox.review_reason = None
    updated.checkbox_corrections.append(
        CheckboxCorrection(
            checkbox_id=checkbox_id,
            previous_state=previous,
            state=state,
            reason=reason.strip(),
            timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
    )
    updated.review_required = [
        message
        for message in updated.review_required
        if repr(checkbox_id) not in message and "Checkbox business rule" not in message
    ]
    updated.review_required.extend(
        _checkbox_rule_errors(updated.checkboxes, updated.business_rules)
    )
    updated.review_items = _review_items(updated.review_required)
    if not updated.review_required and not updated.errors:
        updated.current_state = WorkflowState.ACCEPTED
    updated.events.append(
        WorkflowEvent(
            state=updated.current_state,
            action="user_checkbox_review",
            reason=f"User verified checkbox {checkbox_id!r}.",
        )
    )
    return updated


def _retry_failed_pages_once(local: LocalParseResult, resource: OCRResource) -> None:
    """Retry only failed pages once; successful original evidence is never replaced."""
    attempted = list(local.failed_pages)
    if not attempted:
        return
    local.document_metadata["local_retry_pages"] = attempted
    by_page = {page.page: page for page in local.pages}
    still_failed: list[int] = []
    for number in attempted:
        original = by_page[number]
        try:
            image = Image.open(io.BytesIO(original.image_bytes)).convert("RGB")
            retried = ocr_page(resource, number, image)
            local.pages[local.pages.index(original)] = retried
            local.ocr_attempts.append(_ocr_attempt(retried, 2))
            local.page_statuses[number] = "completed_after_retry"
        except Exception as exc:
            still_failed.append(number)
            local.ocr_attempts.append(
                {
                    "page": number,
                    "attempt": 2,
                    "status": "failed",
                    "raw_evidence": {},
                    "warnings": [f"{type(exc).__name__}: {exc}"],
                    "block_ids": [],
                }
            )
            local.warnings.append(
                f"Bounded local retry for page {number} failed: {type(exc).__name__}: {exc}"
            )
    local.failed_pages = still_failed
    local.markdown = document_markdown(local.pages)


def _ocr_attempt(page: Any, attempt: int) -> dict[str, object]:
    return {
        "page": page.page,
        "attempt": attempt,
        "status": page.status,
        "cache_hit": page.ocr_cache_hit,
        "raw_evidence": copy.deepcopy(page.raw_evidence),
        "warnings": list(page.warnings),
        "block_ids": [block.id for block in page.blocks],
        "ocr_seconds": page.ocr_seconds,
        "engine_seconds": page.engine_elapsed_seconds,
    }


def _evidence(
    items: list[CloudEvidence],
    pages: dict[int, Any],
    blocks: dict[str, Any],
    chunks: dict[str, Any],
    visual_pages: set[int],
) -> list[CloudEvidence]:
    valid = []
    for item in items:
        block = blocks.get(item.block_id or "")
        block_ok = (
            block is not None
            and block.page == item.page
            and (not item.quote or item.quote in block.text)
        )
        chunk = chunks.get(item.chunk_id or "")
        chunk_ok = (
            chunk is not None
            and chunk.page == item.page
            and (
                chunk.provenance != "gpt-visual"
                or (
                    chunk.verification in {"model_verified", "human_approved"}
                    and chunk.type not in {"figure", "chart"}
                )
            )
            and (not item.quote or item.quote in chunk.text)
        )
        box_ok = (
            item.block_id is None
            and item.chunk_id is None
            and item.source == "gpt-visual"
            and item.page in pages
            and item.page in visual_pages
            and item.bbox is not None
            and len(item.bbox) == 4
            and all(0 <= x <= 1 for x in item.bbox)
            and item.bbox[0] < item.bbox[2]
            and item.bbox[1] < item.bbox[3]
        )
        if block_ok or chunk_ok or box_ok:
            valid.append(item)
    return valid


def _splits(
    selected: list[int],
    boundaries: list[int],
    override_reason: str | None,
    cloud: CloudResult | None,
    pages: dict[int, Any],
    blocks: dict[str, Any],
    chunks: dict[str, Any],
    visual_pages: set[int],
    review: list[str],
) -> list[GroundedSplit]:
    if boundaries:
        starts = [selected[0], *sorted(set(boundaries))]
        if any(start not in selected[1:] for start in starts[1:]):
            review.append("Invalid split override; boundary must start on a selected source page.")
            starts = [selected[0]]
        result = []
        for index, start in enumerate(starts):
            end_index = (
                selected.index(starts[index + 1]) if index + 1 < len(starts) else len(selected)
            )
            source_pages = selected[selected.index(start) : end_index]
            result.append(
                GroundedSplit(
                    name=f"Document {index + 1}",
                    page_start=source_pages[0],
                    page_end=source_pages[-1],
                    source_pages=source_pages,
                    overridden=True,
                    override_reason=override_reason,
                )
            )
        return result
    proposed = []
    for index, item in enumerate(cloud.splits if cloud else []):
        source_pages = [page for page in selected if item.page_start <= page <= item.page_end]
        evidence = _evidence(item.evidence, pages, blocks, chunks, visual_pages)
        previous_page = (
            selected[selected.index(item.page_start) - 1]
            if item.page_start in selected[1:]
            else None
        )
        boundary_grounded = index == 0 or any(
            item_evidence.page in {item.page_start, previous_page} for item_evidence in evidence
        )
        if (
            source_pages
            and source_pages[0] == item.page_start
            and source_pages[-1] == item.page_end
            and boundary_grounded
        ):
            proposed.append(
                GroundedSplit(
                    **item.model_dump(exclude={"evidence"}),
                    source_pages=source_pages,
                    evidence=evidence,
                )
            )
    covered = [page for item in proposed for page in item.source_pages]
    if covered != selected:
        review.append(
            "Split proposal did not cover selected pages exactly; one safe segment was used."
        )
        return [
            GroundedSplit(
                name="Document 1",
                page_start=selected[0],
                page_end=selected[-1],
                source_pages=selected,
            )
        ]
    return proposed


def _extract(
    request: DocumentRequest,
    cloud: CloudResult | None,
    pages: dict[int, Any],
    blocks: dict[str, Any],
    chunks: dict[str, Any],
    block_chunks: dict[str, str],
    visual_pages: set[int],
    review: list[str],
) -> list[ValidatedField]:
    if Capability.EXTRACT not in request.capabilities:
        return []
    schema = copy.deepcopy(request.extraction_schema or {})
    try:
        Draft202012Validator.check_schema(schema)
    except Exception:
        return []
    properties = schema.get("properties", {})
    fields: list[ValidatedField] = []
    values: dict[str, Any] = {}
    seen: set[str] = set()
    for proposed in cloud.extracted_fields if cloud else []:
        seen.add(proposed.path)
        grounded = _evidence(proposed.evidence, pages, blocks, chunks, visual_pages)
        status = proposed.status if proposed.status in {"verified", "uncertain"} else "uncertain"
        validation_errors: list[str] = []
        normalized = _normalize_value(proposed.value, properties.get(proposed.path, {}))
        if proposed.path not in properties:
            status = "invalid"
            validation_errors.append("Field is not declared by the JSON Schema.")
        elif proposed.abstention_reason:
            status = "abstained"
        elif not grounded:
            status = "invalid"
            validation_errors.append("No valid source evidence.")
        else:
            validation_errors.extend(
                error.message
                for error in Draft202012Validator(
                    properties[proposed.path], format_checker=FormatChecker()
                ).iter_errors(normalized)
            )
            if validation_errors:
                status = "invalid"
            min_confidence = properties[proposed.path].get("x-min-confidence")
            if (
                not validation_errors
                and min_confidence is not None
                and (proposed.confidence is None or proposed.confidence < min_confidence)
            ):
                status = "uncertain"
                validation_errors.append(
                    f"Confidence does not meet schema-specific minimum {min_confidence}."
                )
        evidence = grounded[0] if grounded else None
        if evidence and evidence.block_id is None and evidence.chunk_id:
            chunk = chunks.get(evidence.chunk_id)
            block_id = chunk.source_block_ids[0] if chunk and chunk.source_block_ids else None
        else:
            block_id = evidence.block_id if evidence else None
        block = blocks.get(block_id) if block_id else None
        field = ValidatedField(
            path=proposed.path,
            value=proposed.value,
            normalized_value=normalized,
            source_text=_source_text(block, pages.get(evidence.page) if evidence else None),
            engine_provenance=["RapidOCR", "PP-DocLayoutV3", MODEL_NAME],
            status=status,
            confidence=proposed.confidence,
            source_page=evidence.page if evidence else None,
            source_block_id=block_id,
            source_chunk_id=(
                evidence.chunk_id
                if evidence and evidence.chunk_id
                else block_chunks.get(block_id or "")
            ),
            bbox=(evidence.bbox if evidence and evidence.bbox else getattr(block, "bbox", None)),
            polygon=getattr(block, "polygon", None),
            validation_errors=validation_errors,
            validation_outcome=(
                "passed"
                if status == "verified"
                else "failed"
                if status == "invalid"
                else "not_validated"
            ),
            review_reason=(
                validation_errors[0] if validation_errors else proposed.abstention_reason
            ),
            abstention_reason=proposed.abstention_reason,
            evidence=grounded,
            confidence_by_engine=[
                ConfidenceRecord(
                    engine="RapidOCR",
                    value=getattr(block, "ocr_score", None),
                    kind="recognition",
                ),
                ConfidenceRecord(
                    engine=MODEL_NAME,
                    value=proposed.confidence,
                    kind="model_asserted",
                ),
            ],
            requires_review=status != "verified",
        )
        fields.append(field)
        if status == "verified":
            values[proposed.path] = normalized
        if status != "verified":
            review.append(f"Field {proposed.path!r} is {status}.")
    required_fields = set(schema.get("required", []))
    for field_name in properties:
        if field_name not in seen:
            required = field_name in required_fields
            fields.append(
                ValidatedField(
                    path=field_name,
                    status="abstained",
                    abstention_reason=(
                        "Required field was not proposed."
                        if required
                        else "No supported optional value was proposed."
                    ),
                    requires_review=required,
                )
            )
            if required:
                review.append(f"Required field {field_name!r} was not extracted.")
    for error in Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(values):
        review.append(f"Schema validation: {error.message}")
    review.extend(_business_rule_errors(values, request.business_rules))
    return fields


def _normalize_value(value: Any, field_schema: dict[str, Any]) -> Any:
    expected = field_schema.get("type")
    if expected in {"array", "object"} and isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if (expected == "array" and isinstance(decoded, list)) or (
            expected == "object" and isinstance(decoded, dict)
        ):
            return decoded
    if expected == "number" and isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return value
    if expected == "integer" and isinstance(value, str):
        try:
            return int(value.replace(",", "").strip())
        except ValueError:
            return value
    if expected == "boolean" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    return value


def _business_rule_errors(values: dict[str, Any], rules: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for rule in rules:
        op = rule.get("op")
        if op == "equals":
            field, expected = rule.get("field"), rule.get("value")
            if values.get(field) != expected:
                errors.append(f"Business rule failed for {field!r}: equals {expected!r}.")
        elif op == "sum_equals":
            names = rule.get("fields", [])
            target = rule.get("target")
            tolerance = float(rule.get("tolerance", 0.01))
            operands = [item for name in names for item in _rule_values(values, name)]
            target_values = _rule_values(values, target)
            actual = target_values[0] if len(target_values) == 1 else None
            numeric_operands = [
                float(value) for value in operands if isinstance(value, (int, float))
            ]
            if (
                not operands
                or actual is None
                or len(numeric_operands) != len(operands)
                or not isinstance(actual, (int, float))
                or abs(sum(numeric_operands) - actual) > tolerance
            ):
                errors.append(
                    f"Business rule failed: sum({names!r}) must equal {target!r} "
                    f"within {tolerance}."
                )
        elif op == "less_than_or_equal":
            left, right = rule.get("left"), rule.get("right")
            left_values = _rule_values(values, left)
            right_values = _rule_values(values, right)
            if len(left_values) != 1 or len(right_values) != 1 or left_values[0] > right_values[0]:
                errors.append(f"Business rule failed: {left!r} must be <= {right!r}.")
        else:
            errors.append(f"Unsupported business rule operation: {op!r}.")
    return errors


def _valid_checkbox_bbox(bbox: list[float]) -> bool:
    return (
        len(bbox) == 4
        and all(0 <= value <= 1 for value in bbox)
        and bbox[0] < bbox[2]
        and bbox[1] < bbox[3]
    )


def _checkbox_quality_risks(local: LocalParseResult, page: int) -> list[str]:
    diagnostic = next((item for item in local.quality_diagnostics if item.get("page") == page), {})
    labels = {
        "blur_warning": "blur",
        "low_contrast": "low contrast",
        "shadow_warning": "shadows",
        "compression_artifacts_likely": "compression artifacts",
        "cropped_edge_warning": "cropped edges",
    }
    return [label for key, label in labels.items() if diagnostic.get(key) is True]


def _checkbox_label(
    candidate: CloudCheckbox, pages: dict[int, Any], blocks: dict[str, Any], chunks: dict[str, Any]
) -> tuple[Any | None, Any | None, list[str]]:
    risks: list[str] = []
    block = blocks.get(candidate.label_block_id or "")
    chunk = chunks.get(candidate.label_chunk_id or "")
    if block is not None and block.page != candidate.page:
        block = None
    if chunk is not None and chunk.page != candidate.page:
        chunk = None
    source_text = block.text if block is not None else chunk.text if chunk is not None else ""
    cited = any(
        item.page == candidate.page
        and (
            (block is not None and item.block_id == block.id)
            or (chunk is not None and item.chunk_id == chunk.id)
        )
        and (not item.quote or item.quote in source_text)
        for item in candidate.evidence
    )
    label_matches = bool(
        source_text
        and candidate.label.strip()
        and (
            candidate.label.casefold() in source_text.casefold()
            or source_text.casefold() in candidate.label.casefold()
        )
    )
    semantic_label = candidate.label.strip()
    source_bbox = block.bbox if block is not None else chunk.bbox if chunk is not None else None
    label_bbox = candidate.label_bbox
    geometry_grounded = False
    if (
        label_bbox is not None
        and source_bbox is not None
        and _valid_checkbox_bbox(label_bbox)
        and _valid_checkbox_bbox(source_bbox)
    ):
        vertical_overlap = max(
            0.0,
            min(candidate.control_bbox[3], label_bbox[3])
            - max(candidate.control_bbox[1], label_bbox[1]),
        )
        minimum_height = min(
            candidate.control_bbox[3] - candidate.control_bbox[1],
            label_bbox[3] - label_bbox[1],
        )
        horizontal_gap = label_bbox[0] - candidate.control_bbox[2]
        label_width = label_bbox[2] - label_bbox[0]
        relative_control_center = (
            ((candidate.control_bbox[0] + candidate.control_bbox[2]) / 2 - label_bbox[0])
            / label_width
            if label_width > 0
            else 0.0
        )
        control_inside_label = (
            label_bbox[0] <= candidate.control_bbox[0]
            and label_bbox[2] >= candidate.control_bbox[2]
            and relative_control_center >= 0.55
        )
        label_center = (
            (label_bbox[0] + label_bbox[2]) / 2,
            (label_bbox[1] + label_bbox[3]) / 2,
        )
        geometry_grounded = bool(
            minimum_height > 0
            and vertical_overlap / minimum_height >= 0.5
            and (control_inside_label or -0.04 <= horizontal_gap <= 0.04)
            and label_bbox[2] >= candidate.control_bbox[2]
            and source_bbox[0] - 0.01 <= label_center[0] <= source_bbox[2] + 0.01
            and source_bbox[1] - 0.01 <= label_center[1] <= source_bbox[3] + 0.01
        )
    if candidate.page not in pages or not (block or chunk) or not cited or not label_matches:
        risks.append("no unique RapidOCR label grounding")
    if not any(character.isalpha() for character in semantic_label):
        risks.append("checkbox label contains no alphabetic text")
    if not geometry_grounded:
        risks.append("checkbox label is not geometrically grounded to the control")
    return block, chunk, risks


def _checkbox_risks(
    local: LocalParseResult,
    candidate: CloudCheckbox,
    pages: dict[int, Any],
    blocks: dict[str, Any],
    chunks: dict[str, Any],
    visual_pages: set[int],
) -> list[str]:
    risks: list[str] = []
    if candidate.page not in visual_pages or not _valid_checkbox_bbox(candidate.control_bbox):
        risks.append("invalid or unreviewed visual geometry")
    elif candidate.page in pages:
        page = pages[candidate.page]
        width = (candidate.control_bbox[2] - candidate.control_bbox[0]) * page.width
        height = (candidate.control_bbox[3] - candidate.control_bbox[1]) * page.height
        aspect = width / height if height else 0
        if not 0.5 <= aspect <= 2:
            risks.append("control geometry is not checkbox-shaped")
    if candidate.label_bbox is not None and not _valid_checkbox_bbox(candidate.label_bbox):
        risks.append("label geometry is invalid")
    if candidate.state not in {"CHECKED", "UNCHECKED"}:
        risks.append(f"state is {candidate.state}")
    _, _, label_risks = _checkbox_label(candidate, pages, blocks, chunks)
    risks.extend(label_risks)
    risks.extend(_checkbox_quality_risks(local, candidate.page))
    return risks


def _checkbox_candidates(local: LocalParseResult, cloud: CloudResult) -> list[CloudCheckbox]:
    """Return Sol discoveries plus local proposals that can pass consensus."""
    candidates: list[CloudCheckbox] = []
    seen: set[str] = set()
    for candidate in cloud.checkboxes:
        if candidate.id not in seen:
            candidates.append(candidate)
            seen.add(candidate.id)
    for page in local.pages:
        for candidate in page.local_checkbox_candidates:
            if not is_credible_checkbox_candidate(page, candidate):
                continue
            if any(
                item.page == candidate.page
                and _bbox_iou(item.control_bbox, candidate.control_bbox) > 0.5
                for item in candidates
            ):
                continue
            identifier = candidate.id
            if identifier in seen:
                identifier = f"{identifier}-local"
            label_bbox = candidate.label_bbox or next(
                (
                    block.bbox
                    for block in page.blocks
                    if block.id == candidate.label_block_id and block.bbox is not None
                ),
                None,
            )
            if label_bbox is None:
                label_bbox = next(
                    (
                        chunk.bbox
                        for chunk in page.chunks
                        if chunk.id == candidate.label_chunk_id and chunk.bbox is not None
                    ),
                    None,
                )
            candidates.append(
                CloudCheckbox(
                    id=identifier,
                    page=candidate.page,
                    label=candidate.source_text or "",
                    state=candidate.state.value,
                    control_bbox=candidate.control_bbox,
                    label_bbox=label_bbox,
                    label_block_id=candidate.label_block_id,
                    label_chunk_id=candidate.label_chunk_id,
                    confidence=candidate.detector_score,
                    reason="OpenCV-only proposal; Sol discovery did not independently match it.",
                    evidence=(
                        [
                            CloudEvidence(
                                page=candidate.page,
                                block_id=candidate.label_block_id,
                                chunk_id=candidate.label_chunk_id,
                                quote=candidate.source_text or "",
                            )
                        ]
                        if candidate.source_text
                        else []
                    ),
                )
            )
            seen.add(identifier)
    return candidates


def _matching_local_checkbox(
    local: LocalParseResult, candidate: CloudCheckbox
) -> LocalCheckboxCandidate | None:
    matches = [
        item
        for page in local.pages
        for item in page.local_checkbox_candidates
        if item.page == candidate.page
        and is_visual_checkbox_candidate(page, item)
        and _bbox_iou(item.control_bbox, candidate.control_bbox) > 0.2
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: _bbox_iou(item.control_bbox, candidate.control_bbox))


def _checkboxes(
    local: LocalParseResult,
    cloud: CloudResult | None,
    pages: dict[int, Any],
    blocks: dict[str, Any],
    chunks: dict[str, Any],
    visual_pages: set[int],
    verifications: dict[str, CheckboxVerification],
    review: list[str],
) -> list[CheckboxRecord]:
    records: list[CheckboxRecord] = []
    seen: set[str] = set()
    candidates = _checkbox_candidates(local, cloud) if cloud else []
    cloud_ids = {item.id for item in cloud.checkboxes} if cloud else set()
    for candidate in candidates:
        identifier = candidate.id.strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", identifier) or identifier in seen:
            review.append("A checkbox proposal had a missing or duplicate ID.")
            continue
        seen.add(identifier)
        verification = verifications.get(identifier)
        # A shape proposal is not a publishable checkbox until the independent
        # crop review confirms that the pixels depict a checkbox control.
        if verification is None or verification.control_status != "checkbox":
            continue
        local_candidate = _matching_local_checkbox(local, candidate)
        if local_candidate is not None and not candidate.label.strip():
            candidate = candidate.model_copy(
                update={
                    "label": local_candidate.source_text or "",
                    "label_bbox": local_candidate.label_bbox,
                    "label_block_id": local_candidate.label_block_id,
                    "label_chunk_id": local_candidate.label_chunk_id,
                    "evidence": [
                        CloudEvidence(
                            page=local_candidate.page,
                            block_id=local_candidate.label_block_id,
                            chunk_id=local_candidate.label_chunk_id,
                            quote=local_candidate.source_text or "",
                        )
                    ],
                }
            )
        block, chunk, label_risks = _checkbox_label(candidate, pages, blocks, chunks)
        risks = _checkbox_risks(local, candidate, pages, blocks, chunks, visual_pages)
        if any(
            item.page == candidate.page
            and _bbox_iou(candidate.control_bbox, item.control_bbox) > 0.5
            for item in records
        ):
            risks.append("control overlaps another checkbox proposal")
        sol_discovered = identifier in cloud_ids
        verification_matches = bool(
            verification
            and verification.page == candidate.page
            and verification.state == candidate.state
            and verification.state in {"CHECKED", "UNCHECKED"}
            and verification.confidence is not None
            and verification.confidence >= _MIN_CHECKBOX_VERIFICATION_CONFIDENCE
        )
        if local_candidate is None:
            risks.append("OpenCV did not independently detect this control")
        else:
            if local_candidate.state.value != candidate.state:
                risks.append("OpenCV and main Sol states disagree")
            label_ids_agree = bool(
                candidate.label_block_id
                and candidate.label_block_id == local_candidate.label_block_id
            ) or bool(
                candidate.label_chunk_id
                and candidate.label_chunk_id == local_candidate.label_chunk_id
            )
            if not label_ids_agree:
                risks.append("OpenCV and GPT label grounding disagree")
        label_score = (
            block.ocr_score
            if block is not None
            else min((score for score in chunk.raw_scores if score is not None), default=None)
            if chunk is not None
            else None
        )
        if label_risks:
            risks.append("RapidOCR label grounding is missing or ambiguous")
        if label_score is None or label_score < 0.85:
            risks.append("RapidOCR label confidence is below 0.85")
        if not sol_discovered and not verification_matches:
            risks.append("Sol did not confirm this control")
        if verification and not verification_matches:
            risks.append(
                "targeted GPT verification did not agree at confidence "
                f"{_MIN_CHECKBOX_VERIFICATION_CONFIDENCE:.2f}"
            )
        automated = not risks
        reason = None if automated else "; ".join(dict.fromkeys(risks))
        if reason:
            review.append(f"Checkbox {identifier!r} requires review: {reason}.")
        source_text = block.text if block is not None else chunk.text if chunk is not None else None
        records.append(
            CheckboxRecord(
                id=identifier,
                page=candidate.page,
                label=candidate.label,
                state=CheckboxState(candidate.state),
                control_bbox=candidate.control_bbox,
                label_bbox=candidate.label_bbox,
                label_block_id=candidate.label_block_id,
                label_chunk_id=candidate.label_chunk_id,
                source_text=source_text,
                confidence=(
                    verification.confidence if automated and verification else candidate.confidence
                ),
                discovery_state=CheckboxState(candidate.state),
                discovery_confidence=candidate.confidence,
                verification_state=(CheckboxState(verification.state) if verification else None),
                verification_confidence=verification.confidence if verification else None,
                verification_reason=verification.reason if verification else None,
                decision_status="automated" if automated else "review_required",
                review_reason=reason,
                crop_ref=(
                    f"checkboxes/{identifier}.jpg"
                    if _valid_checkbox_bbox(candidate.control_bbox) and candidate.page in pages
                    else None
                ),
                local_vision_state=local_candidate.state if local_candidate else None,
                local_vision_score=(local_candidate.detector_score if local_candidate else None),
                local_vision_bbox=(local_candidate.control_bbox if local_candidate else None),
                local_vision_id=local_candidate.id if local_candidate else None,
                ocr_label_score=local_candidate.ocr_score if local_candidate else None,
                ocr_grounding_unique=(
                    local_candidate.ocr_grounding_unique if local_candidate else False
                ),
                agreement=(
                    "consensus"
                    if automated
                    else "disagreement"
                    if local_candidate
                    and verification
                    and (
                        local_candidate.state.value != candidate.state
                        or verification.state != candidate.state
                    )
                    else "incomplete"
                ),
                engine_provenance=(
                    (["OpenCV"] if local_candidate else []) + ["RapidOCR", "gpt-6-sol"]
                ),
            )
        )
    return records


def _bbox_iou(left: list[float], right: list[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _checkbox_rule_errors(
    checkboxes: list[CheckboxRecord], rules: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    by_id = {item.id: item for item in checkboxes}
    by_label = {item.label: item for item in checkboxes}
    for rule in rules:
        op = rule.get("op")
        if not isinstance(op, str) or not op.startswith("checkbox_"):
            continue
        identifiers = rule.get("checkbox_ids", [])
        if op == "checkbox_none_exclusive" and not identifiers:
            identifiers = [rule.get("none_id"), *rule.get("other_ids", [])]
        selected = [
            by_id[item] for item in identifiers if isinstance(item, str) and item in by_id
        ] or [
            by_label[item]
            for item in rule.get("labels", [])
            if isinstance(item, str) and item in by_label
        ]
        checked = sum(item.state is CheckboxState.CHECKED for item in selected)
        if not selected:
            errors.append(f"Checkbox business rule {op!r} matched no controls.")
        elif op == "checkbox_exactly_one" and checked != 1:
            errors.append("Checkbox business rule failed: exactly one control must be checked.")
        elif op == "checkbox_min_selected" and checked < _safe_int(rule.get("minimum"), 1):
            errors.append("Checkbox business rule failed: minimum selected count was not met.")
        elif op == "checkbox_max_selected" and checked > _safe_int(rule.get("maximum"), 1):
            errors.append("Checkbox business rule failed: maximum selected count was exceeded.")
        elif op == "checkbox_mutually_exclusive" and checked > 1:
            errors.append("Checkbox business rule failed: controls are mutually exclusive.")
        elif op == "checkbox_none_exclusive":
            none_id = rule.get("none_id")
            none = by_id.get(none_id) if isinstance(none_id, str) else None
            others = [item for item in selected if item is not none]
            if (
                none
                and none.state is CheckboxState.CHECKED
                and any(item.state is CheckboxState.CHECKED for item in others)
            ):
                errors.append("Checkbox business rule failed: none cannot coexist with a choice.")
        elif op not in {
            "checkbox_exactly_one",
            "checkbox_min_selected",
            "checkbox_max_selected",
            "checkbox_mutually_exclusive",
            "checkbox_none_exclusive",
        }:
            errors.append(f"Unsupported checkbox business rule operation: {op!r}.")
    return errors


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _schema_version(schema: dict[str, Any] | None) -> str | None:
    if not schema:
        return None
    value = schema.get("x-schema-version") or schema.get("$id")
    return str(value) if value is not None else "unversioned"


def _source_text(block: Any, page: Any) -> str | None:
    if block is None:
        return None
    texts = page.raw_evidence.get("texts") if page is not None else None
    try:
        index = int(block.id.rsplit("-b", 1)[1]) - 1
    except (IndexError, ValueError):
        index = -1
    if isinstance(texts, list) and 0 <= index < len(texts):
        return str(texts[index])
    return block.text


def _rule_values(values: dict[str, Any], path: Any) -> list[Any]:
    if not isinstance(path, str):
        return []
    if ".*." not in path:
        return [values[path]] if path in values else []
    collection_name, child_name = path.split(".*.", 1)
    collection = values.get(collection_name)
    if not isinstance(collection, list):
        return []
    return [
        item[child_name] for item in collection if isinstance(item, dict) and child_name in item
    ]


def _review_items(messages: list[str]) -> list[ReviewItem]:
    def stage(
        message: str,
    ) -> Literal["parse", "classify", "section", "split", "extract", "checkbox", "validate"]:
        lowered = message.lower()
        if "checkbox" in lowered:
            return "checkbox"
        if "classification" in lowered:
            return "classify"
        if "section" in lowered:
            return "section"
        if "split" in lowered:
            return "split"
        if "field" in lowered:
            return "extract"
        if "low-confidence block" in lowered:
            return "parse"
        if "table" in lowered and "requires review" in lowered:
            return "parse"
        if "page" in lowered:
            return "parse"
        return "validate"

    items: list[ReviewItem] = []
    for index, message in enumerate(messages, 1):
        checkbox_match = re.search(r"Checkbox '([^']+)'", message)
        block_match = re.search(r"low-confidence block ([^:]+):", message)
        table_match = re.search(r"\bTable\s+(p(\d+)-[^\s]+)\s+requires review", message, re.I)
        classification_match = re.search(r"Classification '([^']+)'", message)
        section_match = re.search(r"Section '([^']+)'", message)
        field_match = re.search(r"(?:Required field|Field) '([^']+)'", message)
        item_stage = stage(message)
        source_ids = (
            [checkbox_match.group(1)]
            if checkbox_match
            else (
                [block_match.group(1)]
                if block_match
                else (
                    [table_match.group(1)]
                    if table_match
                    else (
                        [classification_match.group(1)]
                        if classification_match
                        else (
                            [section_match.group(1)]
                            if section_match
                            else ([field_match.group(1)] if field_match else [])
                        )
                    )
                )
            )
        )
        items.append(
            ReviewItem(
                id=f"review-{index}",
                stage=item_stage,
                code=item_stage.upper() + "_REVIEW",
                message=message,
                retryable=bool(source_ids)
                and table_match is None
                and not any(
                    marker in message.lower() for marker in ("failed pages", "low-confidence block")
                ),
                source_ids=source_ids,
                pages=[int(table_match.group(2))] if table_match else [],
            )
        )
    return items
