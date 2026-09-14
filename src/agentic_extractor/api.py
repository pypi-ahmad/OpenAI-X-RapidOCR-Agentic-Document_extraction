"""Versioned FastAPI surface backed by the canonical document workflow.

Responsible for: exposing the versioned local HTTP API (`/api/v1/jobs/...`) for
document parsing, schema extraction, status queries, and artifact downloads,
backed by `workflow.py` and `pipeline.py`.

Must not: implement extraction logic independently from `workflow.py`/`pipeline.py`,
bypass the three-engine requirement (RapidOCR -> PP-DocLayoutV3 -> gpt-5.6-luna),
expose secrets or raw credentials, or persist jobs across process restarts
(uses in-memory storage with lazy TTL eviction).

Next: `workflow.py` for the state machine execution behind these endpoints, and
`artifacts.py` for how requested artifacts are assembled.
"""

from __future__ import annotations

import base64
import binascii
import copy
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, HTTPException, Response, status
from jsonschema import SchemaError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_extractor.artifacts import LocalArtifacts, build_local_artifacts
from agentic_extractor.config import SETTINGS
from agentic_extractor.layout import (
    PPDocLayoutResource,
    create_pp_doclayout_engine,
)
from agentic_extractor.models import (
    Capability,
    CheckboxCorrection,
    CheckboxRecord,
    DocumentRequest,
    ProcessingMode,
)
from agentic_extractor.ocr import LocalParseResult, OCRResource, create_rapidocr_engine
from agentic_extractor.openai_refiner import OpenAIConfigurationError, OpenAIRefiner
from agentic_extractor.pipeline import Refiner
from agentic_extractor.schema_input import validate_schema
from agentic_extractor.workflow import (
    AgentWorkflowResult,
    FieldCorrection,
    ReviewItem,
    ValidatedField,
    run_agent_workflow,
    run_workflow_from_parse,
)


class JobState(StrEnum):
    PROCESSING = "PROCESSING"
    ACCEPTED = "ACCEPTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"


class ApiError(BaseModel):
    code: str
    message: str
    action: str | None = None
    retryable: bool = False


class ErrorResponse(BaseModel):
    detail: ApiError


class ParseJobRequest(BaseModel):
    file_name: str = Field(min_length=1, max_length=255)
    content_base64: str
    mode: ProcessingMode = ProcessingMode.BALANCED
    selected_pages: list[int] | None = None
    enable_preprocessing: bool = False
    include_atomic_grounding: bool = True

    @field_validator("content_base64")
    @classmethod
    def validate_content(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("content_base64 must be valid base64.") from exc
        if not decoded or len(decoded) > SETTINGS.max_upload_bytes:
            raise ValueError("Decoded document is empty or exceeds the 50 MiB limit.")
        return value

    def content(self) -> bytes:
        return base64.b64decode(self.content_base64, validate=True)

    @field_validator("selected_pages")
    @classmethod
    def validate_pages(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and (not value or any(page < 1 for page in value)):
            raise ValueError("selected_pages must contain positive one-based page numbers.")
        return sorted(set(value)) if value else value


class ExtractionJobRequest(BaseModel):
    schema_version: str = Field(min_length=1, max_length=64)
    schema_definition: dict[str, Any] = Field(alias="schema")
    business_rules: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("schema_definition")
    @classmethod
    def validate_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            validate_schema(value)
        except (SchemaError, ValueError) as exc:
            detail = exc.message if isinstance(exc, SchemaError) else str(exc)
            raise ValueError(f"Invalid JSON Schema: {detail}") from exc
        return value


class JobCreated(BaseModel):
    job_id: str
    state: JobState
    status_url: str


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    warnings: list[str] = Field(default_factory=list)
    failed_pages: list[int] = Field(default_factory=list)
    review_required: list[str] = Field(default_factory=list)
    result: ParseResultPayload | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: ApiError | None = None


class ParseResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    markdown: str
    metadata: dict[str, Any]
    structure: dict[str, Any]


class ExtractionResult(BaseModel):
    job_id: str
    state: JobState
    schema_version: str | None = None
    fields: list[ValidatedField] = Field(default_factory=list)
    corrections: list[FieldCorrection] = Field(default_factory=list)
    review_required: list[str] = Field(default_factory=list)
    review_items: list[ReviewItem] = Field(default_factory=list)
    checkboxes: list[CheckboxRecord] = Field(default_factory=list)
    checkbox_corrections: list[CheckboxCorrection] = Field(default_factory=list)


class ArtifactMetadata(BaseModel):
    name: str
    bytes: int | None = None
    sha256: str | None = None
    generated: bool
    download_url: str


class ArtifactList(BaseModel):
    job_id: str
    artifacts: list[ArtifactMetadata]


@dataclass(slots=True)
class _Job:
    source_request: ParseJobRequest
    created_at: float
    state: JobState = JobState.PROCESSING
    local: LocalParseResult | None = None
    workflow: AgentWorkflowResult | None = None
    artifacts: LocalArtifacts | None = None
    error: ApiError | None = None


def create_app(
    *,
    ocr_resource: OCRResource | None = None,
    layout_resource: PPDocLayoutResource | None = None,
    refiner: Refiner | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    app = FastAPI(
        title="Agentic document extractor API",
        version="1.0.0",
        description=(
            "Local-only API over the canonical RapidOCR, PP-DocLayoutV3, and GPT-5.6-luna pipeline."
        ),
    )
    jobs: dict[str, _Job] = {}
    resource: OCRResource | None = ocr_resource
    layout: PPDocLayoutResource | None = layout_resource
    cloud_refiner: Refiner | None = refiner

    def get_resource() -> OCRResource:
        nonlocal resource
        if resource is None:
            resource = create_rapidocr_engine()
        return resource

    def get_refiner() -> Refiner:
        nonlocal cloud_refiner
        if cloud_refiner is None:
            cloud_refiner = OpenAIRefiner()
        return cloud_refiner

    def get_layout() -> PPDocLayoutResource:
        nonlocal layout
        if layout is None:
            layout = create_pp_doclayout_engine()
        return layout

    def prepare_engines() -> tuple[Refiner, OCRResource, PPDocLayoutResource, dict[str, float]]:
        initialization_timings: dict[str, float] = {}
        configuration_started = time.perf_counter()
        try:
            active_refiner = get_refiner()
            active_refiner.validate_configuration()
        except (OpenAIConfigurationError, RuntimeError) as exc:
            raise _http_error(
                503,
                "openai_unavailable",
                str(exc),
                "Add valid OpenAI configuration and retry.",
                retryable=True,
            ) from exc
        initialization_timings["configuration_seconds"] = (
            time.perf_counter() - configuration_started
        )
        rapidocr_initialization_started = time.perf_counter()
        try:
            active_resource = get_resource()
        except Exception as exc:
            raise _http_error(
                503,
                "rapidocr_unavailable",
                str(exc),
                "Run 'uv sync --all-groups', then verify RapidOCR and ONNX Runtime.",
                retryable=True,
            ) from exc
        initialization_timings["rapidocr_initialization_seconds"] = (
            time.perf_counter() - rapidocr_initialization_started
        )
        layout_initialization_started = time.perf_counter()
        try:
            active_layout = get_layout()
        except Exception as exc:
            raise _http_error(
                503,
                "pp_doclayout_unavailable",
                str(exc),
                "Run 'uv sync --project tools/pp_doclayout --locked', then verify Paddle.",
                retryable=True,
            ) from exc
        initialization_timings["layout_initialization_seconds"] = (
            time.perf_counter() - layout_initialization_started
        )
        return active_refiner, active_resource, active_layout, initialization_timings

    def find_job(job_id: str) -> _Job:
        return _find(job_id, jobs, now=clock())

    @app.post(
        "/api/v1/jobs/parse",
        response_model=JobCreated,
        status_code=status.HTTP_201_CREATED,
        summary="Submit and parse a document",
        tags=["jobs"],
        responses={503: {"model": ErrorResponse}},
    )
    def submit_parse(payload: ParseJobRequest) -> JobCreated:
        active_refiner, active_resource, active_layout, initialization_timings = prepare_engines()
        job_id = secrets.token_urlsafe(24)
        job = _Job(source_request=payload, created_at=clock())
        jobs[job_id] = job
        request = DocumentRequest(
            file_name=payload.file_name,
            file_bytes=payload.content(),
            mode=payload.mode,
            selected_pages=set(payload.selected_pages) if payload.selected_pages else None,
            enable_preprocessing=payload.enable_preprocessing,
        )
        _run(
            job,
            request,
            active_refiner,
            lambda: active_resource,
            active_layout,
            initialization_timings,
        )
        return JobCreated(
            job_id=job_id,
            state=job.state,
            status_url=f"/api/v1/jobs/{job_id}",
        )

    @app.get(
        "/api/v1/jobs/{job_id}",
        response_model=JobStatus,
        summary="Get job status and Parse output",
        tags=["jobs"],
        responses={404: {"model": ErrorResponse}},
    )
    def get_status(job_id: str) -> JobStatus:
        return _status(job_id, jobs, now=clock())

    @app.post(
        "/api/v1/jobs/{job_id}/extract",
        response_model=JobStatus,
        summary="Extract a versioned schema from an existing Parse result",
        tags=["extraction"],
        responses={
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def extract(job_id: str, payload: ExtractionJobRequest) -> JobStatus:
        job = find_job(job_id)
        if job.local is None:
            raise _http_error(
                409,
                "parse_unavailable",
                "Canonical Parse result is not available.",
                "Submit a successful Parse job before extraction.",
            )
        source = job.source_request
        request = DocumentRequest(
            file_name=source.file_name,
            file_bytes=source.content(),
            mode=source.mode,
            selected_pages=set(source.selected_pages) if source.selected_pages else None,
            capabilities={Capability.PARSE, Capability.EXTRACT},
            extraction_schema={
                **payload.schema_definition,
                "x-schema-version": payload.schema_version,
            },
            business_rules=payload.business_rules,
            enable_preprocessing=source.enable_preprocessing,
        )
        try:
            active_refiner = get_refiner()
            active_refiner.validate_configuration()
        except (OpenAIConfigurationError, RuntimeError) as exc:
            raise _http_error(
                503,
                "openai_unavailable",
                str(exc),
                "Add valid OpenAI configuration and retry.",
                retryable=True,
            ) from exc
        _run_from_parse(job, request, active_refiner)
        return _status(job_id, jobs, now=clock())

    @app.get(
        "/api/v1/jobs/{job_id}/extraction",
        response_model=ExtractionResult,
        summary="Get structured extraction results",
        tags=["extraction"],
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def get_extraction(job_id: str) -> ExtractionResult:
        job = find_job(job_id)
        if job.workflow is None:
            raise _http_error(
                409,
                "extraction_unavailable",
                "Extraction result is not available.",
                "Run schema extraction for this job first.",
            )
        return ExtractionResult(
            job_id=job_id,
            state=job.state,
            schema_version=job.workflow.schema_version,
            fields=job.workflow.extracted_fields,
            corrections=job.workflow.corrections,
            review_required=job.workflow.review_required,
            review_items=job.workflow.review_items,
            checkboxes=job.workflow.checkboxes,
            checkbox_corrections=job.workflow.checkbox_corrections,
        )

    @app.get(
        "/api/v1/jobs/{job_id}/artifacts",
        response_model=ArtifactList,
        summary="List generated artifact metadata",
        tags=["artifacts"],
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def list_artifacts(job_id: str) -> ArtifactList:
        job = find_job(job_id)
        artifacts = _require_artifacts(job)
        return ArtifactList(
            job_id=job_id,
            artifacts=[
                ArtifactMetadata(
                    **artifacts.artifact_metadata(name),
                    download_url=f"/api/v1/jobs/{job_id}/artifacts/{name}",
                )
                for name in _ARTIFACT_NAMES
            ],
        )

    @app.get(
        "/api/v1/jobs/{job_id}/artifacts/{artifact_name}",
        summary="Download an allowlisted artifact",
        tags=["artifacts"],
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def download_artifact(job_id: str, artifact_name: str) -> Response:
        job = find_job(job_id)
        # Security allowlist: prevents path traversal and restricts downloads
        # strictly to pipeline-produced artifacts.
        if artifact_name not in _ARTIFACT_NAMES:
            raise _http_error(404, "artifact_not_found", "Unknown artifact.")
        content, media_type = _artifact_payload(job, artifact_name)
        return Response(
            content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{artifact_name}"'},
        )

    return app


def _run(
    job: _Job,
    request: DocumentRequest,
    refiner: Refiner,
    resource_provider: Callable[[], OCRResource],
    layout_resource: PPDocLayoutResource,
    initialization_timings: dict[str, float],
) -> None:
    job.state = JobState.PROCESSING
    try:
        refiner.validate_configuration()
        local, workflow = run_agent_workflow(
            request,
            ocr_resource=resource_provider(),
            layout_resource=layout_resource,
            refiner=refiner,
            initialization_timings=initialization_timings,
        )
        job.local = local
        job.workflow = workflow
        job.artifacts = build_local_artifacts(
            local,
            include_atomic_grounding=job.source_request.include_atomic_grounding,
        )
        job.state = JobState(workflow.current_state.value)
        job.error = None
    except Exception as exc:
        job.state = JobState.FAILED
        job.error = ApiError(
            code="processing_failed",
            message=f"{type(exc).__name__}: {exc}",
            action="Review the job diagnostics, correct the input or configuration, and retry.",
            retryable=True,
        )


def _run_from_parse(job: _Job, request: DocumentRequest, refiner: Refiner) -> None:
    job.state = JobState.PROCESSING
    try:
        if job.local is None:
            raise ValueError("Canonical Parse result is not available.")
        # Defensive deepcopy: keeps canonical Parse evidence immutable across
        # repeated extraction attempts against the same job.
        local, workflow = run_workflow_from_parse(copy.deepcopy(job.local), request, refiner)
        job.local = local
        job.workflow = workflow
        job.artifacts = build_local_artifacts(
            local,
            include_atomic_grounding=job.source_request.include_atomic_grounding,
        )
        job.state = JobState(workflow.current_state.value)
        job.error = None
    except Exception as exc:
        job.state = JobState.FAILED
        job.error = ApiError(
            code="extraction_failed",
            message=f"{type(exc).__name__}: {exc}",
            action="Review the schema and job diagnostics, then retry extraction.",
            retryable=True,
        )


def _find(job_id: str, jobs: dict[str, _Job], *, now: float) -> _Job:
    # Lazy in-memory cleanup: purges expired jobs on lookup to bound process
    # memory without requiring a background reaper thread.
    expired = [
        key
        for key, candidate in jobs.items()
        if now - candidate.created_at >= SETTINGS.job_ttl_seconds
    ]
    for key in expired:
        jobs.pop(key, None)
    job = jobs.get(job_id)
    if job is None:
        raise _http_error(404, "job_not_found", "Job not found or expired.")
    return job


_ARTIFACT_NAMES = (
    "document.md",
    "parse-result.json",
    "annotated.pdf",
    "document.html",
    "bundle.zip",
)


def _require_artifacts(job: _Job) -> LocalArtifacts:
    if job.artifacts is None or job.local is None:
        raise _http_error(
            409,
            "artifacts_unavailable",
            "Artifacts are not available for this job.",
            "Wait for successful processing or review the job failure.",
        )
    return job.artifacts


def _artifact_payload(job: _Job, name: str) -> tuple[bytes, str]:
    artifacts = _require_artifacts(job)
    media_types = {
        "document.md": "text/markdown",
        "parse-result.json": "application/json",
        "annotated.pdf": "application/pdf",
        "document.html": "text/html",
        "bundle.zip": "application/zip",
    }
    return artifacts.get(name), media_types[name]


def _status(job_id: str, jobs: dict[str, _Job], *, now: float) -> JobStatus:
    job = _find(job_id, jobs, now=now)
    local, workflow = job.local, job.workflow
    result = json.loads(job.artifacts.parse_result) if job.artifacts else None
    return JobStatus(
        job_id=job_id,
        state=job.state,
        warnings=(local.warnings if local else ([job.error.message] if job.error else [])),
        failed_pages=local.failed_pages if local else [],
        review_required=workflow.review_required if workflow else [],
        result=result,
        artifacts={name: f"/api/v1/jobs/{job_id}/artifacts/{name}" for name in _ARTIFACT_NAMES}
        if job.artifacts
        else {},
        error=job.error,
    )


def _http_error(
    status_code: int,
    code: str,
    message: str,
    action: str | None = None,
    *,
    retryable: bool = False,
) -> HTTPException:
    detail = ApiError(
        code=code,
        message=message,
        action=action,
        retryable=retryable,
    )
    return HTTPException(status_code, detail=detail.model_dump(mode="json"))


app = create_app()
