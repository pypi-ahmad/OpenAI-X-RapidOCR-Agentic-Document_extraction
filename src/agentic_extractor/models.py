"""Stable public models shared by the pipeline, UI, and exports."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProcessingMode(StrEnum):
    BALANCED = "Balanced"
    HIGH_ACCURACY = "High Accuracy"


class Capability(StrEnum):
    PARSE = "Parse"
    CLASSIFY = "Classify"
    SECTION = "Section"
    SPLIT = "Split"
    EXTRACT = "Extract"


class CheckboxState(StrEnum):
    CHECKED = "CHECKED"
    UNCHECKED = "UNCHECKED"
    INDETERMINATE = "INDETERMINATE"
    CROSSED_OUT = "CROSSED_OUT"
    NOT_DETERMINABLE = "NOT_DETERMINABLE"


class CheckboxRecord(BaseModel):
    """Grounded derived checkbox result; raw OCR evidence remains separate."""

    model_config = ConfigDict(extra="forbid")
    id: str
    page: int = Field(ge=1)
    label: str = ""
    state: CheckboxState
    control_bbox: list[float] = Field(min_length=4, max_length=4)
    label_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    label_block_id: str | None = None
    label_chunk_id: str | None = None
    source_text: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    discovery_state: CheckboxState
    discovery_confidence: float | None = Field(default=None, ge=0, le=1)
    verification_state: CheckboxState | None = None
    verification_confidence: float | None = Field(default=None, ge=0, le=1)
    verification_reason: str | None = None
    decision_status: Literal["automated", "review_required", "user_verified"]
    review_reason: str | None = None
    engine_provenance: list[str] = Field(default_factory=lambda: ["RapidOCR", "gpt-5.6-luna"])
    crop_ref: str | None = None


class CheckboxCorrection(BaseModel):
    checkbox_id: str
    previous_state: CheckboxState
    state: CheckboxState
    reason: str
    timestamp: str
    actor: Literal["user"] = "user"


class Block(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    page: int = Field(ge=1)
    type: str = "text"
    text: str
    ocr_score: float | None = Field(default=None, ge=0, le=1)
    polygon: list[list[float]] | None = None
    bbox: list[float] | None = None
    source: Literal["rapidocr", "gpt-visual"] = "rapidocr"


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int = Field(ge=1)
    block_id: str | None = None
    bbox: list[float] | None = None
    quote: str = ""
    source: Literal["rapidocr", "gpt-visual"]


class RefinementRecord(BaseModel):
    """Auditable GPT proposal kept separate from immutable OCR evidence."""

    model_config = ConfigDict(extra="forbid")
    page: int = Field(ge=1)
    block_id: str | None = None
    status: Literal["accepted", "rejected", "abstained"]
    proposed_text: str | None = None
    proposed_type: str | None = None
    proposed_reading_order: int | None = Field(default=None, ge=1)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    reason: str | None = None
    provider: Literal["gpt-5.6-luna"] = "gpt-5.6-luna"


class UsageRecord(BaseModel):
    call_count: int = 0
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    text_input_tokens: int | None = None
    image_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    cloud_seconds: float = 0
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    total_cost_usd: float | None = None
    cost_status: Literal["exact", "estimate", "unavailable"] = "unavailable"
    calls: list[dict[str, Any]] = Field(default_factory=list)


class Classification(BaseModel):
    label: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class Section(BaseModel):
    title: str
    level: int = Field(ge=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    source_block_ids: list[str] = Field(default_factory=list)


class Split(BaseModel):
    name: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    label: str = "unknown"


class Extraction(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, list[EvidenceRef]] = Field(default_factory=dict)
    valid: bool = True
    errors: list[str] = Field(default_factory=list)


class DocumentRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    file_name: str
    file_bytes: bytes
    mode: ProcessingMode = ProcessingMode.BALANCED
    selected_pages: set[int] | None = None
    forced_cloud_pages: set[int] = Field(default_factory=set)
    capabilities: set[Capability] = Field(default_factory=lambda: {Capability.PARSE})
    allowed_classes: list[str] = Field(default_factory=list)
    extraction_schema: dict[str, Any] | None = None
    split_boundaries: list[int] = Field(default_factory=list)
    split_override_reason: str | None = None
    business_rules: list[dict[str, Any]] = Field(default_factory=list)
    enable_preprocessing: bool = False

    @model_validator(mode="after")
    def validate_options(self) -> DocumentRequest:
        if Capability.CLASSIFY in self.capabilities and not self.allowed_classes:
            raise ValueError("Classify requires at least one allowed class.")
        if Capability.EXTRACT in self.capabilities and not self.extraction_schema:
            raise ValueError("Extract requires a JSON Schema.")
        if self.split_boundaries and not (self.split_override_reason or "").strip():
            raise ValueError("A split override reason is required when split boundaries are set.")
        return self


class DocumentResult(BaseModel):
    status: Literal["complete"] = "complete"
    markdown: str
    blocks: list[Block]
    classifications: list[Classification] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    splits: list[Split] = Field(default_factory=list)
    extraction: Extraction | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    usage: UsageRecord = Field(default_factory=UsageRecord)
    warnings: list[str] = Field(default_factory=list)
    failed_pages: list[int] = Field(default_factory=list)
    refinements: list[RefinementRecord] = Field(default_factory=list)
