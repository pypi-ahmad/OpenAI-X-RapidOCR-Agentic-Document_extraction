"""Stable public models shared by the pipeline, UI, and exports.

Responsible for: declaring core immutable Pydantic schemas, enums, and data
contracts used across the entire system (`ProcessingMode`, `Capability`,
`Block`, `LayoutRegion`, `CheckboxRecord`, `DocumentRequest`, `DocumentResult`).

Must not: implement workflow execution, network calls, or mutating business
logic. Coordinate convention: all bounding boxes (`bbox`) use normalized [0, 1]
floating-point coordinates formatted as `[left, top, right, bottom]` relative
to page width and height.

Next: `pipeline.py` and `workflow.py`, which consume and populate these models.
"""

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


class LocalCheckboxCandidate(BaseModel):
    """Auditable OpenCV proposal linked to nearby immutable RapidOCR evidence."""

    model_config = ConfigDict(extra="forbid")
    id: str
    page: int = Field(ge=1)
    state: CheckboxState
    control_bbox: list[float] = Field(min_length=4, max_length=4)
    detector_score: float = Field(ge=0, le=1)
    border_coverage: float = Field(ge=0, le=1)
    interior_ink_ratio: float = Field(ge=0, le=1)
    label_block_id: str | None = None
    label_chunk_id: str | None = None
    label_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    source_text: str | None = None
    ocr_score: float | None = Field(default=None, ge=0, le=1)
    ocr_grounding_unique: bool = False
    risks: list[str] = Field(default_factory=list)
    engine: Literal["OpenCV"] = "OpenCV"
    engine_version: str
    device: Literal["CPU"] = "CPU"


class LocalRedactionCandidate(BaseModel):
    """Pixel-grounded opaque-mask proposal requiring Luna confirmation."""

    model_config = ConfigDict(extra="forbid")
    id: str
    page: int = Field(ge=1)
    bbox: list[float] = Field(min_length=4, max_length=4)
    detector_score: float = Field(ge=0, le=1)
    fill_ratio: float = Field(ge=0, le=1)
    context: Literal["labeled_value", "standalone_line"]
    label_block_id: str | None = None
    engine: Literal["OpenCV"] = "OpenCV"
    engine_version: str
    device: Literal["CPU"] = "CPU"


class VisualReviewRegion(BaseModel):
    """A locally selected source-image region requiring high-detail Luna review."""

    model_config = ConfigDict(extra="forbid")
    id: str
    page: int = Field(ge=1)
    bbox: list[float] = Field(min_length=4, max_length=4)
    reason_codes: list[str] = Field(min_length=1)
    source_block_ids: list[str] = Field(default_factory=list)
    source_checkbox_ids: list[str] = Field(default_factory=list)
    source_redaction_ids: list[str] = Field(default_factory=list)
    source_layout_region_ids: list[str] = Field(default_factory=list)
    page_wide: bool = False


class LayoutRegion(BaseModel):
    """Immutable PP-DocLayoutV3 region evidence in page coordinates."""

    model_config = ConfigDict(extra="forbid")
    id: str
    page: int = Field(ge=1)
    cls_id: int = Field(ge=0)
    label: str
    score: float = Field(ge=0, le=1)
    coordinate: list[float] = Field(min_length=4, max_length=4)
    bbox: list[float] = Field(min_length=4, max_length=4)
    polygon: list[list[float]] = Field(default_factory=list)
    raw_order: int | None = None
    normalized_order: int | None = Field(default=None, ge=1)
    model: Literal["PP-DocLayoutV3"] = "PP-DocLayoutV3"


class LayoutBlockLink(BaseModel):
    """Deterministic association from one RapidOCR block to V3 regions."""

    model_config = ConfigDict(extra="forbid")
    page: int = Field(ge=1)
    block_id: str
    primary_region_id: str
    region_ids: list[str] = Field(min_length=1)
    block_coverage: float = Field(ge=0, le=1)


class ReadingOrderEvidence(BaseModel):
    """Normalized PP-DocLayoutV3 order with explicit coverage gaps."""

    model_config = ConfigDict(extra="forbid")
    page: int = Field(ge=1)
    source: Literal["PP-DocLayoutV3"] = "PP-DocLayoutV3"
    ordered_region_ids: list[str] = Field(default_factory=list)
    unordered_region_ids: list[str] = Field(default_factory=list)
    ordered_block_ids: list[str] = Field(default_factory=list)
    unmatched_block_ids: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    status: Literal["valid", "ambiguous"] = "valid"


class TableCellEvidence(BaseModel):
    """One locally detected table cell grounded in RapidOCR blocks."""

    model_config = ConfigDict(extra="forbid")
    id: str
    row: int = Field(ge=1)
    column: int = Field(ge=1)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    tag: Literal["th", "td"] = "td"
    bbox: list[float] = Field(min_length=4, max_length=4)
    source_block_ids: list[str] = Field(default_factory=list)
    source_word_ids: list[str] = Field(default_factory=list)
    source_text: str = ""
    raw_scores: list[float | None] = Field(default_factory=list)


class TableStructureEvidence(BaseModel):
    """Auditable local table structure kept separate from raw OCR evidence."""

    model_config = ConfigDict(extra="forbid")
    id: str
    page: int = Field(ge=1)
    layout_region_id: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    style: Literal["wired", "wireless"]
    classifier_score: float = Field(ge=0, le=1)
    structure_score: float = Field(ge=0, le=1)
    classifier_model: Literal["PP-LCNet_x1_0_table_cls"] = "PP-LCNet_x1_0_table_cls"
    structure_model: Literal["SLANeXt_wired", "SLANet_plus"]
    cells: list[TableCellEvidence] = Field(default_factory=list)
    markdown: str = ""
    status: Literal["valid", "invalid"]
    review_required: bool = False
    warnings: list[str] = Field(default_factory=list)


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
    local_vision_state: CheckboxState | None = None
    local_vision_score: float | None = Field(default=None, ge=0, le=1)
    local_vision_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    local_vision_id: str | None = None
    ocr_label_score: float | None = Field(default=None, ge=0, le=1)
    ocr_grounding_unique: bool = False
    agreement: Literal["consensus", "disagreement", "incomplete"] = "incomplete"


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
