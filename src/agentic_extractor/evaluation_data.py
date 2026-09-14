"""Versioned, provenance-safe table and checkbox evaluation datasets.

Responsible for: Pydantic schemas and disk loading/saving of human-approved
ground-truth benchmark review packs for table structures and checkboxes.
Enforces strict provenance (file SHA-256, reviewer name, timestamp,
normalized [0, 1] coordinates).

Must not: allow unapproved or pending review packs to be consumed as benchmark
gold data without explicit `human_approved` verification.

Next: `oracle_eval.py` for how these annotations are scored against runtime
extraction outputs, and `tools/run_corpus_validation.py` for batch execution.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

ReviewStatus = Literal["pending_human_review", "human_approved"]


class EvaluationDataError(ValueError):
    """Raised when evaluation data cannot be used as an approved gold set."""


class SourceDocument(BaseModel):
    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_pages: list[int] = Field(min_length=1)


class Approval(BaseModel):
    status: ReviewStatus
    reviewer_name: str | None = None
    reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def require_reviewer_for_approval(self) -> Self:
        if self.status == "human_approved" and not (
            self.reviewer_name and self.reviewer_name.strip() and self.reviewed_at
        ):
            raise ValueError("Human-approved data requires a reviewer name and review time.")
        return self


class PageSummary(BaseModel):
    page: int = Field(ge=1)
    table_count: int = Field(ge=0)
    checkbox_count: int = Field(ge=0)


class ReviewedAnnotation(BaseModel):
    id: str = Field(min_length=1)
    page: int = Field(ge=1)
    bbox: list[float] = Field(min_length=4, max_length=4)
    review_status: ReviewStatus

    @model_validator(mode="after")
    def validate_bbox(self) -> Self:
        left, top, right, bottom = self.bbox
        if not all(0 <= value <= 1 for value in self.bbox):
            raise ValueError("Normalized annotation coordinates must be between zero and one.")
        if left >= right or top >= bottom:
            raise ValueError("Annotation bbox must have positive width and height.")
        return self


class TableAnnotation(ReviewedAnnotation):
    semantic_role: str = Field(min_length=1)
    expected_rows: int = Field(ge=1)
    expected_columns: int = Field(ge=1)
    expected_cells: int = Field(ge=1)
    row_cell_counts: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grid_counts(self) -> Self:
        if len(self.row_cell_counts) != self.expected_rows:
            raise ValueError("Table row_cell_counts must contain one value per expected row.")
        if sum(self.row_cell_counts) != self.expected_cells:
            raise ValueError("Table row_cell_counts must sum to expected_cells.")
        if any(count < 1 or count > self.expected_columns for count in self.row_cell_counts):
            raise ValueError("Each table row count must fit the expected column count.")
        return self


class CheckboxAnnotation(ReviewedAnnotation):
    group: str = Field(min_length=1)
    label: str = Field(min_length=1)
    state: Literal["checked", "unchecked"]


class RejectedCheckboxCandidate(BaseModel):
    id: str = Field(min_length=1)
    page: int = Field(ge=1)
    source: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    review_status: ReviewStatus


class EvaluationDataset(BaseModel):
    schema_version: Literal["1.0"]
    dataset_id: str = Field(min_length=1)
    coordinate_space: Literal["normalized_top_left_xyxy"]
    source: SourceDocument
    approval: Approval
    candidate_provenance: list[str] = Field(min_length=1)
    page_summaries: list[PageSummary]
    tables: list[TableAnnotation]
    checkboxes: list[CheckboxAnnotation]
    rejected_checkbox_candidates: list[RejectedCheckboxCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dataset(self) -> Self:
        pages = set(self.source.selected_pages)
        if len(pages) != len(self.source.selected_pages):
            raise ValueError("Selected pages must be unique.")
        if {item.page for item in self.page_summaries} != pages:
            raise ValueError("Page summaries must cover every selected page exactly once.")
        annotations = [*self.tables, *self.checkboxes, *self.rejected_checkbox_candidates]
        if any(item.page not in pages for item in annotations):
            raise ValueError("Every annotation must belong to a selected page.")
        ids = [item.id for item in annotations]
        if len(ids) != len(set(ids)):
            raise ValueError("Annotation IDs must be unique.")
        summaries = {item.page: item for item in self.page_summaries}
        for page, summary in summaries.items():
            if summary.table_count != sum(item.page == page for item in self.tables):
                raise ValueError(f"Page {page} table summary does not match its annotations.")
            if summary.checkbox_count != sum(item.page == page for item in self.checkboxes):
                raise ValueError(f"Page {page} checkbox summary does not match its annotations.")
        if self.approval.status == "human_approved" and any(
            item.review_status != "human_approved" for item in annotations
        ):
            raise ValueError("Every annotation must be human-approved before dataset approval.")
        return self

    def require_human_approval(self) -> None:
        """Reject draft/model-assisted labels at the gold-set boundary."""
        if self.approval.status != "human_approved":
            raise EvaluationDataError(
                "Evaluation dataset is pending human review; it cannot be used as gold data."
            )


def load_evaluation_dataset(
    path: Path, *, require_human_approval: bool = True
) -> EvaluationDataset:
    """Load a dataset and optionally enforce explicit human adjudication."""
    dataset = EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))
    if require_human_approval:
        dataset.require_human_approval()
    return dataset
