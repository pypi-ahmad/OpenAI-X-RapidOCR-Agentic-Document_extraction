from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_extractor.evaluation_data import (
    EvaluationDataError,
    EvaluationDataset,
    load_evaluation_dataset,
)

DATASET = Path(__file__).parents[1] / "evaluation-data" / "masked-amerigroup-pages-1-2.review.json"


def test_review_pack_has_object_level_table_and_checkbox_labels() -> None:
    dataset = load_evaluation_dataset(DATASET, require_human_approval=False)

    assert dataset.schema_version == "1.0"
    assert dataset.source.selected_pages == [1, 2]
    assert [
        (item.page, item.table_count, item.checkbox_count) for item in dataset.page_summaries
    ] == [
        (1, 0, 0),
        (2, 3, 25),
    ]
    assert sum(item.expected_cells for item in dataset.tables) == 38
    assert sum(item.state == "checked" for item in dataset.checkboxes) == 6
    assert sum(item.state == "unchecked" for item in dataset.checkboxes) == 19
    assert {item.group for item in dataset.checkboxes} == {
        "referring_provider",
        "servicing_provider",
        "servicing_facility",
        "type_of_service",
        "place_of_service",
    }
    assert len(dataset.rejected_checkbox_candidates) == 4


def test_draft_cannot_be_loaded_as_human_gold_data() -> None:
    with pytest.raises(EvaluationDataError, match="pending human review"):
        load_evaluation_dataset(DATASET)


def test_complete_human_approval_satisfies_strict_gate() -> None:
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    payload["approval"] = {
        "status": "human_approved",
        "reviewer_name": "Test Reviewer",
        "reviewed_at": "2026-08-31T10:00:00+05:30",
    }
    for section in ("tables", "checkboxes", "rejected_checkbox_candidates"):
        for item in payload[section]:
            item["review_status"] = "human_approved"

    dataset = EvaluationDataset.model_validate(payload)
    dataset.require_human_approval()


def test_approval_rejects_partially_reviewed_annotations() -> None:
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    payload["approval"] = {
        "status": "human_approved",
        "reviewer_name": "Test Reviewer",
        "reviewed_at": "2026-08-31T10:00:00+05:30",
    }

    with pytest.raises(ValidationError, match="Every annotation must be human-approved"):
        EvaluationDataset.model_validate(payload)


def test_invalid_normalized_bbox_is_rejected() -> None:
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    payload["checkboxes"][0]["bbox"] = [0.4, 0.2, 0.3, 0.25]

    with pytest.raises(ValidationError, match="positive width and height"):
        EvaluationDataset.model_validate(payload)
