"""Convert and validate cloud output against local document facts."""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator

from agentic_extractor.models import (
    Classification,
    EvidenceRef,
    Extraction,
    Section,
    Split,
)
from agentic_extractor.openai_refiner import CloudEvidence, CloudResult
from agentic_extractor.parse import PageParse


def validate_cloud_result(
    cloud: CloudResult,
    pages: list[PageParse],
    allowed_classes: list[str],
    schema: dict[str, Any] | None,
) -> tuple[list[Classification], list[Section], list[Split], Extraction | None, list[str]]:
    page_numbers = {page.page for page in pages}
    blocks = {block.id: block for page in pages for block in page.blocks}
    warnings: list[str] = []

    def evidence(items: list[CloudEvidence]) -> list[EvidenceRef]:
        valid: list[EvidenceRef] = []
        for item in items:
            source = "gpt-visual" if item.source == "gpt-visual" else "rapidocr"
            if item.page not in page_numbers or (item.block_id and item.block_id not in blocks):
                warnings.append(
                    "Cloud evidence referencing an unknown page or block was discarded."
                )
                continue
            if item.bbox and (
                len(item.bbox) != 4 or any(value < 0 or value > 1 for value in item.bbox)
            ):
                warnings.append("Cloud evidence with an invalid normalized box was discarded.")
                continue
            if item.block_id and item.quote and item.quote not in blocks[item.block_id].text:
                warnings.append("Cloud evidence with a non-matching quote was discarded.")
                continue
            valid.append(
                EvidenceRef(
                    page=item.page,
                    block_id=item.block_id,
                    bbox=item.bbox,
                    quote=item.quote,
                    source=source,
                )
            )
        return valid

    labels = set(allowed_classes) | {"unknown"}
    classifications = [
        Classification(
            label=item.label if item.label in labels else "unknown",
            page_start=item.page_start,
            page_end=item.page_end,
            evidence=evidence(item.evidence),
        )
        for item in cloud.classifications
        if _valid_range(item.page_start, item.page_end, page_numbers)
    ]
    sections = [
        Section(**item.model_dump())
        for item in cloud.sections
        if _valid_range(item.page_start, item.page_end, page_numbers)
    ]
    splits = [
        Split(**item.model_dump())
        for item in cloud.splits
        if _valid_range(item.page_start, item.page_end, page_numbers)
    ]
    splits = _valid_splits(splits, max(page_numbers), warnings)
    extraction = None
    if schema is not None:
        values = {
            item.path: _structured_value(
                item.value, schema.get("properties", {}).get(item.path, {})
            )
            for item in cloud.extracted_fields
        }
        field_evidence = {item.path: evidence(item.evidence) for item in cloud.extracted_fields}
        errors = [error.message for error in Draft202012Validator(schema).iter_errors(values)]
        extraction = Extraction(
            values=values, evidence=field_evidence, valid=not errors, errors=errors
        )
    return classifications, sections, splits, extraction, warnings


def _valid_range(start: int, end: int, pages: set[int]) -> bool:
    return start <= end and start in pages and end in pages


def _structured_value(value: Any, field_schema: dict[str, Any]) -> Any:
    expected = field_schema.get("type")
    if expected not in {"array", "object"} or not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    if (expected == "array" and isinstance(decoded, list)) or (
        expected == "object" and isinstance(decoded, dict)
    ):
        return decoded
    return value


def _valid_splits(splits: list[Split], page_count: int, warnings: list[str]) -> list[Split]:
    ordered = sorted(splits, key=lambda item: item.page_start)
    if not ordered:
        return []
    expected = 1
    for item in ordered:
        if item.page_start != expected:
            warnings.append("Invalid split coverage was discarded.")
            return []
        expected = item.page_end + 1
    if expected != page_count + 1:
        warnings.append("Incomplete split coverage was discarded.")
        return []
    return ordered
