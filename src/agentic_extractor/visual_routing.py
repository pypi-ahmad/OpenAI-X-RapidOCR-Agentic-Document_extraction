"""Deterministic local routing for high-detail Luna image regions.

Responsible for: deciding, from local evidence alone (OCR confidence, risky
text patterns, checkbox/redaction candidates, complex layout regions, page
quality), which page regions get a high-detail crop sent to Luna versus a
low-detail page overview. Must not: decide checkbox/table acceptance itself —
it only decides what Luna gets to look at; acceptance happens in
`openai_refiner.py` and the merge logic in `pipeline.py`. Next:
`openai_refiner.py`, which turns these regions into actual image content in
the Luna request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import hypot
from typing import Any

from agentic_extractor.checkbox_vision import is_automation_eligible_checkbox
from agentic_extractor.models import VisualReviewRegion
from agentic_extractor.parse import LOW_CONFIDENCE_THRESHOLD, PageParse

_QUALITY_REASONS = (
    "blur_warning",
    "low_contrast",
    "shadow_warning",
    "compression_artifacts_likely",
    "cropped_edge_warning",
)
_HIGH_RISK_OCR_PATTERN = re.compile(
    r"(?:\d\|\d|\d:\d|>/=|</=|\b(?:mcg|mg|ml|mmhg)\b[^\x00-\x7f]"
    r"|\b[A-Z]\d{2}(?:\.\d{1,4})?\b"
    r"|\b1\d{2}\.\d{1,4}(?=\s*-\s*\d{3}(?:\.\d+)?\b))",
    re.IGNORECASE,
)
_MAX_LOCALIZED_REGION_AREA = 0.15


@dataclass(slots=True)
class _Candidate:
    bbox: list[float]
    reasons: set[str] = field(default_factory=set)
    block_ids: set[str] = field(default_factory=set)
    checkbox_ids: set[str] = field(default_factory=set)
    redaction_ids: set[str] = field(default_factory=set)
    layout_region_ids: set[str] = field(default_factory=set)


def block_requires_gpt_review(block: Any) -> bool:
    """Flag low-confidence or character-sensitive OCR for mandatory Luna review."""
    return bool(
        (block.ocr_score is not None and block.ocr_score < LOW_CONFIDENCE_THRESHOLD)
        or _HIGH_RISK_OCR_PATTERN.search(block.text)
    )


def plan_visual_review_regions(
    page: PageParse,
    *,
    quality: dict[str, Any] | None = None,
    forced: bool = False,
    max_regions: int = 12,
) -> list[VisualReviewRegion]:
    """Plan high-detail crops from local evidence without dropping uncertainty."""
    if forced:
        return [_region(page.page, 1, _page_candidate("user_forced"))]
    if page.status == "failed" or not page.blocks:
        return [_region(page.page, 1, _page_candidate("failed_or_empty_ocr"))]

    candidates: list[_Candidate] = []
    blocks = {block.id: block for block in page.blocks}
    missing_uncertain_geometry = False
    for block in page.blocks:
        reasons: set[str] = set()
        if block.ocr_score is None or block.ocr_score < LOW_CONFIDENCE_THRESHOLD:
            reasons.add("low_ocr_confidence")
        if _HIGH_RISK_OCR_PATTERN.search(block.text):
            reasons.add("high_risk_ocr_pattern")
        if not reasons:
            continue
        if not _valid_bbox(block.bbox):
            missing_uncertain_geometry = True
            continue
        assert block.bbox is not None
        candidates.append(_Candidate(_pad(block.bbox), reasons, block_ids={block.id}))

    for checkbox in page.local_checkbox_candidates:
        if not is_automation_eligible_checkbox(page, checkbox):
            continue
        bbox = list(checkbox.control_bbox)
        block_ids: set[str] = set()
        if checkbox.label_block_id and checkbox.label_block_id in blocks:
            label_bbox = blocks[checkbox.label_block_id].bbox
            if _valid_bbox(label_bbox):
                assert label_bbox is not None
                bbox = _union_bbox(bbox, label_bbox)
                block_ids.add(checkbox.label_block_id)
        candidates.append(
            _Candidate(
                _pad(bbox),
                {"checkbox_candidate"},
                block_ids=block_ids,
                checkbox_ids={checkbox.id},
            )
        )

    for redaction in page.local_redaction_candidates:
        candidates.append(
            _Candidate(
                _pad(redaction.bbox),
                {"redaction_candidate"},
                block_ids={redaction.label_block_id} if redaction.label_block_id else set(),
                redaction_ids={redaction.id},
            )
        )

    linked_regions = {
        region_id for link in page.layout_block_links for region_id in link.region_ids
    }
    complex_labels = {"algorithm", "chart", "display_formula", "inline_formula"}
    for region in page.layout_regions:
        reasons: set[str] = set()
        if region.label in complex_labels:
            reasons.add("complex_layout_region")
        # Table candidates have their own exact-crop review call. Low layout-model
        # confidence is not OCR uncertainty and must not create a second large crop.
        if reasons and region.id not in linked_regions:
            reasons.add("layout_region_without_ocr")
        if reasons:
            candidates.append(
                _Candidate(
                    _pad(region.bbox),
                    reasons,
                    layout_region_ids={region.id},
                )
            )

    page_reasons = {
        reason for reason in _QUALITY_REASONS if quality and quality.get(reason) is True
    }
    if quality:
        skew = quality.get("skew_degrees")
        if isinstance(skew, int | float) and abs(skew) >= 1:
            page_reasons.add("skew_warning")
        dpi = quality.get("estimated_dpi")
        if isinstance(dpi, int | float) and dpi < 150:
            page_reasons.add("low_dpi")
    if page.layout_signals.get("ambiguous"):
        page_reasons.add("ambiguous_reading_order")
    if page.layout_signals.get("columns_detected") == 2:
        page_reasons.add("multi_column")
    if page_reasons and candidates:
        for candidate in candidates:
            candidate.reasons.update(page_reasons)

    if missing_uncertain_geometry:
        reasons = {"missing_uncertain_geometry"}
        reasons.update(reason for item in candidates for reason in item.reasons)
        return [_region(page.page, 1, _page_candidate(*reasons))]

    merged = _merge_touching(candidates)
    # max_regions is a soft target, not a hard cap: once no more pairs can merge
    # within _MAX_LOCALIZED_REGION_AREA (_nearest_pair returns -1), the loop stops
    # even if len(merged) is still above max_regions. The oversized-region tiling
    # below can then push the final count back above max_regions again — no
    # candidate region is ever dropped to satisfy the target.
    while len(merged) > max_regions:
        left_index, right_index = _nearest_pair(merged)
        if left_index < 0:
            break
        combined = _merge(merged[left_index], merged[right_index])
        merged = [
            item for index, item in enumerate(merged) if index not in {left_index, right_index}
        ]
        merged.append(combined)
        merged = _merge_touching(merged)
    merged = [tile for candidate in merged for tile in _tile_oversized(candidate)]
    merged.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return [_region(page.page, index, item) for index, item in enumerate(merged, 1)]


def _merge_touching(candidates: list[_Candidate]) -> list[_Candidate]:
    pending = list(candidates)
    output: list[_Candidate] = []
    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            for index, other in enumerate(pending):
                if _should_merge(current.bbox, other.bbox):
                    current = _merge(current, pending.pop(index))
                    changed = True
                    break
        output.append(current)
    return output


def _nearest_pair(candidates: list[_Candidate]) -> tuple[int, int]:
    pairs = [
        (_center_distance(left.bbox, right.bbox), left_index, right_index)
        for left_index, left in enumerate(candidates)
        for right_index, right in enumerate(candidates[left_index + 1 :], left_index + 1)
        if _area(_union_bbox(left.bbox, right.bbox)) <= _MAX_LOCALIZED_REGION_AREA
    ]
    return min(pairs, key=lambda item: item[0])[1:] if pairs else (-1, -1)


def _center_distance(left: list[float], right: list[float]) -> float:
    return hypot(
        (left[0] + left[2] - right[0] - right[2]) / 2,
        (left[1] + left[3] - right[1] - right[3]) / 2,
    )


def _gap(left: list[float], right: list[float]) -> float:
    horizontal = max(0.0, max(left[0], right[0]) - min(left[2], right[2]))
    vertical = max(0.0, max(left[1], right[1]) - min(left[3], right[3]))
    return hypot(horizontal, vertical)


def _should_merge(left: list[float], right: list[float]) -> bool:
    if _gap(left, right) > 0:
        return False
    union = _union_bbox(left, right)
    combined_area = _area(left) + _area(right)
    return _area(union) <= _MAX_LOCALIZED_REGION_AREA and _area(union) <= combined_area * 1.5


def _area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _merge(left: _Candidate, right: _Candidate) -> _Candidate:
    return _Candidate(
        _union_bbox(left.bbox, right.bbox),
        left.reasons | right.reasons,
        left.block_ids | right.block_ids,
        left.checkbox_ids | right.checkbox_ids,
        left.redaction_ids | right.redaction_ids,
        left.layout_region_ids | right.layout_region_ids,
    )


def _tile_oversized(candidate: _Candidate) -> list[_Candidate]:
    # Area bisection invariant: localized crops exceeding 15% of page area are
    # recursively split along their longer dimension to preserve token budget.
    pending = [candidate]
    output: list[_Candidate] = []
    while pending:
        current = pending.pop()
        if _area(current.bbox) <= _MAX_LOCALIZED_REGION_AREA:
            output.append(current)
            continue
        left, top, right, bottom = current.bbox
        if right - left >= bottom - top:
            middle = (left + right) / 2
            boxes = ([left, top, middle, bottom], [middle, top, right, bottom])
        else:
            middle = (top + bottom) / 2
            boxes = ([left, top, right, middle], [left, middle, right, bottom])
        for bbox in boxes:
            pending.append(
                _Candidate(
                    bbox,
                    current.reasons | {"oversized_region_tiled"},
                    set(current.block_ids),
                    set(current.checkbox_ids),
                    set(current.redaction_ids),
                    set(current.layout_region_ids),
                )
            )
    return output


def _page_candidate(*reasons: str) -> _Candidate:
    return _Candidate([0.0, 0.0, 1.0, 1.0], set(reasons))


def _region(page: int, ordinal: int, candidate: _Candidate) -> VisualReviewRegion:
    bbox = [round(value, 6) for value in candidate.bbox]
    return VisualReviewRegion(
        id=f"p{page}-vr{ordinal}",
        page=page,
        bbox=bbox,
        reason_codes=sorted(candidate.reasons),
        source_block_ids=sorted(candidate.block_ids),
        source_checkbox_ids=sorted(candidate.checkbox_ids),
        source_redaction_ids=sorted(candidate.redaction_ids),
        source_layout_region_ids=sorted(candidate.layout_region_ids),
        page_wide=bbox == [0.0, 0.0, 1.0, 1.0],
    )


def _pad(bbox: list[float], amount: float = 0.02) -> list[float]:
    # Context padding: expand crops by 2% normalized page margin (clamped [0, 1])
    # so Luna sees immediate surrounding context for characters/controls.
    return [
        max(0.0, bbox[0] - amount),
        max(0.0, bbox[1] - amount),
        min(1.0, bbox[2] + amount),
        min(1.0, bbox[3] + amount),
    ]


def _envelope(boxes: list[list[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _union_bbox(left: list[float], right: list[float]) -> list[float]:
    return [
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    ]


def _valid_bbox(bbox: list[float] | None) -> bool:
    return bool(
        bbox
        and len(bbox) == 4
        and all(0 <= value <= 1 for value in bbox)
        and bbox[0] < bbox[2]
        and bbox[1] < bbox[3]
    )
