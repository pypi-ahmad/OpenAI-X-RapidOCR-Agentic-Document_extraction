"""Deterministic local checkbox proposals from pixels plus RapidOCR label grounding.

Responsible for: proposing checkbox controls from OpenCV geometry/pixel fill and
grounding each one to a unique nearby RapidOCR label. Must not: decide whether a
proposal is safe to auto-apply — that is a three-way agreement across this
module, an independent Sol crop verification, and OCR label confidence, decided
in `visual_routing.py` / `openai_refiner.py`. This module only supplies one of
the three signals. Next: `visual_routing.py`, which turns
`is_automation_eligible_checkbox` candidates into high-detail Sol crops.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from agentic_extractor.models import Block, CheckboxState, LocalCheckboxCandidate
from agentic_extractor.parse import PageParse, build_layout_chunks

_MIN_OCR_SCORE = 0.85
_MIN_DETECTOR_SCORE = 0.70
_MIN_BORDER_COVERAGE = 0.27
_MIN_SQUARE_ASPECT = 0.85
_MAX_SQUARE_ASPECT = 1 / _MIN_SQUARE_ASPECT


@dataclass(frozen=True, slots=True)
class _Detection:
    bbox: tuple[int, int, int, int]
    state: CheckboxState
    detector_score: float
    border_coverage: float
    interior_ink_ratio: float


def detect_page_checkboxes(page: PageParse) -> list[LocalCheckboxCandidate]:
    """Find checkbox-shaped controls and link them to unique nearby OCR labels."""
    image = Image.open(io.BytesIO(page.original_image_bytes or page.image_bytes)).convert("L")
    gray = np.asarray(image)
    dark_ink = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )
    light_ink = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12,
    )
    # Fax compression and skew often leave one- or two-pixel breaks in an
    # otherwise square control. Close only those tiny gaps; later geometry,
    # OCR grounding, and independent crop review still gate publication.
    closed_dark_ink = cv2.morphologyEx(dark_ink, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    shortest = min(page.width, page.height)
    minimum = max(8, round(shortest * 0.01))
    maximum = max(minimum + 1, round(shortest * 0.25))
    detections: list[_Detection] = []
    for contour_binary in (dark_ink, light_ink, closed_dark_ink):
        contours, _ = cv2.findContours(contour_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            x, y, width, height = cv2.boundingRect(contour)
            if (
                len(polygon) != 4
                or width < minimum
                or height < minimum
                or width > maximum
                or height > maximum
                or not _MIN_SQUARE_ASPECT <= width / height <= _MAX_SQUARE_ASPECT
            ):
                continue
            border, _ = _ink_measurements(contour_binary, x, y, width, height)
            _, interior = _ink_measurements(dark_ink, x, y, width, height)
            if border < _MIN_BORDER_COVERAGE:
                continue
            state = _state(interior)
            shape = 1 - min(1.0, abs(width - height) / max(width, height))
            certainty = 1.0 if state in {CheckboxState.CHECKED, CheckboxState.UNCHECKED} else 0.5
            score = min(1.0, 0.45 * shape + 0.35 * border + 0.2 * certainty)
            detection = _Detection((x, y, x + width, y + height), state, score, border, interior)
            if any(
                _pixel_iou(detection.bbox, item.bbox) > 0.6
                or _pixel_smaller_box_coverage(detection.bbox, item.bbox) > 0.8
                for item in detections
            ):
                continue
            detections.append(detection)

    detections.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    chunks = page.chunks or build_layout_chunks(page.blocks)
    results: list[LocalCheckboxCandidate] = []
    for ordinal, detection in enumerate(detections, 1):
        normalized = _normalize(detection.bbox, page.width, page.height)
        block, unique = _nearest_label(normalized, page.blocks)
        chunk = next(
            (item for item in chunks if block is not None and block.id in item.source_block_ids),
            None,
        )
        risks: list[str] = []
        if block is None:
            risks.append("missing RapidOCR label")
        elif not unique:
            risks.append("ambiguous RapidOCR label")
        if _inside_table_region(normalized, page):
            risks.append("inside a table region")
        if detection.state not in {CheckboxState.CHECKED, CheckboxState.UNCHECKED}:
            risks.append(f"local state is {detection.state.value}")
        results.append(
            LocalCheckboxCandidate(
                id=f"p{page.page}-cv{ordinal}",
                page=page.page,
                state=detection.state,
                control_bbox=normalized,
                detector_score=detection.detector_score,
                border_coverage=detection.border_coverage,
                interior_ink_ratio=detection.interior_ink_ratio,
                label_block_id=block.id if block else None,
                label_chunk_id=chunk.id if chunk else None,
                label_bbox=list(block.bbox) if block and block.bbox else None,
                source_text=(
                    _semantic_label_text(block, normalized, detection.state) if block else None
                ),
                ocr_score=block.ocr_score if block else None,
                ocr_grounding_unique=unique,
                risks=risks,
                engine_version=cv2.__version__,
            )
        )
    return results


def is_automation_eligible_checkbox(page: PageParse, candidate: LocalCheckboxCandidate) -> bool:
    """Return whether a local proposal can satisfy the automatic consensus contract.

    Stricter than `is_credible_checkbox_candidate`: eligibility additionally
    requires zero recorded risks (ambiguous label, missing label, inside a
    table, indeterminate local state). Credible-but-risky candidates still get
    a Sol crop review; only risk-free ones can end up auto-accepted, and even
    then only if Sol's independent read agrees (see `openai_refiner.py`).
    """
    return bool(is_credible_checkbox_candidate(page, candidate) and not candidate.risks)


def is_credible_checkbox_candidate(page: PageParse, candidate: LocalCheckboxCandidate) -> bool:
    """Return whether local pixels and OCR justify specialized checkbox review.

    This is the "worth sending to Sol" gate, not the "safe to auto-accept"
    gate — see `is_automation_eligible_checkbox` for the stricter contract.
    """
    return bool(
        is_visual_checkbox_candidate(page, candidate)
        and candidate.ocr_grounding_unique
        and candidate.ocr_score is not None
        and candidate.ocr_score >= _MIN_OCR_SCORE
        and (candidate.label_block_id or candidate.label_chunk_id)
        and bool(candidate.source_text and candidate.source_text.strip())
        and any(character.isalpha() for character in candidate.source_text or "")
    )


def is_visual_checkbox_candidate(page: PageParse, candidate: LocalCheckboxCandidate) -> bool:
    """Return whether pixels support sending a local shape to Sol for adjudication."""
    left, top, right, bottom = candidate.control_bbox
    width = (right - left) * page.width
    height = (bottom - top) * page.height
    aspect = width / height if height > 0 else 0.0
    return bool(
        0 <= left < right <= 1
        and 0 <= top < bottom <= 1
        and _MIN_SQUARE_ASPECT <= aspect <= _MAX_SQUARE_ASPECT
        and candidate.detector_score >= _MIN_DETECTOR_SCORE
        and candidate.state in {CheckboxState.CHECKED, CheckboxState.UNCHECKED}
    )


def _ink_measurements(
    binary: np.ndarray, x: int, y: int, width: int, height: int
) -> tuple[float, float]:
    crop = binary[y : y + height, x : x + width]
    # Inset margin: 20% inset isolates interior check/fill marks from the box's outer border frame.
    margin = max(2, round(min(width, height) * 0.2))
    interior = crop[margin:-margin, margin:-margin]
    border_mask = np.ones(crop.shape, dtype=bool)
    border_mask[margin:-margin, margin:-margin] = False
    border = float(np.count_nonzero(crop[border_mask])) / max(1, int(border_mask.sum()))
    inner = float(np.count_nonzero(interior)) / max(1, int(interior.size))
    return min(border, 1.0), min(inner, 1.0)


def _state(interior_ink_ratio: float) -> CheckboxState:
    # Empirical ink thresholds: <= 6% indicates empty box; >= 10% indicates mark ink;
    # intermediate values abstain as NOT_DETERMINABLE.
    if interior_ink_ratio <= 0.06:
        return CheckboxState.UNCHECKED
    if interior_ink_ratio >= 0.10:
        return CheckboxState.CHECKED
    return CheckboxState.NOT_DETERMINABLE


def _nearest_label(control: list[float], blocks: list[Block]) -> tuple[Block | None, bool]:
    top, bottom = control[1], control[3]
    center = (top + bottom) / 2
    control_height = bottom - top
    ranked: list[tuple[float, Block]] = []
    for block in blocks:
        if not block.bbox or not block.text.strip():
            continue
        if block.text.strip().casefold() in {"x", "✓", "✔", "☑"}:
            # OCR may isolate the selection mark as its own text block. It is
            # control evidence, not the semantic label beside the control.
            continue
        block_center = (block.bbox[1] + block.bbox[3]) / 2
        vertical_distance = abs(center - block_center)
        gap = block.bbox[0] - control[2]
        overlap = max(0.0, min(bottom, block.bbox[3]) - max(top, block.bbox[1]))
        min_height = min(control_height, block.bbox[3] - block.bbox[1])
        overlap_ratio = overlap / min_height if min_height > 0 else 0.0
        block_width = block.bbox[2] - block.bbox[0]
        relative_control_center = (
            ((control[0] + control[2]) / 2 - block.bbox[0]) / block_width
            if block_width > 0
            else 0.0
        )
        control_inside_block = (
            block.bbox[0] <= control[0]
            and block.bbox[2] >= control[2]
            and relative_control_center >= 0.55
        )
        if (
            # RapidOCR commonly includes the checkbox outline in the label line's
            # polygon, creating a small negative gap for the correct label.
            (not control_inside_block and (gap < -0.03 or gap > 0.04))
            or overlap_ratio < 0.35
            or vertical_distance > max(0.03, control_height)
        ):
            continue
        horizontal_distance = 0.0 if control_inside_block else gap
        ranked.append((horizontal_distance + vertical_distance * 2, block))
    ranked.sort(key=lambda item: item[0])
    if not ranked:
        return None, False
    best_distance, best = ranked[0]
    separated = len(ranked) == 1 or ranked[1][0] - best_distance >= 0.03
    return best, bool(separated and best.ocr_score is not None and best.ocr_score >= _MIN_OCR_SCORE)


def _normalize(bbox: tuple[int, int, int, int], width: int, height: int) -> list[float]:
    return [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]


def _semantic_label_text(block: Block, control: list[float], state: CheckboxState) -> str:
    """Remove an OCR-fused selection mark while preserving immutable block text."""
    text = block.text.strip()
    if (
        state is CheckboxState.CHECKED
        and block.bbox
        and re.match(r"^[Xx](?=[A-Z])", text)
        and control[0] - 0.01 <= block.bbox[0] < control[2]
        and control[2] - block.bbox[0] <= 0.03
    ):
        return text[1:].lstrip()
    return text


def _inside_table_region(bbox: list[float], page: PageParse) -> bool:
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return any(
        region.label == "table"
        and region.bbox[0] <= center_x <= region.bbox[2]
        and region.bbox[1] <= center_y <= region.bbox[3]
        for region in page.layout_regions
    )


def _pixel_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    intersection_width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _pixel_smaller_box_coverage(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    intersection_width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    smaller = min(left_area, right_area)
    return intersection / smaller if smaller else 0.0
