"""Conservative local detection of visible opaque redaction masks.

Responsible for: proposing likely black-box redactions from pixel fill,
geometry, and layout/OCR context. Deliberately biased toward false negatives
over false positives — the thresholds below (fill ratio, aspect ratio, size as
a fraction of page, vertical text context on both sides) are tuned to avoid
misflagging tables, figures, or normal underlines as redactions, at the cost
of missing some real ones. Must not: publish a proposal directly — every
candidate still requires Luna review before it can appear in output (see
`pipeline.py`, which calls `detect_page_redactions` and routes candidates
through `visual_routing.py` for a high-detail crop).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image

from agentic_extractor.models import Block, LocalRedactionCandidate
from agentic_extractor.parse import PageParse

_LABEL_PATTERN = re.compile(r"^(?:patient\s*name|dob)\s*:", re.IGNORECASE)
_EXCLUDED_LAYOUT_LABELS = {"algorithm", "chart", "figure", "image", "table"}


@dataclass(frozen=True, slots=True)
class _Detection:
    bbox: tuple[int, int, int, int]
    fill_ratio: float
    context: Literal["labeled_value", "standalone_line"]
    label_block_id: str | None = None


def detect_page_redactions(page: PageParse) -> list[LocalRedactionCandidate]:
    """Propose likely black masks; proposals are not publishable without Luna review.

    Two independent detectors feed one dedup pass: `_labeled_detections` looks
    for a dark run immediately after a known PII label (patient name, DOB) on
    the same line, and `_standalone_detections` looks for dark bars with text
    above and below but not overlapping OCR (i.e. not a table rule or a real
    printed bar). Overlapping proposals (IoU >= 0.35) are deduped, keeping
    whichever was found first.
    """
    source = page.original_image_bytes or page.layout_image_bytes or page.image_bytes
    image = Image.open(io.BytesIO(source)).convert("L")
    gray = np.asarray(image)
    height, width = gray.shape
    dark = gray < 70
    detections = _labeled_detections(dark, page.blocks, width, height)
    detections.extend(_standalone_detections(dark, page, width, height))

    unique: list[_Detection] = []
    for detection in detections:
        if any(_iou(detection.bbox, existing.bbox) >= 0.35 for existing in unique):
            continue
        unique.append(detection)
    unique.sort(key=lambda item: (item.bbox[1], item.bbox[0]))

    return [
        LocalRedactionCandidate(
            id=f"p{page.page}-red{ordinal}",
            page=page.page,
            bbox=_normalize(detection.bbox, width, height),
            detector_score=min(1.0, 0.55 + detection.fill_ratio * 0.45),
            fill_ratio=detection.fill_ratio,
            context=detection.context,
            label_block_id=detection.label_block_id,
            engine_version=cv2.__version__,
        )
        for ordinal, detection in enumerate(unique, 1)
    ]


def _labeled_detections(
    dark: np.ndarray, blocks: list[Block], width: int, height: int
) -> list[_Detection]:
    output: list[_Detection] = []
    for block in blocks:
        if not block.bbox or not _LABEL_PATTERN.search(block.text.strip()):
            continue
        left, top, right, bottom = _pixels(block.bbox, width, height)
        block_height = max(1, bottom - top)
        line_margin = max(1, block_height // 10)
        strip_top = max(0, top - line_margin)
        strip_bottom = min(height, bottom + line_margin)
        search_left = max(0, right - round(width * 0.025))
        search_right = min(width, search_left + round(width * 0.40))
        strip = dark[strip_top:strip_bottom, search_left:search_right]
        if not strip.size:
            continue
        density = strip.mean(axis=0)
        dense_columns = cv2.morphologyEx(
            (density > 0.38).astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((1, 9), dtype=np.uint8),
        ).ravel()
        for run_left, run_right in _runs(dense_columns):
            if run_right - run_left < round(width * 0.025):
                continue
            crop = strip[:, run_left:run_right]
            ys, xs = np.where(crop)
            if not len(xs):
                continue
            bbox = (
                search_left + run_left + int(xs.min()),
                strip_top + int(ys.min()),
                search_left + run_left + int(xs.max()) + 1,
                strip_top + int(ys.max()) + 1,
            )
            fill = float(dark[bbox[1] : bbox[3], bbox[0] : bbox[2]].mean())
            if fill >= 0.35:
                output.append(_Detection(bbox, fill, "labeled_value", block.id))
                break
    return output


def _standalone_detections(
    dark: np.ndarray, page: PageParse, width: int, height: int
) -> list[_Detection]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    output: list[_Detection] = []
    for x, y, box_width, box_height, area in stats[1:count]:
        fill = float(area / max(1, box_width * box_height))
        aspect = box_width / max(1, box_height)
        bbox = (int(x), int(y), int(x + box_width), int(y + box_height))
        normalized = _normalize(bbox, width, height)
        if not (
            0.03 <= box_width / width <= 0.40
            and 0.006 <= box_height / height <= 0.04
            and 3.0 <= aspect <= 30.0
            and fill >= 0.65
            and normalized[1] >= 0.055
            and normalized[0] >= 0.02
            and normalized[2] <= 0.98
        ):
            continue
        if _overlaps_ocr(normalized, page.blocks) or _inside_excluded_layout(normalized, page):
            continue
        if not _has_vertical_text_context(normalized, page.blocks):
            continue
        output.append(_Detection(bbox, fill, "standalone_line"))
    return output


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    start: int | None = None
    last = -1
    for index, value in enumerate(values):
        if value:
            if start is None:
                start = index
            last = index
        elif start is not None and index - last > 4:
            output.append((start, last + 1))
            start = None
    if start is not None:
        output.append((start, last + 1))
    return output


def _overlaps_ocr(candidate: list[float], blocks: list[Block]) -> bool:
    area = _area(candidate)
    return any(
        block.bbox and _intersection(candidate, block.bbox) / area > 0.12 for block in blocks
    )


def _inside_excluded_layout(candidate: list[float], page: PageParse) -> bool:
    center_x = (candidate[0] + candidate[2]) / 2
    center_y = (candidate[1] + candidate[3]) / 2
    return any(
        region.label.lower() in _EXCLUDED_LAYOUT_LABELS
        and region.bbox[0] <= center_x <= region.bbox[2]
        and region.bbox[1] <= center_y <= region.bbox[3]
        for region in page.layout_regions
    )


def _has_vertical_text_context(candidate: list[float], blocks: list[Block]) -> bool:
    above = False
    below = False
    candidate_width = candidate[2] - candidate[0]
    for block in blocks:
        if not block.bbox or not block.text.strip():
            continue
        overlap = max(0.0, min(candidate[2], block.bbox[2]) - max(candidate[0], block.bbox[0]))
        if overlap / max(candidate_width, 1e-9) < 0.30:
            continue
        above_gap = candidate[1] - block.bbox[3]
        below_gap = block.bbox[1] - candidate[3]
        above = above or -0.002 <= above_gap <= 0.025
        below = below or -0.002 <= below_gap <= 0.025
    return above and below


def _pixels(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        round(bbox[0] * width),
        round(bbox[1] * height),
        round(bbox[2] * width),
        round(bbox[3] * height),
    )


def _normalize(bbox: tuple[int, int, int, int], width: int, height: int) -> list[float]:
    return [
        round(bbox[0] / width, 6),
        round(bbox[1] / height, 6),
        round(bbox[2] / width, 6),
        round(bbox[3] / height, 6),
    ]


def _intersection(left: list[float], right: list[float]) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _area(bbox: list[float]) -> float:
    return max(1e-9, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return intersection / union if union else 0.0
