"""Comparable wall-clock stage timings for bottleneck diagnosis.

Responsible for: defining standardized stage timing keys and measuring/summarizing
wall-clock durations across each pipeline phase to identify execution bottlenecks.

Must not: record negative or non-finite float durations, or conflate distinct
pipeline stages into one metric.

Next: `pipeline.py` and `workflow.py` which record these timings, and
`app_pages/diagnostics.py` which renders them.
"""

from __future__ import annotations

import math
from typing import Any

STAGE_TIMING_LABELS: tuple[tuple[str, str], ...] = (
    ("configuration_seconds", "OpenAI preflight"),
    ("rapidocr_initialization_seconds", "RapidOCR initialization"),
    ("layout_initialization_seconds", "Layout initialization"),
    ("document_ingest_seconds", "Document ingest and rendering"),
    ("page_preparation_seconds", "Quality analysis and preprocessing"),
    ("ocr_wall_seconds", "RapidOCR wall time"),
    ("ocr_retry_seconds", "RapidOCR retry"),
    ("layout_detection_seconds", "Layout detection and normalization"),
    ("table_structure_seconds", "Table structure"),
    ("routing_seconds", "Evidence routing"),
    ("checkbox_detection_seconds", "Checkbox detection"),
    ("redaction_detection_seconds", "Redaction detection"),
    ("visual_routing_seconds", "Visual review planning"),
    ("gpt_refinement_seconds", "GPT parse refinement"),
    ("refinement_merge_seconds", "Grounded refinement merge"),
    ("markdown_workflow_seconds", "Optional Markdown workflows"),
    ("checkbox_verification_seconds", "Checkbox verification"),
    ("workflow_validation_seconds", "Deterministic validation"),
    ("object_repair_seconds", "Bounded object repair"),
    ("result_finalization_seconds", "Canonical result finalization"),
    ("artifact_finalization_seconds", "Core artifact finalization"),
    ("annotated_pdf_generation_seconds", "Annotated PDF generation"),
    ("html_generation_seconds", "HTML generation"),
    ("zip_packaging_seconds", "ZIP packaging"),
)


def record_stage_timing(timings: dict[str, float], key: str, elapsed: float) -> None:
    """Accumulate a non-negative wall-clock duration."""
    if math.isfinite(elapsed) and elapsed >= 0:
        timings[key] = timings.get(key, 0.0) + elapsed


def stage_timing_summary(timings: dict[str, float]) -> dict[str, Any]:
    """Return exclusive stages ordered slowest-first with a named bottleneck."""
    measured: list[tuple[str, float]] = []
    for key, label in STAGE_TIMING_LABELS:
        value = timings.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            measured.append((label, float(value)))
    measured.sort(key=lambda item: item[1], reverse=True)
    total = sum(seconds for _, seconds in measured)
    stages = [
        {
            "stage": label,
            "seconds": round(seconds, 6),
            "share_percent": round(seconds / total * 100, 1) if total else 0.0,
        }
        for label, seconds in measured
    ]
    return {
        "measured_stage_seconds": round(total, 6),
        "pipeline_total_seconds": timings.get("total_seconds"),
        "bottleneck": stages[0] if stages else None,
        "stages": stages,
    }
