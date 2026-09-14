"""Evidence-based page quality diagnostics and bounded batch backoff.

Responsible for: diagnosing raw page image quality (sharpness, blur, contrast,
skew angle) and applying strictly verifiable preprocessing (deskewing,
auto-contrast) only if metrics improve. Also manages bounded batch backoff.

Must not: apply speculative or degrading image transforms without verifying
that target diagnostic metrics improved.

Next: `pipeline.py`, which invokes `preprocess_if_improved` prior to OCR.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field


class PageQuality(BaseModel):
    page: int = Field(ge=1)
    pixel_dimensions: list[int]
    estimated_dpi: int | None = None
    physical_size_inches: list[float] | None = None
    skew_degrees: float | None = None
    rotation_indication: str = "not measured; no reliable local detector configured"
    layout_strategy: str = "normal_detection_and_geometry_layout"
    segmentation_strategy: str = (
        "No RapidOCR segmentation API; deterministic local geometry reconstruction"
    )
    sharpness_score: float
    blur_warning: bool
    contrast_score: float
    low_contrast: bool
    shadow_warning: bool
    compression_artifacts_likely: bool
    cropped_edge_warning: bool
    recommendations: list[str] = Field(default_factory=list)
    preprocessing_actions: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class BatchAudit:
    initial_batch_size: int
    final_batch_size: int
    retries: int = 0
    attempts: list[dict[str, int]] = field(default_factory=list)


def preprocess_if_improved(
    page: int,
    image: Image.Image,
    diagnostic: PageQuality,
) -> tuple[Image.Image, list[str]]:
    """Retain transforms only after their corresponding diagnostic improves."""
    # Measurable improvement invariant: image transforms are rolled back unless
    # post-transform diagnostics confirm measurable improvement (e.g. skew reduced by >= 0.5°).
    candidate = image
    current = diagnostic
    actions: list[str] = []
    if diagnostic.skew_degrees is not None and abs(diagnostic.skew_degrees) >= 1:
        deskewed = image.rotate(
            -diagnostic.skew_degrees,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor="white",
        )
        deskewed_quality = analyze_page(page, deskewed)
        remaining_skew = abs(deskewed_quality.skew_degrees or 0)
        if remaining_skew + 0.5 < abs(diagnostic.skew_degrees):
            candidate = deskewed
            current = deskewed_quality
            actions.append(
                f"deskew retained: measured skew reduced from {diagnostic.skew_degrees:.1f}° "
                f"to {remaining_skew:.1f}°"
            )
    if current.low_contrast:
        normalized = ImageOps.autocontrast(candidate.convert("RGB"), cutoff=1)
        normalized_quality = analyze_page(page, normalized)
        if normalized_quality.contrast_score > current.contrast_score + 5:
            candidate = normalized
            actions.append(
                "contrast normalization retained: measured contrast score increased "
                f"from {current.contrast_score:.3f} to "
                f"{normalized_quality.contrast_score:.3f}"
            )
    return candidate, actions


def analyze_page(
    page: int,
    image: Image.Image,
    *,
    dpi: tuple[float, float] | None = None,
    source_format: str | None = None,
    orientation_correction_degrees: int = 0,
) -> PageQuality:
    """Estimate diagnostic signals; scores are heuristics, not accuracy measurements."""
    gray = np.asarray(image.convert("L").resize((min(image.width, 1200), min(image.height, 1200))))
    gray_f = gray.astype(np.float32)
    gx = np.diff(gray_f, axis=1)
    gy = np.diff(gray_f, axis=0)
    sharpness = float((np.var(gx) + np.var(gy)) / 2)
    contrast = float(np.std(gray_f))
    edge = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    cropped = float(np.mean(edge < 48)) > 0.08
    corners = np.array(
        [
            gray[:50, :50].mean(),
            gray[:50, -50:].mean(),
            gray[-50:, :50].mean(),
            gray[-50:, -50:].mean(),
        ]
    )
    shadow = float(np.ptp(corners)) > 45
    blockiness = _blockiness(gray_f)
    skew = _estimate_skew(Image.fromarray(gray))
    estimated_dpi = round((dpi[0] + dpi[1]) / 2) if dpi and all(dpi) else None
    physical = (
        [round(image.width / dpi[0], 2), round(image.height / dpi[1], 2)]
        if dpi and all(dpi)
        else None
    )
    recommendations: list[str] = []
    if estimated_dpi is None or estimated_dpi < 300:
        recommendations.append("For rescanning, use 300–400 DPI when practical.")
    if sharpness < 35:
        recommendations.append("Page appears blurred; rescan with stable focus and no motion.")
    if contrast < 25:
        recommendations.append(
            "Low contrast detected; use even lighting or contrast normalization."
        )
    if shadow:
        recommendations.append("Uneven illumination detected; avoid shadows across the page.")
    if cropped:
        recommendations.append("Content may touch a page edge; include margins when rescanning.")
    if blockiness > 12 and (source_format or "").upper() == "JPEG":
        recommendations.append(
            "Possible JPEG artifacts; prefer lossless PNG or a higher-quality scan."
        )
    if skew is not None and abs(skew) >= 1:
        recommendations.append("Small page skew detected; enable measured deskew preprocessing.")
    degraded = sharpness < 35 or contrast < 25 or shadow or cropped or bool(skew and abs(skew) >= 1)
    return PageQuality(
        page=page,
        pixel_dimensions=[image.width, image.height],
        estimated_dpi=estimated_dpi,
        physical_size_inches=physical,
        skew_degrees=skew,
        rotation_indication=(
            f"EXIF orientation corrected by {orientation_correction_degrees}°"
            if orientation_correction_degrees
            else "no reliable 90°/180° content detector; small-angle skew measured separately"
        ),
        layout_strategy=(
            "local_preprocessing_and_geometry_layout"
            if degraded
            else "normal_detection_and_geometry_layout"
        ),
        sharpness_score=round(sharpness, 3),
        blur_warning=sharpness < 35,
        contrast_score=round(contrast, 3),
        low_contrast=contrast < 25,
        shadow_warning=shadow,
        compression_artifacts_likely=blockiness > 12 and (source_format or "").upper() == "JPEG",
        cropped_edge_warning=cropped,
        recommendations=recommendations,
    )


def _estimate_skew(image: Image.Image) -> float | None:
    """Estimate ±5° text-line skew using projection concentration on a thumbnail."""
    sample = image.copy()
    sample.thumbnail((800, 800))
    gray = np.asarray(sample.convert("L"), dtype=np.float32)
    if float(np.std(gray)) < 8:
        return None

    def projection_score(angle: float) -> float:
        rotated = sample.rotate(
            angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor="white"
        )
        ink_by_row = (255 - np.asarray(rotated.convert("L"), dtype=np.float32)).sum(axis=1)
        return float(np.var(ink_by_row) / (np.mean(ink_by_row) + 1))

    candidates = [(projection_score(step / 2), step / 2) for step in range(-10, 11)]
    _, correction = max(candidates)
    return round(-correction, 1)


def run_adaptive_batches[InputT, OutputT](
    items: Sequence[InputT],
    handler: Callable[[list[InputT]], list[OutputT]],
    *,
    initial_batch_size: int = 4,
    max_retries: int = 3,
) -> tuple[list[OutputT], BatchAudit]:
    """Retry memory-pressure failures with smaller batches, never indefinitely."""
    if initial_batch_size < 1:
        raise ValueError("initial_batch_size must be positive.")
    batch_size = min(initial_batch_size, max(1, len(items)))
    audit = BatchAudit(initial_batch_size=batch_size, final_batch_size=batch_size)
    output: list[OutputT] = []
    index = 0
    while index < len(items):
        batch = list(items[index : index + batch_size])
        audit.attempts.append({"start_index": index, "batch_size": len(batch)})
        try:
            output.extend(handler(batch))
            index += len(batch)
        except Exception as exc:
            if not _memory_pressure(exc) or batch_size == 1 or audit.retries >= max_retries:
                raise
            batch_size = max(1, batch_size // 2)
            audit.final_batch_size = batch_size
            audit.retries += 1
    return output, audit


def _blockiness(gray: np.ndarray) -> float:
    columns = np.arange(8, gray.shape[1], 8)
    rows = np.arange(8, gray.shape[0], 8)
    vertical = np.abs(gray[:, columns] - gray[:, columns - 1]).mean() if columns.size else 0
    horizontal = np.abs(gray[rows, :] - gray[rows - 1, :]).mean() if rows.size else 0
    return float((vertical + horizontal) / 2)


def _memory_pressure(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in message or "cuda oom" in message or "memoryerror" in message
