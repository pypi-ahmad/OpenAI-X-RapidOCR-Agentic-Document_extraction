"""RapidOCR adapter with GPU preference and safe CPU fallback."""

from __future__ import annotations

import importlib.metadata
import io
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from agentic_extractor.ingest import IngestedDocument
from agentic_extractor.models import (
    Block,
    CheckboxCorrection,
    CheckboxRecord,
    ProcessingMode,
    RefinementRecord,
    UsageRecord,
)
from agentic_extractor.parse import (
    PageParse,
    build_layout_chunks,
    document_markdown,
    reconstruct_layout,
)


@dataclass(slots=True)
class OCRResource:
    engine: Any
    device: str
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class EngineProvenance:
    name: str
    version: str
    device: str
    confidence_calibrated: bool = False
    model_metadata: dict[str, object] | None = None


@dataclass(slots=True)
class LocalParseResult:
    document_metadata: dict[str, object]
    selected_pages: list[int]
    pages: list[PageParse]
    markdown: str
    engine: EngineProvenance
    timings: dict[str, float]
    page_statuses: dict[int, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failed_pages: list[int] = field(default_factory=list)
    requested_mode: ProcessingMode = ProcessingMode.BALANCED
    effective_mode: ProcessingMode = ProcessingMode.BALANCED
    cloud_pages: list[int] = field(default_factory=list)
    cloud_image_pages: list[int] = field(default_factory=list)
    routing: dict[int, list[str]] = field(default_factory=dict)
    usage: UsageRecord = field(default_factory=UsageRecord)
    cloud_output: dict[str, object] | None = None
    cloud_attempts: list[dict[str, object]] = field(default_factory=list)
    ocr_attempts: list[dict[str, object]] = field(default_factory=list)
    refinements: list[RefinementRecord] = field(default_factory=list)
    workflow_manifest: dict[str, object] | None = None
    quality_diagnostics: list[dict[str, object]] = field(default_factory=list)
    adaptive_processing: dict[str, object] = field(default_factory=dict)
    checkboxes: list[CheckboxRecord] = field(default_factory=list)
    checkbox_corrections: list[CheckboxCorrection] = field(default_factory=list)
    checkbox_verifications: list[dict[str, object]] = field(default_factory=list)


def rapidocr_package_version() -> str | None:
    """Return the installed package version without initializing OCR models."""
    try:
        return importlib.metadata.version("rapidocr")
    except importlib.metadata.PackageNotFoundError:
        return None


def create_rapidocr_engine(prefer_cuda: bool = True) -> OCRResource:
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "RapidOCR is not installed. Run 'uv sync --all-groups' and verify the local "
            "environment."
        ) from exc
    if prefer_cuda and _cuda_device_available():
        try:
            return OCRResource(RapidOCR(params={"EngineConfig.onnxruntime.use_cuda": True}), "CUDA")
        except Exception as exc:
            warning = f"CUDA initialization failed; RapidOCR uses CPU ({type(exc).__name__})."
            return OCRResource(RapidOCR(), "CPU", warning)
    if prefer_cuda:
        return OCRResource(
            RapidOCR(), "CPU", "No usable CUDA execution-provider device; RapidOCR uses CPU."
        )
    return OCRResource(RapidOCR(), "CPU")


def _cuda_device_available() -> bool:
    import onnxruntime as ort

    get_devices = getattr(ort, "get_ep_devices", None)
    if callable(get_devices) and any(
        device.ep_name == "CUDAExecutionProvider" for device in get_devices()
    ):
        return True
    return "CUDAExecutionProvider" in ort.get_available_providers() and ort.get_device() == "GPU"


def encode_jpeg(image: Image.Image, quality: int = 88) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def ocr_page(resource: OCRResource, page_number: int, image: Image.Image) -> PageParse:
    started = time.perf_counter()
    output: Any = resource.engine(np.asarray(image.convert("RGB")), return_word_box=True)
    if resource.device == "CUDA":
        providers = resource.engine.text_det.session.session.get_providers()
        if "CUDAExecutionProvider" not in providers:
            resource.device = "CPU"
            resource.warning = "CUDA was requested but unavailable; RapidOCR uses CPU."
    boxes = list(output.boxes) if output.boxes is not None else []
    texts = list(output.txts) if output.txts is not None else []
    scores = list(output.scores) if output.scores is not None else []
    raw: list[tuple[list[list[float]] | None, str, float | None]] = []
    for index, text in enumerate(texts):
        box = np.asarray(boxes[index]).astype(float).tolist() if index < len(boxes) else None
        score = float(scores[index]) if index < len(scores) else None
        raw.append((box, str(text), score))
    blocks, layout_signals = reconstruct_layout(
        [
            _to_block(page_number, i, item, image.width, image.height)
            for i, item in enumerate(raw, 1)
        ]
    )
    chunks = build_layout_chunks(blocks)
    elapsed_list = [_json_value(item) for item in getattr(output, "elapse_list", [])]
    stages = ["detection", "classification", "recognition"]
    word_results = _json_value(getattr(output, "word_results", []))
    warnings: list[str] = []
    if layout_signals["ambiguous"]:
        warnings.append("Reading order is ambiguous; deterministic best-effort order was used.")
    return PageParse(
        page=page_number,
        width=image.width,
        height=image.height,
        blocks=blocks,
        chunks=chunks,
        image_bytes=encode_jpeg(image),
        ocr_seconds=time.perf_counter() - started,
        engine_elapsed_seconds=getattr(output, "elapse", None),
        stage_timings={
            name: elapsed_list[index] if index < len(elapsed_list) else None
            for index, name in enumerate(stages)
        },
        raw_evidence={
            "output_type": f"{type(output).__module__}.{type(output).__name__}",
            "boxes": _json_value(boxes),
            "texts": texts,
            "scores": [_json_value(score) for score in scores],
            "word_results": word_results,
            "elapse_list": elapsed_list,
            "word_boxes_requested": True,
            "score_semantics": "recognition_confidence",
            "orientation_available": False,
            "orientation_scope": "text_line_0_180",
        },
        layout_signals=layout_signals,
        warnings=warnings,
    )


def _to_block(
    page: int,
    index: int,
    item: tuple[list[list[float]] | None, str, float | None],
    width: int,
    height: int,
) -> Block:
    polygon, text, score = item
    bbox = None
    if polygon:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        bbox = [min(xs) / width, min(ys) / height, max(xs) / width, max(ys) / height]
    return Block(
        id=f"p{page}-b{index}", page=page, text=text, ocr_score=score, polygon=polygon, bbox=bbox
    )


def parse_document_local(
    document: IngestedDocument, selected_pages: set[int], resource: OCRResource
) -> LocalParseResult:
    """Parse only selected source pages, preserving their original order."""
    available = {page.number for page in document.pages}
    if not selected_pages or not selected_pages <= available:
        raise ValueError("Selected pages must be a non-empty subset of the document.")
    started = time.perf_counter()
    pages: list[PageParse] = []
    failed: list[int] = []
    warnings = [resource.warning] if resource.warning else []
    for source_page in document.pages:
        if source_page.number not in selected_pages:
            continue
        try:
            page = ocr_page(resource, source_page.number, source_page.image)
            pages.append(page)
            warnings.extend(f"Page {page.page}: {warning}" for warning in page.warnings)
        except Exception as exc:
            failed.append(source_page.number)
            message = f"Page {source_page.number} failed: {type(exc).__name__}: {exc}"
            warnings.append(message)
            pages.append(
                PageParse(
                    page=source_page.number,
                    width=source_page.image.width,
                    height=source_page.image.height,
                    image_bytes=encode_jpeg(source_page.image),
                    warnings=[message],
                    status="failed",
                )
            )
    total = time.perf_counter() - started
    engine = EngineProvenance(
        name="RapidOCR",
        version=importlib.metadata.version("rapidocr"),
        device=resource.device,
        model_metadata=_engine_metadata(resource.engine),
    )
    return LocalParseResult(
        document_metadata={
            "file_name": document.file_name,
            "mime_type": document.mime_type,
            "byte_size": document.byte_size,
            "source_page_count": document.page_count,
        },
        selected_pages=sorted(selected_pages),
        pages=pages,
        markdown=document_markdown(pages),
        engine=engine,
        timings={
            "total_seconds": total,
            "ocr_seconds": sum(page.ocr_seconds for page in pages),
        },
        page_statuses={
            number: "failed" if number in failed else "completed"
            for number in sorted(selected_pages)
        },
        warnings=warnings,
        failed_pages=failed,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _engine_metadata(engine: Any) -> dict[str, object]:
    cfg = getattr(engine, "cfg", None)
    detector = getattr(cfg, "Det", None)
    recognizer = getattr(cfg, "Rec", None)
    classifier = getattr(cfg, "Cls", None)
    global_config = getattr(cfg, "Global", None)
    return {
        "detector": {
            "engine_type": str(getattr(detector, "engine_type", "unknown")),
            "ocr_version": str(getattr(detector, "ocr_version", "unknown")),
            "model_type": str(getattr(detector, "model_type", "unknown")),
            "language": str(getattr(detector, "lang_type", "unknown")),
        },
        "recognizer": {
            "engine_type": str(getattr(recognizer, "engine_type", "unknown")),
            "ocr_version": str(getattr(recognizer, "ocr_version", "unknown")),
            "model_type": str(getattr(recognizer, "model_type", "unknown")),
            "language": str(getattr(recognizer, "lang_type", "unknown")),
        },
        "classifier": {
            "enabled": bool(getattr(global_config, "use_cls", False)),
            "engine_type": str(getattr(classifier, "engine_type", "unknown")),
            "ocr_version": str(getattr(classifier, "ocr_version", "unknown")),
            "model_type": str(getattr(classifier, "model_type", "unknown")),
            "language": str(getattr(classifier, "lang_type", "unknown")),
        },
        "classifier_enabled": bool(getattr(global_config, "use_cls", False)),
        "execution_providers": _execution_providers(engine),
    }


def _execution_providers(engine: Any) -> dict[str, list[str]]:
    providers: dict[str, list[str]] = {}
    for name, attribute in (
        ("detector", "text_det"),
        ("classifier", "text_cls"),
        ("recognizer", "text_rec"),
    ):
        component = getattr(engine, attribute, None)
        session = getattr(getattr(component, "session", None), "session", None)
        getter = getattr(session, "get_providers", None)
        if callable(getter):
            providers[name] = [str(provider) for provider in getter()]
    return providers
