"""RapidOCR adapter with GPU preference and safe CPU fallback.

Responsible for turning ingested pages into raw, immutable OCR evidence
(`Block`s with text, score, and polygon) plus honest engine provenance. Must
not fabricate a result for a page that failed OCR (it must come back with
`status="failed"` instead), and must not report a device (CUDA vs CPU) that
wasn't actually verified active for that engine instance. Coordinates on
`Block.bbox` are normalized to [0, 1] as a fraction of page width/height, not
pixels — every downstream consumer of bbox (parse.py, artifacts.py,
layout_html.py, table_structure.py) relies on that convention. Next:
`layout.py`, which runs after OCR to add structure on top of these blocks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import pickle
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Any

import numpy as np
from PIL import Image

from agentic_extractor.cache import OCR_CACHE, page_image_hash
from agentic_extractor.config import SETTINGS
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

OCR_CACHE_VERSION = 3
_CUDA_DLL_HANDLES: list[Any] = []
_CUDA_DLL_DIRECTORY: str | None = None
_ONNX_CUDA_PREPARED = False


@dataclass(slots=True)
class OCRResource:
    engine: Any
    device: str
    warning: str | None = None
    spawn: Callable[[int], OCRResource] | None = None
    cpu_threads: int | None = None
    initialized: bool = False
    cache_namespace: str | None = None
    warm_worker_pools: dict[tuple[int, int], list[OCRResource]] = field(
        default_factory=dict, repr=False
    )
    inference_lock: Any = field(default_factory=threading.Lock, repr=False)


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
    visual_routing: dict[str, object] = field(default_factory=dict)
    layout_engine: EngineProvenance | None = None


def rapidocr_package_version() -> str | None:
    """Return the installed package version without initializing OCR models."""
    try:
        return importlib.metadata.version("rapidocr")
    except importlib.metadata.PackageNotFoundError:
        return None


def create_rapidocr_engine(
    prefer_cuda: bool = True,
    *,
    cpu_threads: int | None = None,
    spawnable: bool = True,
) -> OCRResource:
    if prefer_cuda:
        _configure_windows_cuda_dlls()
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "RapidOCR is not installed. Run 'uv sync --all-groups' and verify the local "
            "environment."
        ) from exc
    if prefer_cuda and _cuda_device_available():
        try:
            resource = OCRResource(
                RapidOCR(
                    params={
                        "EngineConfig.onnxruntime.use_cuda": True,
                        "Cls.cls_batch_num": 1,
                        "Rec.rec_batch_num": 1,
                    }
                ),
                "CUDA",
            )
            return _configure_resource(resource, prefer_cuda=True, spawnable=spawnable)
        except Exception as exc:
            warning = f"CUDA initialization failed; RapidOCR uses CPU ({type(exc).__name__})."
            resource = OCRResource(_cpu_engine(RapidOCR, cpu_threads), "CPU", warning)
            return _configure_resource(resource, prefer_cuda=False, spawnable=spawnable)
    if prefer_cuda:
        resource = OCRResource(
            _cpu_engine(RapidOCR, cpu_threads),
            "CPU",
            "No usable CUDA execution-provider device; RapidOCR uses CPU.",
            cpu_threads=cpu_threads,
        )
        return _configure_resource(resource, prefer_cuda=False, spawnable=spawnable)
    resource = OCRResource(_cpu_engine(RapidOCR, cpu_threads), "CPU", cpu_threads=cpu_threads)
    return _configure_resource(resource, prefer_cuda=False, spawnable=spawnable)


def _cpu_engine(rapidocr_type: Any, cpu_threads: int | None) -> Any:
    if cpu_threads is None:
        return rapidocr_type()
    return rapidocr_type(
        params={
            "EngineConfig.onnxruntime.intra_op_num_threads": cpu_threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )


def _configure_resource(
    resource: OCRResource, *, prefer_cuda: bool, spawnable: bool
) -> OCRResource:
    signature = {
        "package_version": rapidocr_package_version(),
        "device": resource.device,
        "engine": _engine_metadata(resource.engine),
    }
    resource.cache_namespace = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if spawnable:
        resource.spawn = lambda threads: create_rapidocr_engine(
            prefer_cuda=prefer_cuda,
            cpu_threads=threads,
            spawnable=False,
        )
    return resource


def _cuda_device_available() -> bool:
    global _ONNX_CUDA_PREPARED

    import onnxruntime as ort

    if not _ONNX_CUDA_PREPARED:
        if _CUDA_DLL_DIRECTORY and callable(getattr(ort, "preload_dlls", None)):
            ort.preload_dlls(directory=_CUDA_DLL_DIRECTORY)
        # ORT 1.29 emits a false plugin-device warning while its conventional
        # CUDA provider is active. Preserve errors while suppressing that noise.
        ort.set_default_logger_severity(3)
        _ONNX_CUDA_PREPARED = True
    # Two independent eligibility checks because ONNX Runtime's device-reporting
    # API changed across versions: newer builds expose `get_ep_devices()` (plugin
    # device enumeration), older ones only the provider-name list plus
    # `get_device()`. Either signal being true is treated as "CUDA looks usable" —
    # `ocr_page` still verifies the *active* session providers after a real
    # inference call, since either signal here can be true while registration
    # still silently fails.
    get_devices = getattr(ort, "get_ep_devices", None)
    if callable(get_devices) and any(
        device.ep_name == "CUDAExecutionProvider" for device in get_devices()
    ):
        return True
    return "CUDAExecutionProvider" in ort.get_available_providers() and ort.get_device() == "GPU"


def _configure_windows_cuda_dlls() -> str | None:
    """Register the configured CUDA Toolkit directory with Python on Windows."""
    global _CUDA_DLL_DIRECTORY

    if os.name != "nt":
        return None
    root = os.environ.get("CUDA_PATH")
    if not root:
        return None
    for relative in (("bin", "x64"), ("bin",)):
        directory = os.path.join(root, *relative)
        if not os.path.isdir(directory):
            continue
        if directory != _CUDA_DLL_DIRECTORY:
            add_directory = getattr(os, "add_dll_directory", None)
            if callable(add_directory):
                _CUDA_DLL_HANDLES.append(add_directory(directory))
            os.environ["PATH"] = os.pathsep.join([directory, os.environ.get("PATH", "")])
            _CUDA_DLL_DIRECTORY = directory
        return directory
    return None


def encode_jpeg(image: Image.Image, quality: int = 88) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def encode_png(image: Image.Image) -> bytes:
    """Preserve the exact OCR/layout raster for coordinate artifacts."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def ocr_page(resource: OCRResource, page_number: int, image: Image.Image) -> PageParse:
    started = time.perf_counter()
    output: Any = resource.engine(np.asarray(image.convert("RGB")), return_word_box=True)
    resource.initialized = True
    # Provenance is only trusted after a real inference call: CUDA provider
    # registration can succeed at engine-construction time yet silently fall
    # back to CPU inside ONNX Runtime. Inspecting the detector session's actual
    # active providers here is the only way to catch that and report it honestly.
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
        layout_image_bytes=encode_png(image),
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
        # Normalize to [0, 1] fractions of page width/height, not pixels; `polygon`
        # itself is kept in raw pixel coordinates as returned by RapidOCR.
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
    source_pages = [page for page in document.pages if page.number in selected_pages]
    document_sha256 = (
        document.document_sha256 or hashlib.sha256(document.original_bytes).hexdigest()
    )
    cache_keys: dict[int, str] = {}
    completed: dict[int, PageParse] = {}
    misses = []
    if resource.cache_namespace:
        for source_page in source_pages:
            key = _ocr_cache_key(source_page, resource.cache_namespace)
            cache_keys[source_page.number] = key
            cached = OCR_CACHE.get(key)
            if cached is None:
                misses.append(source_page)
                continue
            page = pickle.loads(cached)
            if not isinstance(page, PageParse):
                misses.append(source_page)
                continue
            _rebind_cached_page(page, source_page.number)
            page.ocr_cache_hit = True
            completed[page.page] = page
    else:
        misses = list(source_pages)
    if misses:
        # RapidOCR model instances are expensive and are not safe to share across
        # concurrent inference. Keep the bounded pool warm on the app resource and
        # lease it to one document at a time.
        with resource.inference_lock:
            worker_resources, worker_threads, requested_workers, pool_warnings = _worker_resources(
                resource, len(misses)
            )
            parsed_misses = _parse_pages(misses, worker_resources)
    else:
        worker_resources, worker_threads, requested_workers, pool_warnings = [], None, 0, []
        parsed_misses = []
    for page in parsed_misses:
        completed[page.page] = page
        key = cache_keys.get(page.page)
        if key and page.status == "completed":
            OCR_CACHE.put(key, pickle.dumps(page, protocol=pickle.HIGHEST_PROTOCOL))
    pages = [completed[source_page.number] for source_page in source_pages]
    failed = [page.page for page in pages if page.status == "failed"]
    warnings = [resource.warning] if resource.warning else []
    warnings.extend(pool_warnings)
    for page in pages:
        warnings.extend(
            page.warnings
            if page.status == "failed"
            else (f"Page {page.page}: {warning}" for warning in page.warnings)
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
            "source_sha256": document_sha256,
        },
        selected_pages=sorted(selected_pages),
        pages=pages,
        markdown=document_markdown(pages),
        engine=engine,
        timings={
            "total_seconds": total,
            "ocr_seconds": sum(page.ocr_seconds for page in parsed_misses),
            "ocr_concurrency_ratio": round(
                sum(page.ocr_seconds for page in parsed_misses) / total if total else 0.0, 3
            ),
        },
        page_statuses={
            number: "failed" if number in failed else "completed"
            for number in sorted(selected_pages)
        },
        warnings=warnings,
        failed_pages=failed,
        adaptive_processing={
            "strategy": "bounded dedicated-engine page workers",
            "requested_workers": requested_workers,
            "actual_workers": len(worker_resources),
            "max_workers": SETTINGS.ocr_max_workers,
            "cpu_threads_per_worker": worker_threads,
            "cuda_serialized": resource.device == "CUDA",
            "worker_pool_warnings": pool_warnings,
            "ocr_cache_hits": len(pages) - len(parsed_misses),
            "ocr_cache_misses": len(parsed_misses),
            "ocr_cached_seconds_avoided": round(
                sum(page.ocr_seconds for page in pages if getattr(page, "ocr_cache_hit", False)),
                3,
            ),
        },
    )


def _ocr_cache_key(source_page: Any, namespace: str) -> str:
    # Preprocessing may replace the rendered image after ingestion; hash the
    # exact pixels entering OCR rather than trusting stored render metadata.
    digest = page_image_hash(source_page.image)
    return f"ocr:{OCR_CACHE_VERSION}:{digest}:{namespace}"


def _rebind_cached_page(page: PageParse, page_number: int) -> None:
    old_page = page.page
    old_prefix = f"p{old_page}-"
    new_prefix = f"p{page_number}-"

    def rebind(source_id: str) -> str:
        return (
            new_prefix + source_id[len(old_prefix) :]
            if source_id.startswith(old_prefix)
            else source_id
        )

    page.page = page_number
    for block in page.blocks:
        block.id = rebind(block.id)
        block.page = page_number
    for chunk in page.chunks:
        chunk.id = rebind(chunk.id)
        chunk.page = page_number
        chunk.source_block_ids = [rebind(source_id) for source_id in chunk.source_block_ids]


def _requested_worker_count(page_count: int, resource: OCRResource) -> int:
    if resource.device == "CUDA" or resource.spawn is None:
        return 1
    cpu_capacity = max(1, (os.cpu_count() or 1) // 4)
    return min(page_count, SETTINGS.ocr_max_workers, cpu_capacity)


def _worker_resources(
    primary: OCRResource, page_count: int
) -> tuple[list[OCRResource], int | None, int, list[str]]:
    requested = _requested_worker_count(page_count, primary)
    if requested == 1:
        return [primary], primary.cpu_threads, requested, []
    logical_cpus = os.cpu_count() or 1
    cpu_threads = max(1, logical_cpus // requested)
    pool_key = (requested, cpu_threads)
    warm = primary.warm_worker_pools.get(pool_key)
    if warm is not None:
        return warm, cpu_threads, requested, []
    resources: list[OCRResource] = []
    if primary.cpu_threads == cpu_threads:
        resources.append(primary)
    warnings: list[str] = []
    while len(resources) < requested:
        try:
            assert primary.spawn is not None
            resources.append(primary.spawn(cpu_threads))
        except Exception as exc:
            warnings.append(
                f"RapidOCR worker pool reduced to {max(1, len(resources))}: "
                f"{type(exc).__name__}: {exc}"
            )
            break
    if not resources:
        resources.append(primary)
        cpu_threads = primary.cpu_threads
    primary.warm_worker_pools[pool_key] = resources
    return resources, cpu_threads, requested, warnings


def _parse_pages(source_pages: list[Any], resources: list[OCRResource]) -> list[PageParse]:
    if len(resources) == 1:
        return [_parse_source_page(resources[0], source_page) for source_page in source_pages]
    available: SimpleQueue[OCRResource] = SimpleQueue()
    for resource in resources:
        available.put(resource)

    def process(source_page: Any) -> PageParse:
        active = available.get()
        try:
            return _parse_source_page(active, source_page)
        finally:
            available.put(active)

    completed: dict[int, PageParse] = {}
    with ThreadPoolExecutor(max_workers=len(resources), thread_name_prefix="rapidocr-page") as pool:
        futures = {pool.submit(process, page): page.number for page in source_pages}
        for future in as_completed(futures):
            page = future.result()
            completed[page.page] = page
    return [completed[source_page.number] for source_page in source_pages]


def _parse_source_page(resource: OCRResource, source_page: Any) -> PageParse:
    try:
        return ocr_page(resource, source_page.number, source_page.image)
    except Exception as exc:
        message = f"Page {source_page.number} failed: {type(exc).__name__}: {exc}"
        return PageParse(
            page=source_page.number,
            width=source_page.image.width,
            height=source_page.image.height,
            image_bytes=encode_jpeg(source_page.image),
            layout_image_bytes=encode_png(source_page.image),
            warnings=[message],
            status="failed",
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
