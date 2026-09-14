"""PP-DocLayoutV3 worker adapter and additive layout grounding.

Responsible for: starting/talking to the isolated `tools/pp_doclayout` worker
process, validating its output against the locked V3 contract, and merging
layout regions/reading order/table routing onto `PageParse` without touching
RapidOCR's own blocks.

Must not: treat worker output as trusted structure. Every value pulled from
the worker (labels, coordinates, scores, order) is revalidated here and
raises `LayoutContractError` rather than being coerced, because the worker
runs a different PaddleX/OpenCV stack than this process and its output is
effectively an untrusted external boundary, not a same-process call.

Next: `table_structure.py` for how detected table regions become grounded
cell evidence, and `tools/pp_doclayout/worker.py` for the other end of the
subprocess protocol used below.
"""

from __future__ import annotations

import atexit
import base64
import hashlib
import io
import json
import math
import os
import queue
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from agentic_extractor.cache import LAYOUT_CACHE, TABLE_CACHE, page_image_hash
from agentic_extractor.models import LayoutBlockLink, LayoutRegion, ReadingOrderEvidence
from agentic_extractor.ocr import EngineProvenance, encode_jpeg
from agentic_extractor.parse import PageParse, build_layout_chunks, document_markdown
from agentic_extractor.table_structure import enrich_page_tables, normalize_table_result

PP_DOC_LAYOUT_MODEL = "PP-DocLayoutV3"
PP_DOC_LAYOUT_THRESHOLD = 0.5
# Preserve detector recall at the table-model seam. The specialized table
# classifier, structure model, grounding checks, and Luna review remain the
# acceptance gates, so this threshold should not pre-emptively discard regions.
TABLE_STRUCTURE_CANDIDATE_THRESHOLD = PP_DOC_LAYOUT_THRESHOLD
PP_DOC_LAYOUT_LABELS = (
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
)
LAYOUT_CACHE_VERSION = 2
TABLE_CACHE_VERSION = 4


# Two distinct failure modes on purpose: setup errors mean the worker can't
# run at all (missing env, dead process, timeout) and should tell the user
# how to fix their install; contract errors mean the worker ran but returned
# a shape this code doesn't trust, which is treated as a layout failure
# rather than silently patched or downgraded.
class PPDocLayoutSetupError(RuntimeError):
    """The required local PP-DocLayoutV3 runtime is unavailable."""


class LayoutContractError(RuntimeError):
    """The installed worker returned an incompatible V3 result contract."""


LayoutSetupError = PPDocLayoutSetupError


@dataclass(slots=True)
class PPDocLayoutResource:
    client: Any
    device: str
    version: str
    paddle_version: str
    model: str = PP_DOC_LAYOUT_MODEL
    warning: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    cache_namespace: str = ""

    def predict(self, pages: list[Any]) -> dict[int, dict[str, Any]]:
        return self.client.predict(pages)

    def predict_tables(self, tables: list[Any]) -> dict[str, dict[str, Any]]:
        return self.client.predict_tables(tables)

    @property
    def provenance(self) -> EngineProvenance:
        return EngineProvenance(
            name=self.model,
            version=self.version,
            device=self.device,
            model_metadata={
                "model_name": self.model,
                "paddle_version": self.paddle_version,
                "threshold": PP_DOC_LAYOUT_THRESHOLD,
                "label_count": len(PP_DOC_LAYOUT_LABELS),
            },
        )


def _runtime_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    project = root / "tools" / "pp_doclayout"
    python = project / ".venv" / "Scripts" / "python.exe"
    return project, python


def pp_doclayout_runtime_installed() -> bool:
    project, python = _runtime_paths()
    return project.joinpath("uv.lock").is_file() and python.is_file()


def create_pp_doclayout_engine() -> PPDocLayoutResource:
    project, python = _runtime_paths()
    worker = project / "worker.py"
    if not python.is_file() or not worker.is_file():
        raise LayoutSetupError(
            "PP-DocLayoutV3 is unavailable. Run "
            "'uv sync --project tools/pp_doclayout --locked' and retry."
        )
    client: _WorkerClient | None = None
    health: dict[str, Any] = {}
    for attempt in range(2):
        client = _WorkerClient(python, worker, project)
        try:
            health = client.request({"op": "health"}, timeout=300)
        except Exception:
            client.close()
            raise
        # gpu_fallback means the worker saw CUDA but still had to init on CPU
        # (see worker.py's own GPU-try/CPU-except path). Retry once with a
        # fresh worker process in case that was a transient init failure;
        # a second failure blocks extraction instead of silently accepting a
        # CPU run for what looked like a GPU-capable machine, and the CPU
        # fallback is deliberately not cached across attempts.
        gpu_fallback = (
            bool(health.get("gpu_available")) and str(health.get("device", "CPU")).upper() != "GPU"
        )
        if not gpu_fallback:
            break
        client.close()
        if attempt == 1:
            raise LayoutSetupError(
                "PP-DocLayoutV3 detected a CUDA device but GPU model initialization failed twice. "
                "Run 'uv sync --project tools/pp_doclayout --locked', restart the app, and verify "
                "the NVIDIA driver before retrying. CPU fallback was not cached."
            )
    assert client is not None
    if health.get("model") != PP_DOC_LAYOUT_MODEL:
        client.close()
        raise LayoutContractError("PP-DocLayoutV3 worker returned an unexpected model.")
    device = str(health.get("device", "CPU")).upper()
    client.batch_size = 2 if device == "GPU" else 1
    warning = health.get("warning")
    resource = PPDocLayoutResource(
        client=client,
        device=device,
        version=str(health.get("paddlex_version", "unknown")),
        paddle_version=str(health.get("paddle_version", "unknown")),
        warning=str(warning) if warning else None,
    )
    resource.cache_namespace = hashlib.sha256(
        json.dumps(
            {
                "model": resource.model,
                "paddlex": resource.version,
                "paddle": resource.paddle_version,
                "device": resource.device,
                "threshold": PP_DOC_LAYOUT_THRESHOLD,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return resource


# Wire protocol with tools/pp_doclayout/worker.py: newline-delimited JSON
# over the child's stdin/stdout, one request/response object per line, each
# tagged with a request_id. Any stdout line that fails to parse as a JSON
# object is treated as a stray diagnostic (e.g. a library that wrote to real
# stdout instead of the redirected stderr) and buffered as a log line rather
# than surfaced as a protocol error.
class _WorkerClient:
    def __init__(self, python: Path, worker: Path, cwd: Path) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(
            [str(python), str(worker)],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=flags,
        )
        self._responses: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self._logs: deque[str] = deque(maxlen=20)
        self._lock = threading.Lock()
        self.batch_size = 1
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        atexit.register(self.close)

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                self._logs.append(line.rstrip())
                continue
            if isinstance(value, dict):
                self._responses.put(value)
        self._responses.put(LayoutSetupError("PP-DocLayoutV3 worker exited unexpectedly."))

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._logs.append(line.rstrip())

    def request(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        # The lock serializes callers so exactly one request is ever in
        # flight; the request_id match below is a safety net against a
        # response left over from a previous call, not real multiplexing.
        with self._lock:
            if self._process.poll() is not None:
                raise LayoutSetupError(self._failure_message("worker is not running"))
            request_id = uuid.uuid4().hex
            message = {**payload, "request_id": request_id}
            assert self._process.stdin is not None
            self._process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
            while True:
                try:
                    response = self._responses.get(timeout=timeout)
                except queue.Empty as exc:
                    raise LayoutSetupError(self._failure_message("worker timed out")) from exc
                if isinstance(response, BaseException):
                    raise response
                if response.get("request_id") != request_id:
                    continue
                if not response.get("ok"):
                    raise LayoutSetupError(
                        self._failure_message(str(response.get("error", "prediction failed")))
                    )
                return response

    def predict(self, pages: list[Any]) -> dict[int, dict[str, Any]]:
        output: dict[int, dict[str, Any]] = {}
        for start in range(0, len(pages), self.batch_size):
            batch = pages[start : start + self.batch_size]
            payload_pages = [
                {
                    "page": page.number,
                    "image": base64.b64encode(encode_jpeg(page.image, quality=92)).decode("ascii"),
                }
                for page in batch
            ]
            response = self.request({"op": "predict", "pages": payload_pages}, timeout=120)
            values = response.get("pages")
            if not isinstance(values, list):
                raise LayoutContractError("PP-DocLayoutV3 worker returned no page results.")
            output.update({int(item["page"]): item["result"] for item in values})
        return output

    def predict_tables(self, tables: list[Any]) -> dict[str, dict[str, Any]]:
        payload = [
            {
                "id": table.id,
                "page": table.page,
                "image": base64.b64encode(encode_jpeg(table.image, quality=95)).decode("ascii"),
            }
            for table in tables
        ]
        response = self.request({"op": "predict_tables", "tables": payload}, timeout=300)
        values = response.get("tables")
        if not isinstance(values, list):
            raise LayoutContractError("Table worker returned no table results.")
        return {str(item["id"]): item["result"] for item in values}

    def _failure_message(self, reason: str) -> str:
        detail = next((line for line in reversed(self._logs) if line), "no worker log available")
        return f"PP-DocLayoutV3 {reason}. {detail}"

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()


def apply_document_layout(
    local: Any, source_pages: list[Any] | None, resource: PPDocLayoutResource
) -> None:
    """Run V3 on every selected page and attach immutable layout evidence."""
    started = time.perf_counter()
    if source_pages is None:
        source_pages = [
            SimpleNamespace(
                number=page.page,
                image=Image.open(io.BytesIO(page.image_bytes)).convert("RGB"),
            )
            for page in local.pages
        ]
    by_number = {page.number: page for page in source_pages}
    selected = [by_number[number] for number in local.selected_pages]
    raw_by_page: dict[int, dict[str, Any]] = {}
    misses: list[Any] = []
    cache_keys: dict[int, str] = {}
    for source in selected:
        key = _layout_cache_key(source, resource.cache_namespace)
        cache_keys[source.number] = key
        cached = LAYOUT_CACHE.get(key)
        if cached is None:
            misses.append(source)
        else:
            raw_by_page[source.number] = json.loads(cached)
    if misses:
        predicted = resource.predict(misses)
        expected = {page.number for page in misses}
        if set(predicted) != expected:
            raise LayoutContractError("PP-DocLayoutV3 did not return every selected page.")
        for page_number, value in predicted.items():
            raw_by_page[page_number] = value
            LAYOUT_CACHE.put(cache_keys[page_number], json.dumps(value).encode())
    pages = {page.page: page for page in local.pages}
    for source in selected:
        page = pages[source.number]
        regions = normalize_layout_result(
            raw_by_page[source.number], page.page, page.width, page.height
        )
        enrich_page_layout(page, regions)
        page.layout_raw_evidence = raw_by_page[source.number]
    local.timings["layout_detection_seconds"] = time.perf_counter() - started
    _apply_table_structures(local, selected, resource)
    # Layout and table enrichment replace the derived page chunks. Keep the
    # document-level Parse snapshot in sync so downstream workflows and
    # artifacts receive the grounded table HTML instead of the pre-layout OCR.
    local.markdown = document_markdown(local.pages)
    local.layout_engine = resource.provenance
    if resource.warning:
        local.warnings.append(resource.warning)
    local.adaptive_processing["pp_doclayout_v3"] = {
        "model": resource.model,
        "device": resource.device,
        "threshold": PP_DOC_LAYOUT_THRESHOLD,
        "cache_hits": len(selected) - len(misses),
        "cache_misses": len(misses),
    }
    local.timings["layout_seconds"] = time.perf_counter() - started


def _apply_table_structures(local: Any, selected: list[Any], resource: PPDocLayoutResource) -> None:
    started = time.perf_counter()
    source_by_page = {page.number: page for page in selected}
    candidates: list[Any] = []
    cache_keys: dict[str, str] = {}
    raw_by_id: dict[str, dict[str, Any]] = {}
    candidate_region_ids: set[str] = set()
    skipped_region_ids: set[str] = set()
    for page in local.pages:
        source = source_by_page[page.page]
        for region in page.layout_regions:
            if region.label != "table":
                continue
            if region.score < TABLE_STRUCTURE_CANDIDATE_THRESHOLD:
                skipped_region_ids.add(region.id)
                continue
            candidate_region_ids.add(region.id)
            left, top, right, bottom = (int(round(value)) for value in region.coordinate)
            crop = source.image.crop((left, top, right, bottom)).convert("RGB")
            candidate = SimpleNamespace(id=region.id, page=page.page, image=crop)
            key = _table_cache_key(candidate, resource.cache_namespace)
            cache_keys[region.id] = key
            cached = TABLE_CACHE.get(key)
            if cached is None:
                candidates.append(candidate)
            else:
                raw_by_id[region.id] = json.loads(cached)
    failures: dict[str, str] = {}
    if candidates:
        try:
            predicted = resource.predict_tables(candidates)
            for table_id, value in predicted.items():
                raw_by_id[table_id] = value
                TABLE_CACHE.put(cache_keys[table_id], json.dumps(value).encode())
            for candidate in candidates:
                if candidate.id not in predicted:
                    failures[candidate.id] = "Table worker returned no result for this region."
        except Exception as exc:
            # Deliberately broad: a table-model failure (worker crash, bad
            # crop, OOM) must not fail the whole page. Every candidate table
            # region degrades to an "invalid, review_required" evidence row
            # below instead of raising past this point.
            for candidate in candidates:
                failures[candidate.id] = (
                    f"Table structure detection failed: {type(exc).__name__}: {exc}"
                )
    for page in local.pages:
        tables = []
        for region in (item for item in page.layout_regions if item.id in candidate_region_ids):
            value = raw_by_id.get(region.id)
            if value is None:
                # Worker never returned this region (see the failures dict
                # above): fail closed with an explicit invalid/review-required
                # row rather than dropping the table region silently, so the
                # gap is visible in the manifest and to Luna's table review.
                from agentic_extractor.models import TableStructureEvidence

                tables.append(
                    TableStructureEvidence(
                        id=f"{region.id}-table",
                        page=page.page,
                        layout_region_id=region.id,
                        bbox=region.bbox,
                        style="wireless",
                        classifier_score=0,
                        structure_score=0,
                        structure_model="SLANet_plus",
                        status="invalid",
                        review_required=True,
                        warnings=[
                            failures.get(region.id, "Table structure evidence is unavailable.")
                        ],
                    )
                )
                continue
            page.table_raw_evidence.append(value)
            try:
                tables.append(normalize_table_result(region.id, value, page))
            except (TypeError, ValueError) as exc:
                from agentic_extractor.models import TableStructureEvidence

                tables.append(
                    TableStructureEvidence(
                        id=f"{region.id}-table",
                        page=page.page,
                        layout_region_id=region.id,
                        bbox=region.bbox,
                        style="wireless",
                        classifier_score=0,
                        structure_score=0,
                        structure_model="SLANet_plus",
                        status="invalid",
                        review_required=True,
                        warnings=[f"Invalid table structure evidence: {exc}"],
                    )
                )
        enrich_page_tables(page, tables)
    local.adaptive_processing["table_structure"] = {
        "regions": sum(len(page.table_structures) for page in local.pages),
        "candidate_threshold": TABLE_STRUCTURE_CANDIDATE_THRESHOLD,
        "candidate_threshold_calibrated": False,
        "candidate_regions": len(candidate_region_ids),
        "skipped_regions": len(skipped_region_ids),
        "skipped_pages": sorted(
            page.page
            for page in local.pages
            if not any(region.id in candidate_region_ids for region in page.layout_regions)
        ),
        "cache_hits": len(cache_keys) - len(candidates),
        "cache_misses": len(candidates),
        "models": ["PP-LCNet_x1_0_table_cls", "SLANeXt_wired", "SLANet_plus"],
    }
    local.timings["table_structure_seconds"] = time.perf_counter() - started


def _table_cache_key(table: Any, namespace: str) -> str:
    image = table.image.convert("RGB")
    return f"table:{TABLE_CACHE_VERSION}:{page_image_hash(image)}:{namespace}"


def normalize_layout_result(
    value: dict[str, Any], page: int, width: int, height: int
) -> list[LayoutRegion]:
    """Validate raw worker boxes and convert pixel coordinates to page-normalized bbox (0-1).

    `coordinate` stays in source-pixel units (used later to crop table
    regions); `bbox` is `coordinate` divided by page width/height, which is
    the space `enrich_page_layout` and downstream OCR-block matching use.
    """
    if set(value) == {"res"} and isinstance(value["res"], dict):
        value = value["res"]
    boxes = value.get("boxes")
    if not isinstance(boxes, list):
        raise LayoutContractError("PP-DocLayoutV3 result has no boxes list.")
    raw_orders = sorted({int(item["order"]) for item in boxes if item.get("order") is not None})
    order_map = {raw: index for index, raw in enumerate(raw_orders, 1)}
    regions: list[LayoutRegion] = []
    for index, item in enumerate(boxes, 1):
        cls_id = int(item.get("cls_id", -1))
        label = str(item.get("label", ""))
        if not 0 <= cls_id < len(PP_DOC_LAYOUT_LABELS) or PP_DOC_LAYOUT_LABELS[cls_id] != label:
            raise LayoutContractError("PP-DocLayoutV3 taxonomy does not match the locked contract.")
        coordinate = _finite_numbers(item.get("coordinate"), 4, "coordinate")
        left, top, right, bottom = coordinate
        if left >= right or top >= bottom:
            raise LayoutContractError("PP-DocLayoutV3 coordinate is invalid.")
        polygon_value = item.get("polygon_points") or []
        polygon = [_finite_numbers(point, 2, "polygon") for point in polygon_value]
        score = float(item.get("score", -1))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise LayoutContractError("PP-DocLayoutV3 score is invalid.")
        raw_order = int(item["order"]) if item.get("order") is not None else None
        regions.append(
            LayoutRegion(
                id=f"p{page}-l{index}",
                page=page,
                cls_id=cls_id,
                label=label,
                score=score,
                coordinate=coordinate,
                bbox=[left / width, top / height, right / width, bottom / height],
                polygon=polygon,
                raw_order=raw_order,
                normalized_order=order_map.get(raw_order),
            )
        )
    return regions


def enrich_page_layout(page: PageParse, regions: list[LayoutRegion]) -> None:
    links: list[LayoutBlockLink] = []
    region_by_id = {region.id: region for region in regions}
    block_region: dict[str, LayoutRegion] = {}
    for block in page.blocks:
        if not block.bbox:
            continue
        candidates: list[tuple[float, LayoutRegion]] = []
        for region in regions:
            coverage = _coverage(block.bbox, region.bbox)
            center = ((block.bbox[0] + block.bbox[2]) / 2, (block.bbox[1] + block.bbox[3]) / 2)
            inside = (
                region.bbox[0] <= center[0] <= region.bbox[2]
                and region.bbox[1] <= center[1] <= region.bbox[3]
            )
            if inside or coverage >= 0.5:
                candidates.append((coverage, region))
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: (item[0], item[1].score, -_area(item[1].bbox)), reverse=True
        )
        primary_coverage, primary = candidates[0]
        block_region[block.id] = primary
        links.append(
            LayoutBlockLink(
                page=page.page,
                block_id=block.id,
                primary_region_id=primary.id,
                region_ids=[region.id for _, region in candidates],
                block_coverage=primary_coverage,
            )
        )
    slots = [
        index
        for index, block in enumerate(page.blocks)
        if block.id in block_region and block_region[block.id].normalized_order is not None
    ]
    ordered = sorted(
        (page.blocks[index] for index in slots),
        key=lambda block: (
            region_by_id[block_region[block.id].id].normalized_order or 0,
            block.bbox[1] if block.bbox else 1,
            block.bbox[0] if block.bbox else 1,
        ),
    )
    for index, block in zip(slots, ordered, strict=True):
        page.blocks[index] = block
    page.layout_regions = regions
    page.layout_block_links = links
    ordered_regions = [
        region.id
        for region in sorted(
            (item for item in regions if item.normalized_order is not None),
            key=lambda item: item.normalized_order or 0,
        )
    ]
    unordered_regions = [region.id for region in regions if region.normalized_order is None]
    unmatched_blocks = [block.id for block in page.blocks if block.id not in block_region]
    conflicts: list[str] = []
    if unordered_regions:
        conflicts.append("PP-DocLayoutV3 omitted order for one or more regions.")
    if unmatched_blocks:
        conflicts.append("One or more RapidOCR blocks are outside ordered layout regions.")
    page.reading_order_evidence = ReadingOrderEvidence(
        page=page.page,
        ordered_region_ids=ordered_regions,
        unordered_region_ids=unordered_regions,
        ordered_block_ids=[block.id for block in page.blocks if block.id in block_region],
        unmatched_block_ids=unmatched_blocks,
        conflicts=conflicts,
        status="ambiguous" if conflicts or not regions else "valid",
    )
    page.layout_signals.update(
        {
            "source": "pp-doclayout-v3",
            "native_layout": True,
            "model": PP_DOC_LAYOUT_MODEL,
            "region_count": len(regions),
            "ordered_region_count": sum(region.normalized_order is not None for region in regions),
            "ambiguous": bool(conflicts) or not regions,
        }
    )
    if not regions:
        page.warnings.append("PP-DocLayoutV3 returned no regions; Luna review is required.")
    page.chunks = build_layout_chunks(page.blocks)


def _layout_cache_key(page: Any, namespace: str) -> str:
    # Keep preprocessing variants isolated by hashing the exact model input.
    digest = page_image_hash(page.image)
    return f"layout:{LAYOUT_CACHE_VERSION}:{digest}:{namespace}"


def _finite_numbers(value: Any, count: int, field_name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise LayoutContractError(f"PP-DocLayoutV3 {field_name} is invalid.")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise LayoutContractError(f"PP-DocLayoutV3 {field_name} is invalid.")
    return result


def _coverage(block: list[float], region: list[float]) -> float:
    intersection = max(0.0, min(block[2], region[2]) - max(block[0], region[0])) * max(
        0.0, min(block[3], region[3]) - max(block[1], region[1])
    )
    return min(1.0, intersection / _area(block)) if _area(block) else 0.0


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
