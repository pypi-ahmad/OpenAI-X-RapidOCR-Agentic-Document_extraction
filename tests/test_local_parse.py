import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from rapidocr.utils.output import RapidOCROutput

import agentic_extractor.ocr as ocr_module
from agentic_extractor.config import Settings
from agentic_extractor.ingest import IngestedDocument, InputPage
from agentic_extractor.models import Block, CheckboxRecord, CheckboxState
from agentic_extractor.ocr import (
    OCRResource,
    _cuda_device_available,
    create_rapidocr_engine,
    ocr_page,
    parse_document_local,
)
from agentic_extractor.parse import (
    PageParse,
    document_markdown,
    document_markdown_with_checkboxes,
    reconstruct_layout,
)


class ExactRapidOutputEngine:
    return_word_box = None
    text_det = SimpleNamespace(
        session=SimpleNamespace(
            session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
        )
    )

    def __call__(self, image, *, return_word_box=False):
        self.return_word_box = return_word_box
        return RapidOCROutput(
            img=image,
            boxes=np.array([[[10, 10], [80, 10], [80, 30], [10, 30]]]),
            txts=("Invoice",),
            scores=(0.91,),
            word_results=(("Invoice", 0.91, [[10, 10], [80, 30]]),),
            elapse_list=[0.1, 0.2, 0.3],
        )


def test_cuda_detection_accepts_conventional_provider_when_plugin_devices_are_cpu_only(
    monkeypatch,
) -> None:
    import onnxruntime as ort

    monkeypatch.setattr(
        ort,
        "get_ep_devices",
        lambda: [SimpleNamespace(ep_name="CPUExecutionProvider")],
    )
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(ort, "get_device", lambda: "GPU")

    assert _cuda_device_available() is True


def test_rapidocr_cuda_uses_stable_single_item_batches(monkeypatch) -> None:
    constructor_arguments: list[dict[str, object]] = []

    def rapidocr(*args, **kwargs):
        constructor_arguments.append(kwargs)
        return object()

    monkeypatch.setattr("agentic_extractor.ocr._configure_windows_cuda_dlls", lambda: None)
    monkeypatch.setattr("agentic_extractor.ocr._cuda_device_available", lambda: True)
    monkeypatch.setattr("rapidocr.RapidOCR", rapidocr)

    resource = create_rapidocr_engine(prefer_cuda=True, spawnable=False)

    assert resource.device == "CUDA"
    assert constructor_arguments == [
        {
            "params": {
                "EngineConfig.onnxruntime.use_cuda": True,
                "Cls.cls_batch_num": 1,
                "Rec.rec_batch_num": 1,
            }
        }
    ]


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows DLL search behavior")
def test_windows_cuda_search_registers_bin_x64(monkeypatch, tmp_path) -> None:
    toolkit = tmp_path / "cuda"
    directory = toolkit / "bin" / "x64"
    directory.mkdir(parents=True)
    added_directories: list[str] = []
    monkeypatch.setenv("CUDA_PATH", str(toolkit))
    monkeypatch.setenv("PATH", "existing")
    monkeypatch.setattr(ocr_module, "_CUDA_DLL_HANDLES", [])
    monkeypatch.setattr(ocr_module, "_CUDA_DLL_DIRECTORY", None)
    monkeypatch.setattr(
        ocr_module.os,
        "add_dll_directory",
        lambda value: added_directories.append(value) or object(),
    )

    configured = ocr_module._configure_windows_cuda_dlls()

    assert configured == str(directory)
    assert added_directories == [str(directory)]
    assert ocr_module.os.environ["PATH"].split(ocr_module.os.pathsep)[0] == str(directory)


def test_rapidocr_skips_cuda_initialization_without_a_registered_device(monkeypatch) -> None:
    constructor_arguments: list[dict[str, object]] = []

    def rapidocr(*args, **kwargs):
        constructor_arguments.append(kwargs)
        return object()

    monkeypatch.setattr("agentic_extractor.ocr._cuda_device_available", lambda: False)
    monkeypatch.setattr("rapidocr.RapidOCR", rapidocr)

    resource = create_rapidocr_engine(prefer_cuda=True)

    assert constructor_arguments == [{}]
    assert resource.device == "CPU"
    assert resource.warning == "No usable CUDA execution-provider device; RapidOCR uses CPU."


def test_cpu_worker_configures_onnx_threads(monkeypatch) -> None:
    constructor_arguments: list[dict[str, object]] = []

    def rapidocr(*args, **kwargs):
        constructor_arguments.append(kwargs)
        return object()

    monkeypatch.setattr("rapidocr.RapidOCR", rapidocr)

    resource = create_rapidocr_engine(prefer_cuda=False, cpu_threads=3, spawnable=False)

    assert constructor_arguments == [
        {
            "params": {
                "EngineConfig.onnxruntime.intra_op_num_threads": 3,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        }
    ]
    assert resource.cpu_threads == 3


def test_rapidocr_output_is_normalized_without_invented_orientation() -> None:
    engine = ExactRapidOutputEngine()
    page = ocr_page(
        OCRResource(engine, "CPU"),
        3,
        Image.new("RGB", (100, 100), "white"),
    )
    assert page.blocks[0].text == "Invoice"
    assert page.blocks[0].ocr_score == 0.91
    assert page.blocks[0].bbox == [0.1, 0.1, 0.8, 0.3]
    assert page.layout_image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert page.raw_evidence["elapse_list"] == [0.1, 0.2, 0.3]
    word_results = page.raw_evidence["word_results"]
    assert isinstance(word_results, list)
    assert word_results[0][0] == "Invoice"
    assert engine.return_word_box is True
    assert page.layout_signals["orientation_degrees"] is None
    assert page.layout_signals["source"] == "derived_text_geometry"
    assert page.layout_signals["native_layout"] is False
    assert page.raw_evidence["score_semantics"] == "recognition_confidence"
    assert page.raw_evidence["orientation_scope"] == "text_line_0_180"
    assert page.warnings == []
    assert page.chunks[0].source_block_ids == ["p3-b1"]
    assert page.engine_elapsed_seconds == 0.6


def test_selected_pages_keep_source_order(monkeypatch) -> None:
    document = IngestedDocument(
        "scan.tiff",
        "image/tiff",
        b"source",
        [InputPage(number, Image.new("RGB", (20, 20))) for number in range(1, 5)],
    )
    monkeypatch.setattr(
        "agentic_extractor.ocr.ocr_page",
        lambda resource, number, image: SimpleNamespace(
            page=number,
            width=20,
            height=20,
            blocks=[],
            image_bytes=b"jpeg",
            ocr_seconds=0.01,
            engine_elapsed_seconds=0.01,
            raw_evidence={},
            layout_signals={},
            warnings=["orientation unavailable"],
            status="completed",
            markdown=f"page {number}",
        ),
    )
    result = parse_document_local(document, {4, 2}, OCRResource(object(), "CPU"))
    assert [page.page for page in result.pages] == [2, 4]
    assert result.selected_pages == [2, 4]
    assert "page 2" in result.markdown and "page 4" in result.markdown
    assert result.warnings == [
        "Page 2: orientation unavailable",
        "Page 4: orientation unavailable",
    ]


def test_cpu_pages_run_concurrently_on_dedicated_bounded_engines(monkeypatch) -> None:
    document = IngestedDocument(
        "scan.pdf",
        "application/pdf",
        b"source",
        [InputPage(number, Image.new("RGB", (20, 20))) for number in range(1, 5)],
    )
    created: list[object] = []

    def spawn(cpu_threads: int) -> OCRResource:
        assert cpu_threads == 4
        engine = object()
        created.append(engine)
        return OCRResource(engine, "CPU")

    primary = object()
    active = 0
    peak_active = 0
    active_engines: set[int] = set()
    lock = threading.Lock()

    def fake_ocr(resource: OCRResource, number: int, image: Image.Image) -> PageParse:
        nonlocal active, peak_active
        engine_id = id(resource.engine)
        with lock:
            assert engine_id not in active_engines
            active_engines.add(engine_id)
            active += 1
            peak_active = max(peak_active, active)
        time.sleep((5 - number) * 0.015)
        with lock:
            active -= 1
            active_engines.remove(engine_id)
        return PageParse(
            page=number,
            width=20,
            height=20,
            image_bytes=b"jpeg",
            ocr_seconds=0.06,
            blocks=[Block(id=f"p{number}-b1", page=number, text=f"page {number}")],
        )

    monkeypatch.setenv("ADE_OCR_MAX_WORKERS", "4")
    monkeypatch.setattr("agentic_extractor.ocr.os.cpu_count", lambda: 16)
    monkeypatch.setattr("agentic_extractor.ocr.ocr_page", fake_ocr)

    result = parse_document_local(
        document,
        {1, 2, 3, 4},
        OCRResource(primary, "CPU", spawn=spawn, cpu_threads=4),
    )

    assert [page.page for page in result.pages] == [1, 2, 3, 4]
    assert peak_active >= 2
    assert len(created) == 3
    assert result.adaptive_processing["actual_workers"] == 4
    assert result.adaptive_processing["cpu_threads_per_worker"] == 4
    assert result.timings["ocr_concurrency_ratio"] > 1.5


def test_cpu_worker_models_stay_warm_across_documents(monkeypatch) -> None:
    documents = [
        IngestedDocument(
            f"scan-{run}.pdf",
            "application/pdf",
            f"source-{run}".encode(),
            [InputPage(number, Image.new("RGB", (20, 20))) for number in range(1, 5)],
        )
        for run in (1, 2)
    ]
    spawned: list[OCRResource] = []

    def spawn(cpu_threads: int) -> OCRResource:
        resource = OCRResource(object(), "CPU", cpu_threads=cpu_threads)
        spawned.append(resource)
        return resource

    monkeypatch.setenv("ADE_OCR_MAX_WORKERS", "4")
    monkeypatch.setattr("agentic_extractor.ocr.os.cpu_count", lambda: 16)
    monkeypatch.setattr(
        "agentic_extractor.ocr.ocr_page",
        lambda resource, number, image: PageParse(
            page=number, width=20, height=20, image_bytes=b"jpeg"
        ),
    )
    primary = OCRResource(object(), "CPU", spawn=spawn, cpu_threads=4)

    first = parse_document_local(documents[0], {1, 2, 3, 4}, primary)
    first_worker_ids = [id(worker) for worker in primary.warm_worker_pools[(4, 4)]]
    second = parse_document_local(documents[1], {1, 2, 3, 4}, primary)

    assert len(spawned) == 3
    assert [id(worker) for worker in primary.warm_worker_pools[(4, 4)]] == first_worker_ids
    assert first.adaptive_processing["actual_workers"] == 4
    assert second.adaptive_processing["actual_workers"] == 4


def test_cuda_and_non_spawnable_resources_are_sequential(monkeypatch) -> None:
    document = IngestedDocument(
        "scan.pdf",
        "application/pdf",
        b"source",
        [InputPage(number, Image.new("RGB", (20, 20))) for number in (1, 2)],
    )
    calls: list[int] = []

    def fake_ocr(resource: OCRResource, number: int, image: Image.Image) -> PageParse:
        calls.append(number)
        return PageParse(page=number, width=20, height=20, image_bytes=b"jpeg")

    monkeypatch.setattr("agentic_extractor.ocr.ocr_page", fake_ocr)
    monkeypatch.setenv("ADE_OCR_MAX_WORKERS", "4")
    cuda_spawns = 0

    def cuda_spawn(cpu_threads: int) -> OCRResource:
        nonlocal cuda_spawns
        cuda_spawns += 1
        return OCRResource(object(), "CUDA")

    cuda = parse_document_local(document, {1, 2}, OCRResource(object(), "CUDA", spawn=cuda_spawn))
    injected = parse_document_local(document, {1, 2}, OCRResource(object(), "CPU"))

    assert cuda.adaptive_processing["actual_workers"] == 1
    assert injected.adaptive_processing["actual_workers"] == 1
    assert cuda_spawns == 0
    assert calls == [1, 2, 1, 2]


def test_worker_factory_failure_reduces_pool_without_losing_pages(monkeypatch) -> None:
    document = IngestedDocument(
        "scan.pdf",
        "application/pdf",
        b"source",
        [InputPage(number, Image.new("RGB", (20, 20))) for number in (1, 2)],
    )
    monkeypatch.setenv("ADE_OCR_MAX_WORKERS", "4")
    monkeypatch.setattr("agentic_extractor.ocr.os.cpu_count", lambda: 16)
    monkeypatch.setattr(
        "agentic_extractor.ocr.ocr_page",
        lambda resource, number, image: PageParse(
            page=number, width=20, height=20, image_bytes=b"jpeg"
        ),
    )

    def broken_spawn(cpu_threads: int) -> OCRResource:
        raise MemoryError("worker model allocation")

    result = parse_document_local(
        document, {1, 2}, OCRResource(object(), "CPU", spawn=broken_spawn)
    )

    assert [page.page for page in result.pages] == [1, 2]
    assert result.adaptive_processing["actual_workers"] == 1
    assert any("worker pool reduced to 1" in warning for warning in result.warnings)


def test_parallel_page_failure_is_isolated_and_ordered(monkeypatch) -> None:
    document = IngestedDocument(
        "scan.pdf",
        "application/pdf",
        b"source",
        [InputPage(number, Image.new("RGB", (20, 20))) for number in (1, 2, 3)],
    )

    def fake_ocr(resource: OCRResource, number: int, image: Image.Image) -> PageParse:
        if number == 2:
            raise ValueError("unreadable")
        return PageParse(page=number, width=20, height=20, image_bytes=b"jpeg")

    monkeypatch.setattr("agentic_extractor.ocr.os.cpu_count", lambda: 16)
    monkeypatch.setattr("agentic_extractor.ocr.ocr_page", fake_ocr)
    resource = OCRResource(
        object(),
        "CPU",
        spawn=lambda threads: OCRResource(object(), "CPU", cpu_threads=threads),
        cpu_threads=5,
    )

    result = parse_document_local(document, {1, 2, 3}, resource)

    assert [page.page for page in result.pages] == [1, 2, 3]
    assert result.page_statuses == {1: "completed", 2: "failed", 3: "completed"}
    assert result.failed_pages == [2]
    assert any(warning.startswith("Page 2 failed:") for warning in result.warnings)


def test_ocr_worker_override_is_bounded_and_validated(monkeypatch) -> None:
    monkeypatch.delenv("ADE_OCR_MAX_WORKERS", raising=False)
    assert Settings().ocr_max_workers == 4
    monkeypatch.setenv("ADE_OCR_MAX_WORKERS", "2")
    assert Settings().ocr_max_workers == 2
    monkeypatch.setenv("ADE_OCR_MAX_WORKERS", "9")

    with pytest.raises(RuntimeError, match="ADE_OCR_MAX_WORKERS"):
        _ = Settings().ocr_max_workers


def test_layout_markdown_preserves_evidence_backed_structure_and_page_boundaries() -> None:
    blocks = [
        Block(id="p1-b4", page=1, text="Item | Qty | Price", bbox=[0.1, 0.7, 0.8, 0.75]),
        Block(id="p1-b5", page=1, text="Widget | 2 | 21", bbox=[0.1, 0.8, 0.8, 0.85]),
        Block(id="p1-b2", page=1, text="- First item", bbox=[0.1, 0.4, 0.5, 0.45]),
        Block(id="p1-b1", page=1, text="Invoice", bbox=[0.1, 0.1, 0.6, 0.3]),
        Block(id="p1-b3", page=1, text="Total: 42.00", bbox=[0.1, 0.55, 0.5, 0.6]),
    ]
    ordered, signals = reconstruct_layout(blocks)
    page = PageParse(page=1, width=100, height=100, blocks=ordered)

    assert [block.id for block in ordered] == ["p1-b1", "p1-b2", "p1-b3", "p1-b4", "p1-b5"]
    assert signals["reading_order"] == "top-to-bottom-left-to-right"
    assert page.markdown == (
        "## Invoice\n\n- First item\n\nTotal: 42.00\n\n"
        "<table><tr><th>Item</th><th>Qty</th><th>Price</th></tr>"
        "<tr><td>Widget</td><td>2</td><td>21</td></tr></table>"
    )
    assert document_markdown([page]).startswith("<!-- page: 1 -->\n\n## Invoice")


def test_inferred_table_html_escapes_untrusted_cell_text() -> None:
    blocks = [
        Block(id="p1-b1", page=1, text="Name | Value | Note", bbox=[0.1, 0.1, 0.8, 0.2]),
        Block(
            id="p1-b2",
            page=1,
            text="A&B | <script>alert(1)</script> | safe",
            bbox=[0.1, 0.3, 0.8, 0.4],
        ),
    ]
    ordered, _ = reconstruct_layout(blocks)

    page = PageParse(page=1, width=100, height=100, blocks=ordered)

    assert "A&amp;B" in page.markdown
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page.markdown
    assert "<script>" not in page.markdown


def test_final_markdown_publishes_only_accepted_checkbox_decisions() -> None:
    page = PageParse(
        page=1,
        width=100,
        height=100,
        blocks=[Block(id="p1-b1", page=1, text="Form", bbox=[0.1, 0.1, 0.5, 0.2])],
    )
    checkboxes = [
        CheckboxRecord(
            id="accepted",
            page=1,
            label="Accepted option",
            state=CheckboxState.CHECKED,
            control_bbox=[0.1, 0.3, 0.2, 0.4],
            discovery_state=CheckboxState.CHECKED,
            decision_status="automated",
        ),
        CheckboxRecord(
            id="review",
            page=1,
            label="Uncertain option",
            state=CheckboxState.CHECKED,
            control_bbox=[0.1, 0.3, 0.2, 0.4],
            discovery_state=CheckboxState.CHECKED,
            decision_status="review_required",
        ),
    ]

    markdown = document_markdown_with_checkboxes([page], checkboxes)

    assert "[x] Accepted option" in markdown
    assert "Uncertain option" not in markdown


def test_final_markdown_does_not_publish_unlabelled_checkbox_internal_id() -> None:
    page = PageParse(page=1, width=100, height=100)
    checkbox = CheckboxRecord(
        id="p1-cv20",
        page=1,
        state=CheckboxState.CHECKED,
        control_bbox=[0.1, 0.3, 0.2, 0.4],
        discovery_state=CheckboxState.CHECKED,
        decision_status="automated",
    )

    markdown = document_markdown_with_checkboxes([page], [checkbox])

    assert "p1-cv20" not in markdown
    assert "[x]" not in markdown
