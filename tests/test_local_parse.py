from types import SimpleNamespace

import numpy as np
from PIL import Image
from rapidocr.utils.output import RapidOCROutput

from agentic_extractor.ingest import IngestedDocument, InputPage
from agentic_extractor.models import Block
from agentic_extractor.ocr import (
    OCRResource,
    _cuda_device_available,
    create_rapidocr_engine,
    ocr_page,
    parse_document_local,
)
from agentic_extractor.parse import PageParse, document_markdown, reconstruct_layout


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
        "| Item | Qty | Price |\n| --- | --- | --- |\n| Widget | 2 | 21 |"
    )
    assert document_markdown([page]).startswith("<!-- page: 1 -->\n\n## Invoice")
