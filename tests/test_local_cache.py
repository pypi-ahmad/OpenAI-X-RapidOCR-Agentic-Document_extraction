import io
from types import SimpleNamespace

from PIL import Image

from agentic_extractor.cache import ByteLRUCache, clear_local_caches
from agentic_extractor.config import Settings
from agentic_extractor.ingest import IngestedDocument, InputPage, load_document
from agentic_extractor.layout import PPDocLayoutResource, apply_document_layout
from agentic_extractor.models import Block
from agentic_extractor.ocr import (
    EngineProvenance,
    LocalParseResult,
    OCRResource,
    parse_document_local,
)
from agentic_extractor.parse import PageParse


def _png_bytes(color: str = "white") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (20, 10), color).save(output, "PNG")
    return output.getvalue()


def test_byte_lru_is_bounded_and_oversized_values_bypass_cache() -> None:
    cache = ByteLRUCache(max_bytes=5, max_entries=2)

    assert cache.put("a", b"12")
    assert cache.put("b", b"345")
    assert cache.get("a") == b"12"
    assert cache.put("c", b"67")
    assert cache.get("b") is None
    assert cache.put("large", b"123456") is False
    assert cache.get("large") is None


def test_rendered_pages_are_reused_by_content_with_independent_images() -> None:
    clear_local_caches()
    source = _png_bytes()

    first = load_document("first.png", source)
    second = load_document("renamed.png", source)

    assert first.document_sha256 == second.document_sha256
    assert first.pages[0].page_hash == second.pages[0].page_hash
    assert first.render_cache_hits == 0
    assert second.render_cache_hits == 1
    first.pages[0].image.putpixel((0, 0), (0, 0, 0))
    assert second.pages[0].image.getpixel((0, 0)) == (255, 255, 255)
    clear_local_caches()


def test_pdf_render_cache_is_invalidated_by_render_dpi() -> None:
    clear_local_caches()
    output = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(output, "PDF")
    source = output.getvalue()

    first = load_document("scan.pdf", source, Settings(render_dpi=72))
    second = load_document("scan.pdf", source, Settings(render_dpi=72))
    different_dpi = load_document("scan.pdf", source, Settings(render_dpi=144))

    assert first.render_cache_misses == 1
    assert second.render_cache_hits == 1
    assert different_dpi.render_cache_misses == 1
    assert first.pages[0].image.size != different_dpi.pages[0].image.size
    clear_local_caches()


def test_successful_ocr_is_reused_but_failures_are_not(monkeypatch) -> None:
    clear_local_caches()
    document = IngestedDocument(
        "scan.png",
        "image/png",
        _png_bytes(),
        [InputPage(1, Image.new("RGB", (20, 10), "white"))],
    )
    calls = 0

    def fake_ocr(resource: OCRResource, number: int, image: Image.Image) -> PageParse:
        nonlocal calls
        calls += 1
        return PageParse(
            page=number,
            width=image.width,
            height=image.height,
            image_bytes=b"jpeg",
            blocks=[Block(id="p1-b1", page=1, text="Invoice")],
            ocr_seconds=0.25,
        )

    monkeypatch.setattr("agentic_extractor.ocr.ocr_page", fake_ocr)
    resource = OCRResource(object(), "CPU", cache_namespace="rapidocr-test")

    first = parse_document_local(document, {1}, resource)
    second = parse_document_local(document, {1}, resource)

    assert calls == 1
    assert first.adaptive_processing["ocr_cache_hits"] == 0
    assert second.adaptive_processing["ocr_cache_hits"] == 1
    assert second.timings["ocr_seconds"] == 0
    second.pages[0].blocks[0].text = "mutated"
    third = parse_document_local(document, {1}, resource)
    assert third.pages[0].blocks[0].text == "Invoice"

    def failing_ocr(resource: OCRResource, number: int, image: Image.Image) -> PageParse:
        nonlocal calls
        calls += 1
        raise ValueError("unreadable")

    clear_local_caches()
    monkeypatch.setattr("agentic_extractor.ocr.ocr_page", failing_ocr)
    parse_document_local(document, {1}, resource)
    parse_document_local(document, {1}, resource)
    assert calls == 3
    clear_local_caches()


def test_ocr_cache_key_changes_with_pixels_and_engine_namespace(monkeypatch) -> None:
    clear_local_caches()
    calls = 0

    def fake_ocr(resource: OCRResource, number: int, image: Image.Image) -> PageParse:
        nonlocal calls
        calls += 1
        return PageParse(page=number, width=image.width, height=image.height, image_bytes=b"jpeg")

    monkeypatch.setattr("agentic_extractor.ocr.ocr_page", fake_ocr)
    source = _png_bytes()
    white = IngestedDocument(
        "scan.png", "image/png", source, [InputPage(1, Image.new("RGB", (20, 10), "white"))]
    )
    black = IngestedDocument(
        "scan.png", "image/png", source, [InputPage(1, Image.new("RGB", (20, 10), "black"))]
    )

    parse_document_local(white, {1}, OCRResource(object(), "CPU", cache_namespace="v1"))
    parse_document_local(black, {1}, OCRResource(object(), "CPU", cache_namespace="v1"))
    parse_document_local(white, {1}, OCRResource(object(), "CPU", cache_namespace="v2"))

    assert calls == 3
    clear_local_caches()


def test_derived_cache_uses_current_pixels_after_preprocessing(monkeypatch) -> None:
    clear_local_caches()
    calls = 0

    def fake_ocr(resource: OCRResource, number: int, image: Image.Image) -> PageParse:
        nonlocal calls
        calls += 1
        return PageParse(page=number, width=image.width, height=image.height, image_bytes=b"jpeg")

    monkeypatch.setattr("agentic_extractor.ocr.ocr_page", fake_ocr)
    page = InputPage(
        1,
        Image.new("RGB", (20, 10), "white"),
        page_hash="stale-render-hash",
    )
    document = IngestedDocument("scan.png", "image/png", b"source", [page])
    resource = OCRResource(object(), "CPU", cache_namespace="same-engine")

    parse_document_local(document, {1}, resource)
    page.image = Image.new("RGB", (20, 10), "black")
    parse_document_local(document, {1}, resource)

    assert calls == 2
    clear_local_caches()


def test_ocr_cache_reuses_identical_page_pixels_and_rebinds_grounding(monkeypatch) -> None:
    clear_local_caches()
    calls = 0

    def fake_ocr(resource: OCRResource, number: int, image: Image.Image) -> PageParse:
        nonlocal calls
        calls += 1
        return PageParse(
            page=number,
            width=image.width,
            height=image.height,
            image_bytes=b"jpeg",
            blocks=[Block(id=f"p{number}-b1", page=number, text="Same page")],
        )

    monkeypatch.setattr("agentic_extractor.ocr.ocr_page", fake_ocr)
    image = Image.new("RGB", (20, 10), "white")
    first = IngestedDocument("a.pdf", "application/pdf", b"a", [InputPage(1, image)])
    second = IngestedDocument("b.pdf", "application/pdf", b"b", [InputPage(7, image.copy())])
    resource = OCRResource(object(), "CPU", cache_namespace="same-engine")

    parse_document_local(first, {1}, resource)
    result = parse_document_local(second, {7}, resource)

    assert calls == 1
    assert result.adaptive_processing["ocr_cache_hits"] == 1
    assert result.pages[0].page == 7
    assert result.pages[0].blocks[0].id == "p7-b1"
    assert result.pages[0].blocks[0].page == 7
    clear_local_caches()


def test_layout_and_table_caches_reuse_identical_page_pixels_with_new_ids() -> None:
    clear_local_caches()

    class Client:
        layout_calls = 0
        table_calls = 0

        def predict(self, pages):
            self.layout_calls += len(pages)
            return {
                page.number: {
                    "boxes": [
                        {
                            "cls_id": 21,
                            "label": "table",
                            "score": 0.99,
                            "coordinate": [0, 0, 200, 100],
                            "polygon_points": [[0, 0], [200, 0], [200, 100], [0, 100]],
                            "order": 1,
                        }
                    ]
                }
                for page in pages
            }

        def predict_tables(self, tables):
            self.table_calls += len(tables)
            return {
                table.id: {
                    "classifier": {"label_names": ["wired_table"], "scores": [0.99]},
                    "structure_model": "SLANeXt_wired",
                    "structure_geometry_valid": True,
                    "structure": {
                        "bbox": [[0, 0, 100, 100], [100, 0, 200, 100]],
                        "structure": ["<table><tr><td></td><td></td></tr></table>"],
                        "structure_score": 0.99,
                    },
                }
                for table in tables
            }

    client = Client()
    resource = PPDocLayoutResource(
        client=client,
        device="CPU",
        version="test",
        paddle_version="test",
        cache_namespace="same-layout-and-table-models",
    )
    image = Image.new("RGB", (200, 100), "white")

    def local(page_number: int) -> LocalParseResult:
        page = PageParse(
            page=page_number,
            width=200,
            height=100,
            blocks=[
                Block(id=f"p{page_number}-b1", page=page_number, text="A", bbox=[0, 0, 0.5, 1]),
                Block(id=f"p{page_number}-b2", page=page_number, text="B", bbox=[0.5, 0, 1, 1]),
            ],
        )
        return LocalParseResult(
            document_metadata={},
            selected_pages=[page_number],
            pages=[page],
            markdown="",
            engine=EngineProvenance(name="RapidOCR", version="test", device="CPU"),
            timings={},
        )

    first = local(1)
    second = local(7)
    apply_document_layout(first, [SimpleNamespace(number=1, image=image)], resource)
    apply_document_layout(second, [SimpleNamespace(number=7, image=image.copy())], resource)

    assert client.layout_calls == 1
    assert client.table_calls == 1
    assert second.adaptive_processing["pp_doclayout_v3"]["cache_hits"] == 1
    assert second.adaptive_processing["table_structure"]["cache_hits"] == 1
    assert second.pages[0].layout_regions[0].id == "p7-l1"
    assert second.pages[0].table_structures[0].id == "p7-l1-table"
    clear_local_caches()


def test_ocr_cache_is_bypassed_without_a_stable_engine_namespace(monkeypatch) -> None:
    clear_local_caches()
    calls = 0

    def fake_ocr(resource: OCRResource, number: int, image: Image.Image) -> PageParse:
        nonlocal calls
        calls += 1
        return PageParse(page=number, width=20, height=10, image_bytes=b"jpeg")

    monkeypatch.setattr("agentic_extractor.ocr.ocr_page", fake_ocr)
    document = IngestedDocument(
        "scan.png",
        "image/png",
        _png_bytes(),
        [InputPage(1, Image.new("RGB", (20, 10), "white"))],
    )
    resource = OCRResource(object(), "CPU")

    parse_document_local(document, {1}, resource)
    parse_document_local(document, {1}, resource)

    assert calls == 2
    clear_local_caches()


def test_mixed_ocr_hits_and_misses_preserve_page_order(monkeypatch) -> None:
    clear_local_caches()
    calls: list[int] = []

    def fake_ocr(resource: OCRResource, number: int, image: Image.Image) -> PageParse:
        calls.append(number)
        return PageParse(page=number, width=20, height=10, image_bytes=b"jpeg")

    monkeypatch.setattr("agentic_extractor.ocr.ocr_page", fake_ocr)
    document = IngestedDocument(
        "scan.pdf",
        "application/pdf",
        b"same document",
        [
            InputPage(1, Image.new("RGB", (20, 10), "white")),
            InputPage(2, Image.new("RGB", (20, 10), "black")),
        ],
    )
    resource = OCRResource(object(), "CPU", cache_namespace="mixed-test")

    parse_document_local(document, {1}, resource)
    result = parse_document_local(document, {1, 2}, resource)

    assert calls == [1, 2]
    assert [page.page for page in result.pages] == [1, 2]
    assert result.adaptive_processing["ocr_cache_hits"] == 1
    assert result.adaptive_processing["ocr_cache_misses"] == 1
    clear_local_caches()
