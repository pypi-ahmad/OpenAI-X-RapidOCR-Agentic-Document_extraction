from types import SimpleNamespace

import pytest
from PIL import Image

import agentic_extractor.layout as layout_module
from agentic_extractor.layout import (
    PP_DOC_LAYOUT_MODEL,
    PP_DOC_LAYOUT_THRESHOLD,
    TABLE_STRUCTURE_CANDIDATE_THRESHOLD,
    LayoutContractError,
    LayoutSetupError,
    PPDocLayoutResource,
    apply_document_layout,
    create_pp_doclayout_engine,
    enrich_page_layout,
    normalize_layout_result,
)
from agentic_extractor.models import Block
from agentic_extractor.ocr import EngineProvenance, LocalParseResult
from agentic_extractor.parse import PageParse


def _raw_regions() -> dict:
    return {
        "input_path": None,
        "page_index": None,
        "boxes": [
            {
                "cls_id": 6,
                "label": "doc_title",
                "score": 0.97,
                "coordinate": [10, 10, 190, 50],
                "polygon_points": [[10, 10], [190, 10], [190, 50], [10, 50]],
                "order": 1,
            },
            {
                "cls_id": 22,
                "label": "text",
                "score": 0.91,
                "coordinate": [10, 70, 190, 180],
                "polygon_points": [[10, 70], [190, 70], [190, 180], [10, 180]],
                "order": 2,
            },
        ],
    }


def test_normalizes_document_layout_v3_contract() -> None:
    regions = normalize_layout_result(_raw_regions(), page=3, width=200, height=200)

    assert [region.id for region in regions] == ["p3-l1", "p3-l2"]
    assert regions[0].model == PP_DOC_LAYOUT_MODEL
    assert regions[0].bbox == [0.05, 0.05, 0.95, 0.25]
    assert regions[0].polygon == [[10.0, 10.0], [190.0, 10.0], [190.0, 50.0], [10.0, 50.0]]
    assert regions[0].raw_order == 1
    assert PP_DOC_LAYOUT_THRESHOLD == 0.5


def test_rejects_taxonomy_drift_and_invalid_geometry() -> None:
    value = _raw_regions()
    value["boxes"][0]["label"] = "made_up"
    with pytest.raises(LayoutContractError, match="taxonomy"):
        normalize_layout_result(value, page=1, width=200, height=200)

    value = _raw_regions()
    value["boxes"][0]["coordinate"] = [20, 20, 10, 10]
    with pytest.raises(LayoutContractError, match="coordinate"):
        normalize_layout_result(value, page=1, width=200, height=200)


def test_layout_links_and_order_are_additive_to_rapidocr_evidence() -> None:
    page = PageParse(
        page=1,
        width=200,
        height=200,
        blocks=[
            Block(id="p1-b1", page=1, text="Body", bbox=[0.1, 0.4, 0.9, 0.8]),
            Block(id="p1-b2", page=1, text="Title", bbox=[0.1, 0.08, 0.9, 0.2]),
        ],
        raw_evidence={"texts": ["Body", "Title"]},
    )
    original_raw = dict(page.raw_evidence)

    enrich_page_layout(page, normalize_layout_result(_raw_regions(), 1, 200, 200))

    assert page.raw_evidence == original_raw
    assert [block.id for block in page.blocks] == ["p1-b2", "p1-b1"]
    assert [link.primary_region_id for link in page.layout_block_links] == ["p1-l2", "p1-l1"]
    assert page.layout_signals["source"] == "pp-doclayout-v3"
    assert page.layout_signals["native_layout"] is True


def test_empty_layout_is_valid_but_forces_review() -> None:
    page = PageParse(
        page=1,
        width=100,
        height=100,
        blocks=[Block(id="p1-b1", page=1, text="Text", bbox=[0.1, 0.1, 0.9, 0.2])],
    )

    enrich_page_layout(page, [])

    assert page.layout_regions == []
    assert page.layout_signals["ambiguous"] is True
    assert "returned no regions" in page.warnings[-1]


class FakeLayoutResource:
    device = "GPU"
    version = "3.4.0"
    model = PP_DOC_LAYOUT_MODEL
    calls: list[list[int]]

    def __init__(self) -> None:
        self.calls = []

    def predict(self, pages: list[SimpleNamespace]) -> dict[int, dict]:
        self.calls.append([page.number for page in pages])
        return {page.number: _raw_regions() for page in pages}


def test_fake_resource_shape_matches_selected_page_images() -> None:
    resource = FakeLayoutResource()
    pages = [SimpleNamespace(number=2, image=Image.new("RGB", (200, 200)))]

    assert list(resource.predict(pages)) == [2]
    assert resource.calls == [[2]]


def test_layout_rebuilds_document_markdown_after_table_enrichment() -> None:
    class TableClient:
        def predict(self, pages: list[SimpleNamespace]) -> dict[int, dict]:
            return {
                page.number: {
                    "boxes": [
                        {
                            "cls_id": 21,
                            "label": "table",
                            "score": 0.98,
                            "coordinate": [0, 0, 200, 100],
                            "polygon_points": [[0, 0], [200, 0], [200, 100], [0, 100]],
                            "order": 1,
                        }
                    ]
                }
                for page in pages
            }

        def predict_tables(self, tables: list[SimpleNamespace]) -> dict[str, dict]:
            return {
                table.id: {
                    "classifier": {"label_names": ["wired_table"], "scores": [0.99]},
                    "structure_model": "SLANeXt_wired",
                    "structure_geometry_valid": True,
                    "structure": {
                        "bbox": [[0, 0, 100, 50], [100, 0, 200, 50]],
                        "structure": ["<table><tr><td></td><td></td></tr></table>"],
                        "structure_score": 0.99,
                    },
                }
                for table in tables
            }

    page = PageParse(
        page=4,
        width=200,
        height=100,
        blocks=[
            Block(id="p4-b1", page=4, text="FENTANYL", bbox=[0.05, 0.1, 0.45, 0.4]),
            Block(id="p4-b2", page=4, text="BACLOFEN", bbox=[0.55, 0.1, 0.95, 0.4]),
        ],
    )
    local = LocalParseResult(
        document_metadata={},
        selected_pages=[4],
        pages=[page],
        markdown="<!-- page: 4 -->\n\nFENTANYL\n\nBACLOFEN",
        engine=EngineProvenance(name="RapidOCR", version="test", device="CPU"),
        timings={},
    )
    resource = PPDocLayoutResource(
        client=TableClient(),
        device="CPU",
        version="test",
        paddle_version="test",
        cache_namespace="test-document-markdown-table-enrichment-v1",
    )

    apply_document_layout(
        local,
        [SimpleNamespace(number=4, image=Image.new("RGB", (200, 100)))],
        resource,
    )

    assert local.pages[0].table_structures[0].status == "valid"
    assert "<table><tr><td>FENTANYL</td><td>BACLOFEN</td></tr></table>" in local.markdown


def test_low_confidence_table_region_skips_structure_models() -> None:
    class LowConfidenceTableClient:
        table_calls = 0

        def predict(self, pages: list[SimpleNamespace]) -> dict[int, dict]:
            return {
                page.number: {
                    "boxes": [
                        {
                            "cls_id": 21,
                            "label": "table",
                            "score": TABLE_STRUCTURE_CANDIDATE_THRESHOLD - 0.01,
                            "coordinate": [0, 0, 200, 100],
                            "polygon_points": [[0, 0], [200, 0], [200, 100], [0, 100]],
                            "order": 1,
                        }
                    ]
                }
                for page in pages
            }

        def predict_tables(self, tables: list[SimpleNamespace]) -> dict[str, dict]:
            self.table_calls += len(tables)
            raise AssertionError("Low-confidence table regions must not reach structure models.")

    client = LowConfidenceTableClient()
    page = PageParse(
        page=1,
        width=200,
        height=100,
        blocks=[Block(id="p1-b1", page=1, text="Ordinary paragraph", bbox=[0.1, 0.1, 0.9, 0.3])],
    )
    local = LocalParseResult(
        document_metadata={},
        selected_pages=[1],
        pages=[page],
        markdown="",
        engine=EngineProvenance(name="RapidOCR", version="test", device="CPU"),
        timings={},
    )
    resource = PPDocLayoutResource(
        client=client,
        device="CPU",
        version="test",
        paddle_version="test",
        cache_namespace="test-skip-low-confidence-tables-v1",
    )

    apply_document_layout(
        local,
        [SimpleNamespace(number=1, image=Image.new("RGB", (200, 100)))],
        resource,
    )

    assert client.table_calls == 0
    assert local.pages[0].table_structures == []
    assert local.adaptive_processing["table_structure"]["candidate_regions"] == 0
    assert local.adaptive_processing["table_structure"]["skipped_regions"] == 1
    assert local.adaptive_processing["table_structure"]["skipped_pages"] == [1]


def test_engine_retries_transient_gpu_fallback_in_fresh_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    project = tmp_path / "worker"
    project.mkdir()
    python = project / "python.exe"
    worker = project / "worker.py"
    python.touch()
    worker.touch()
    health = iter(
        [
            {
                "model": PP_DOC_LAYOUT_MODEL,
                "device": "CPU",
                "gpu_available": True,
                "warning": "GPU probe failed.",
            },
            {
                "model": PP_DOC_LAYOUT_MODEL,
                "device": "GPU",
                "gpu_available": True,
                "paddlex_version": "3.4.0",
                "paddle_version": "3.2.0",
            },
        ]
    )
    clients = []

    class Client:
        def __init__(self, *_args) -> None:
            self.closed = False
            self.batch_size = 1
            clients.append(self)

        def request(self, _payload, *, timeout):
            assert timeout == 300
            return next(health)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(layout_module, "_runtime_paths", lambda: (project, python))
    monkeypatch.setattr(layout_module, "_WorkerClient", Client)

    resource = create_pp_doclayout_engine()

    assert resource.device == "GPU"
    assert clients[0].closed is True
    assert clients[1].closed is False
    assert clients[1].batch_size == 2


def test_engine_does_not_cache_persistent_cpu_fallback_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    project = tmp_path / "worker"
    project.mkdir()
    python = project / "python.exe"
    worker = project / "worker.py"
    python.touch()
    worker.touch()
    clients = []

    class Client:
        def __init__(self, *_args) -> None:
            self.closed = False
            clients.append(self)

        def request(self, _payload, *, timeout):
            assert timeout == 300
            return {
                "model": PP_DOC_LAYOUT_MODEL,
                "device": "CPU",
                "gpu_available": True,
                "warning": "GPU probe failed.",
            }

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(layout_module, "_runtime_paths", lambda: (project, python))
    monkeypatch.setattr(layout_module, "_WorkerClient", Client)

    with pytest.raises(LayoutSetupError, match="failed twice"):
        create_pp_doclayout_engine()

    assert len(clients) == 2
    assert all(client.closed for client in clients)
