import json

from agentic_extractor.landing_contract import build_landing_parse
from agentic_extractor.models import (
    Block,
    LayoutBlockLink,
    LayoutRegion,
    TableCellEvidence,
    TableStructureEvidence,
)
from agentic_extractor.ocr import EngineProvenance, LocalParseResult
from agentic_extractor.parse import PageParse, ParseChunk


def _result() -> LocalParseResult:
    page = PageParse(
        page=2,
        width=100,
        height=100,
        blocks=[
            Block(id="p2-b1", page=2, text="Item", bbox=[0.1, 0.2, 0.4, 0.3]),
            Block(id="p2-b2", page=2, text="Price", bbox=[0.5, 0.2, 0.8, 0.3]),
        ],
        table_structures=[
            TableStructureEvidence(
                id="p2-l1-table",
                page=2,
                layout_region_id="p2-l1",
                bbox=[0.05, 0.1, 0.9, 0.4],
                style="wired",
                classifier_score=0.9,
                structure_score=0.9,
                structure_model="SLANeXt_wired",
                cells=[
                    TableCellEvidence(
                        id="c1",
                        row=1,
                        column=1,
                        tag="th",
                        bbox=[0.1, 0.2, 0.4, 0.3],
                        source_block_ids=["p2-b1"],
                        source_text="Item",
                    ),
                    TableCellEvidence(
                        id="c2",
                        row=1,
                        column=2,
                        tag="th",
                        bbox=[0.5, 0.2, 0.8, 0.3],
                        source_block_ids=["p2-b2"],
                        source_text="Price",
                    ),
                ],
                markdown="<table><tr><th>Item</th><th>Price</th></tr></table>",
                status="valid",
            )
        ],
    )
    page.chunks = []
    return LocalParseResult(
        document_metadata={"filename": "table.pdf", "page_count": 2},
        selected_pages=[2],
        pages=[page],
        markdown="<!-- page: 2 -->\n\n<table><tr><th>Item</th><th>Price</th></tr></table>",
        engine=EngineProvenance("RapidOCR", "test", "CPU"),
        timings={"total_seconds": 1.25},
    )


def test_landing_parse_shape_and_unicode_ranges_reference_emitted_markdown() -> None:
    parsed = build_landing_parse(_result())

    assert set(parsed) == {"markdown", "metadata", "structure"}
    assert parsed["metadata"]["range_units"] == "unicode_codepoints"
    assert parsed["metadata"]["billing"] == {"service_tier": "local", "total_credits": None}
    assert parsed["structure"]["type"] == "document"
    page = parsed["structure"]["children"][0]
    assert page["grounding"]["page"] == 2
    assert parsed["markdown"][
        page["grounding"]["range"]["start"] : page["grounding"]["range"]["end"]
    ].startswith("<table>")


def test_landing_table_cells_are_zero_based_and_grounded_to_html_text() -> None:
    parsed = build_landing_parse(_result())
    table = parsed["structure"]["children"][0]["children"][0]

    assert table["type"] == "table"
    assert [(cell["row"], cell["col"]) for cell in table["children"]] == [(0, 0), (0, 1)]
    for cell in table["children"]:
        span = cell["grounding"]["range"]
        assert parsed["markdown"][span["start"] : span["end"]] in {"Item", "Price"}


def test_consecutive_ocr_chunks_in_one_layout_region_form_one_semantic_node() -> None:
    page = PageParse(
        page=1,
        width=100,
        height=100,
        blocks=[
            Block(id="p1-b1", page=1, text="First line", bbox=[0.1, 0.1, 0.8, 0.2]),
            Block(id="p1-b2", page=1, text="Second line", bbox=[0.1, 0.2, 0.8, 0.3]),
        ],
        layout_regions=[
            LayoutRegion(
                id="p1-l1",
                page=1,
                cls_id=2,
                label="text",
                score=0.9,
                coordinate=[10, 10, 80, 30],
                bbox=[0.1, 0.1, 0.8, 0.3],
            )
        ],
        layout_block_links=[
            LayoutBlockLink(
                page=1,
                block_id=block_id,
                primary_region_id="p1-l1",
                region_ids=["p1-l1"],
                block_coverage=1,
            )
            for block_id in ("p1-b1", "p1-b2")
        ],
    )
    result = LocalParseResult(
        document_metadata={"page_count": 1},
        selected_pages=[1],
        pages=[page],
        markdown="<!-- page: 1 -->\n\nFirst line\n\nSecond line",
        engine=EngineProvenance("RapidOCR", "test", "CPU"),
        timings={},
    )

    children = build_landing_parse(result)["structure"]["children"][0]["children"]

    assert len(children) == 1
    assert len(children[0]["atomic_grounding"]) == 2


def test_atomic_grounding_can_be_omitted_without_removing_node_grounding() -> None:
    parsed = build_landing_parse(_result(), include_atomic_grounding=False)
    page = parsed["structure"]["children"][0]
    table = page["children"][0]

    assert "atomic_grounding" not in json.dumps(parsed)
    assert page["grounding"]["page"] == 2
    assert table["grounding"]["box"] == {
        "xmin": 0.05,
        "ymin": 0.1,
        "xmax": 0.9,
        "ymax": 0.4,
    }


def test_accepted_attestation_chunk_remains_attestation_in_public_hierarchy() -> None:
    result = _result()
    page = result.pages[0]
    page.table_structures = []
    page.blocks = [
        Block(
            id="p2-b1",
            page=2,
            text="[SIGNED]\n[ILLEGIBLE_SIGNATURE]",
            bbox=[0.1, 0.2, 0.4, 0.3],
        )
    ]
    page.chunks = [
        ParseChunk(
            id="p2-s1",
            page=2,
            type="attestation",
            reading_order=1,
            text=page.blocks[0].text,
            markdown=page.blocks[0].text,
            source_block_ids=["p2-b1"],
            bbox=page.blocks[0].bbox,
            raw_scores=[None],
        )
    ]
    result.markdown = "<!-- page: 2 -->\n\n[SIGNED]\n[ILLEGIBLE_SIGNATURE]"

    node = build_landing_parse(result)["structure"]["children"][0]["children"][0]

    assert node["type"] == "attestation"


def test_marker_text_recovers_attestation_type_from_plain_chunk() -> None:
    result = _result()
    page = result.pages[0]
    page.table_structures = []
    page.blocks = [
        Block(
            id="p2-b1",
            page=2,
            text="[E-SIGNED]\nElectronically signed by: Clinician",
            bbox=[0.1, 0.2, 0.6, 0.3],
        )
    ]
    page.chunks = [
        ParseChunk(
            id="p2-s1",
            page=2,
            type="text",
            reading_order=1,
            text=page.blocks[0].text,
            markdown=page.blocks[0].text,
            source_block_ids=["p2-b1"],
            bbox=page.blocks[0].bbox,
            raw_scores=[None],
        )
    ]
    result.markdown = "<!-- page: 2 -->\n\n" + page.blocks[0].text

    node = build_landing_parse(result)["structure"]["children"][0]["children"][0]

    assert node["type"] == "attestation"


def test_accepted_scan_code_chunk_remains_scan_code_in_public_hierarchy() -> None:
    result = _result()
    page = result.pages[0]
    page.table_structures = []
    page.blocks = [Block(id="p2-b1", page=2, text="Scan me", bbox=[0.1, 0.2, 0.4, 0.3])]
    page.chunks = [
        ParseChunk(
            id="p2-s1",
            page=2,
            type="scan_code",
            reading_order=1,
            text="Scan me",
            markdown="Scan me",
            source_block_ids=["p2-b1"],
            bbox=page.blocks[0].bbox,
            raw_scores=[None],
        )
    ]
    result.markdown = "<!-- page: 2 -->\n\nScan me"

    node = build_landing_parse(result)["structure"]["children"][0]["children"][0]

    assert node["type"] == "scan_code"


def test_special_semantic_chunks_are_not_merged_by_shared_pp_region() -> None:
    result = _result()
    page = result.pages[0]
    page.table_structures = []
    page.blocks = [
        Block(id="p2-b1", page=2, text="Brand", bbox=[0.1, 0.1, 0.4, 0.2]),
        Block(id="p2-b2", page=2, text="Body", bbox=[0.1, 0.2, 0.4, 0.3]),
    ]
    page.layout_regions = [
        LayoutRegion(
            id="p2-l1",
            page=2,
            cls_id=22,
            label="text",
            score=0.9,
            coordinate=[10, 10, 40, 30],
            bbox=[0.1, 0.1, 0.4, 0.3],
        )
    ]
    page.layout_block_links = [
        LayoutBlockLink(
            page=2,
            block_id=block_id,
            primary_region_id="p2-l1",
            region_ids=["p2-l1"],
            block_coverage=1,
        )
        for block_id in ("p2-b1", "p2-b2")
    ]
    page.chunks = [
        ParseChunk(
            id="p2-s1",
            page=2,
            type="logo",
            reading_order=1,
            text="Brand",
            markdown="Brand",
            source_block_ids=["p2-b1"],
            bbox=page.blocks[0].bbox,
            raw_scores=[None],
        ),
        ParseChunk(
            id="p2-s2",
            page=2,
            type="text",
            reading_order=2,
            text="Body",
            markdown="Body",
            source_block_ids=["p2-b2"],
            bbox=page.blocks[1].bbox,
            raw_scores=[None],
        ),
    ]
    result.markdown = "<!-- page: 2 -->\n\nBrand\n\nBody"

    children = build_landing_parse(result)["structure"]["children"][0]["children"]

    assert [node["type"] for node in children] == ["logo", "text"]
