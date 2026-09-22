import copy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agentic_extractor.budget import RequestBudget
from agentic_extractor.models import Block
from agentic_extractor.parse import PageParse, build_layout_chunks
from agentic_extractor.rich_document import (
    DocumentLink,
    VisualDecision,
    VisualObject,
    VisualVerification,
    apply_visual_objects,
    prepare_visual_objects,
    proposal_digest,
    record_visual_review,
    validate_links,
    visual_issues,
    visual_status,
)
from agentic_extractor.workflow import _verify_visual_round


def proposal(**kwargs):
    return VisualObject.model_validate(
        {
            "id": "new",
            "page": 1,
            "kind": "text",
            "bbox": [0.1, 0.4, 0.8, 0.5],
            "content": "Total: $42.00",
            "reading_order": 2,
            **kwargs,
        }
    )


def page():
    result = PageParse(
        page=1,
        width=100,
        height=100,
        blocks=[
            Block(id="p1-b1", page=1, text="Invoice", bbox=[0, 0, 0.4, 0.2]),
        ],
    )
    result.chunks = build_layout_chunks(result.blocks)
    return result


@pytest.mark.parametrize(
    "bbox", [[0, 0, 0, 1], [1, 0, 0, 1], [-1, 0, 1, 1], [0, 0, float("nan"), 1], [0, 1, 1]]
)
def test_visual_boxes_fail_closed(bbox):
    with pytest.raises(ValidationError):
        proposal(bbox=bbox)


def test_stable_ids_deduplicate_without_trusting_model_identity():
    first = prepare_visual_objects([proposal(id="a"), proposal(id="b")])
    assert len(first) == 1
    assert first == prepare_visual_objects(first)
    assert first[0].id.startswith("visual-p1-")


def test_pending_content_is_not_published_and_raw_ocr_never_changes():
    source = page()
    original = copy.deepcopy(source.blocks)
    item = proposal()
    assert apply_visual_objects([source], [item], [])
    assert "$42" not in source.markdown
    audits = record_visual_review(item, [], "approve", "Read the source crop")
    assert not apply_visual_objects([source], [item], audits)
    assert "Total: $42.00" in source.markdown
    apply_visual_objects([source], [item], audits)
    assert source.markdown.count("Total: $42.00") == 1
    assert source.blocks == original


def test_approval_cannot_be_replayed_for_changed_proposal():
    item = proposal()
    audits = record_visual_review(item, [], "approve", "Source checked")
    assert visual_status(proposal(content="Total: $420.00"), audits)[0] == "pending"
    assert audits[0]["digest"] == proposal_digest(item)


def test_visual_insertion_preserves_gaps_in_semantic_reading_order():
    from agentic_extractor.parse import ParseChunk

    source = page()
    source.chunks = [
        ParseChunk(
            id="heading",
            page=1,
            type="text",
            reading_order=1,
            text="Before",
            markdown="Before",
            source_block_ids=[],
            bbox=None,
            raw_scores=[],
        ),
        ParseChunk(
            id="paragraph",
            page=1,
            type="text",
            reading_order=2,
            text="After",
            markdown="After",
            source_block_ids=[],
            bbox=None,
            raw_scores=[],
        ),
    ]
    item = proposal(reading_order=2)
    apply_visual_objects(
        [source],
        [item],
        record_visual_review(item, [], "approve", "source"),
        {(1, "heading"): 1, (1, "paragraph"): 3},
    )
    assert [chunk.id for chunk in source.chunks] == ["heading", "new", "paragraph"]


def test_human_correction_rejection_and_audit_history():
    source, item = page(), proposal()
    with pytest.raises(ValueError):
        record_visual_review(item, [], "approve", " ")
    with pytest.raises(ValueError):
        record_visual_review(item, [], "correct", "source", "")
    with pytest.raises(ValueError):
        record_visual_review(item, [], "other", "source")
    audits = record_visual_review(item, [], "correct", "source", "Total: $41.00")
    apply_visual_objects([source], [item], audits)
    assert "$41.00" in source.markdown and "$42.00" not in source.markdown
    audits = record_visual_review(item, audits, "reject", "Wrong region")
    apply_visual_objects([source], [item], audits)
    assert "$41.00" not in source.markdown
    assert len(audits) == 2


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("figure", "Figure description (not a transcription)"),
        ("chart", "Chart description (not a transcription)"),
        ("equation", "$$\nx^2\n$$"),
    ],
)
def test_rich_content_is_explicitly_typed(kind, expected):
    source, item = page(), proposal(kind=kind, content="x^2")
    apply_visual_objects([source], [item], record_visual_review(item, [], "approve", "source"))
    assert expected in source.markdown


def test_unknown_source_ids_cannot_be_approved():
    item = proposal(source_block_ids=["unknown"])
    assert visual_issues([page()], [item], record_visual_review(item, [], "approve", "checked"))


def test_document_links_check_order_references_and_cycles():
    source = page()
    item = proposal()
    apply_visual_objects([source], [item], record_visual_review(item, [], "approve", "source"))
    first = source.chunks[0].id
    good = DocumentLink(source_id=first, target_id="new", relation="continues")
    assert not validate_links([source], [good])
    assert validate_links([source], [good, good])
    assert validate_links(
        [source], [DocumentLink(source_id="new", target_id=first, relation="continues")]
    )
    assert validate_links(
        [source], [good, DocumentLink(source_id="new", target_id=first, relation="parent_of")]
    )
    assert validate_links(
        [source], [DocumentLink(source_id="missing", target_id=first, relation="caption_of")]
    )


def test_crop_verification_requires_independent_exact_text_and_deduplicates():
    from agentic_extractor.models import UsageRecord
    from agentic_extractor.openai_refiner import CloudResult

    source, item = page(), proposal()
    local = SimpleNamespace(pages=[source], usage=UsageRecord(), cloud_attempts=[], warnings=[])
    cloud = CloudResult(refined_markdown="", reviewed_pages=[1], visual_objects=[item])
    calls = []

    def verify(pages, objects):
        calls.append(objects)
        return VisualVerification(
            decisions=[
                VisualDecision(
                    id=item.id,
                    supported=True,
                    transcription="Total: $420.00",
                    reason="pixels",
                )
            ]
        ), UsageRecord(call_count=1)

    inspected = set()
    refiner = SimpleNamespace(verify_visual_objects=verify)
    _verify_visual_round(local, cloud, refiner, inspected, 1)
    _verify_visual_round(local, cloud, refiner, inspected, 2)
    assert len(calls) == 1
    assert visual_status(item, source.visual_audits)[0] == "pending"


def test_budget_reserves_cache_write_maximum_and_stops_on_unknown():
    budget = RequestBudget(0.25)
    entry = budget.reserve(1000, 10000)
    assert entry["reserved_usd"] == "0.1025"
    budget.settle(entry, SimpleNamespace(total_cost_usd=None, cost_status="unavailable"))
    with pytest.raises(RuntimeError, match="unknown"):
        budget.reserve(1, 1)


def test_budget_persists_reservation_before_generation(tmp_path):
    import json

    budget = RequestBudget()
    budget.ledger_path = tmp_path / "billing.json"
    entry = budget.reserve(100, 1000)
    pending = json.loads(budget.ledger_path.read_text())
    assert pending["reservations"][0]["status"] == "pending"
    budget.settle(entry, SimpleNamespace(total_cost_usd=None, cost_status="unavailable"))
    assert json.loads(budget.ledger_path.read_text())["billing_unknown"] is True


def test_inspection_pdf_render_is_bounded_and_raw_evidence_is_preserved():
    import io

    from PIL import Image

    from agentic_extractor.rich_document import prepare_inspection_images

    output = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(output, "PDF", resolution=150)
    source = page()
    before = copy.deepcopy(source.blocks)
    prepare_inspection_images([source], output.getvalue(), [proposal()])
    assert source.inspection_image_bytes is not None
    image = Image.open(io.BytesIO(source.inspection_image_bytes))
    assert image.width * image.height <= 25_000_000
    assert source.blocks == before
    prepare_inspection_images([source], b"%PDF-broken", [proposal()])
    assert source.warnings


def test_budget_cannot_exceed_authority_and_does_not_release_estimates():
    with pytest.raises(ValueError):
        RequestBudget(5.01)
    budget = RequestBudget(0.11)
    entry = budget.reserve(1000, 10000)
    budget.settle(entry, SimpleNamespace(total_cost_usd=0.001, cost_status="estimate"))
    with pytest.raises(RuntimeError, match="exceeds"):
        budget.reserve(1000, 10000)
    budget.settle(entry, SimpleNamespace(total_cost_usd=0.001, cost_status="exact"))
    budget.reserve(1000, 10000)
