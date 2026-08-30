from agentic_extractor.models import Block, ProcessingMode
from agentic_extractor.parse import PageParse, routing_reasons, should_send_image


def block(index: int, score: float) -> Block:
    return Block(
        id=f"p1-b{index}",
        page=1,
        text="text",
        ocr_score=score,
        polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
        bbox=[0, 0, 0.1, 0.1],
    )


def test_balanced_routes_low_mean_confidence() -> None:
    page = PageParse(page=1, width=100, height=100, blocks=[block(1, 0.7)])
    assert should_send_image(page, ProcessingMode.BALANCED, set())


def test_high_accuracy_routes_every_selected_page() -> None:
    page = PageParse(page=1, width=100, height=100, blocks=[block(1, 0.99)])
    assert should_send_image(page, ProcessingMode.HIGH_ACCURACY, set())


def test_forced_page_is_routed() -> None:
    page = PageParse(page=1, width=100, height=100, blocks=[block(1, 0.99)])
    assert should_send_image(page, ProcessingMode.BALANCED, {1})


def test_balanced_escalates_complex_or_ambiguous_pages() -> None:
    page = PageParse(
        page=1,
        width=100,
        height=100,
        blocks=[block(1, 0.99)],
        layout_signals={"ambiguous": True, "columns_detected": 2},
    )
    reasons = routing_reasons(page, ProcessingMode.BALANCED, set())
    assert "ambiguous_reading_order" in reasons
    assert "multi_column" in reasons
