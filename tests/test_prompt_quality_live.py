"""Opt-in paid prompt checks against GPT-5.6-luna."""

from __future__ import annotations

import io
import json
import os

import pytest
from PIL import Image, ImageDraw

from agentic_extractor.models import Block
from agentic_extractor.openai_refiner import OpenAIRefiner
from agentic_extractor.parse import PageParse

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_PROMPT_EVAL") != "1",
        reason="Set RUN_LIVE_PROMPT_EVAL=1 to authorize paid live prompt evaluation.",
    ),
]


@pytest.fixture(scope="module")
def refiner() -> OpenAIRefiner:
    instance = OpenAIRefiner()
    instance.validate_configuration()
    return instance


def _page_image(lines: list[str]) -> bytes:
    image = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((30, 25 + index * 40), line, fill="black")
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def test_live_grounded_extraction_abstains_from_unsupported_field(
    refiner: OpenAIRefiner,
) -> None:
    text = "Invoice ID: INV-42"
    page = PageParse(
        page=1,
        width=900,
        height=300,
        image_bytes=_page_image([text]),
        blocks=[Block(id="p1-b1", page=1, text=text, bbox=[0.03, 0.08, 0.35, 0.2])],
    )
    schema = {
        "type": "object",
        "properties": {
            "invoice_id": {"type": "string"},
            "account_secret": {"type": "string"},
        },
        "required": ["invoice_id", "account_secret"],
    }

    result, usage = refiner.refine([page], {1}, {"Parse", "Extract"}, [], schema)
    fields = {field.path: field for field in result.extracted_fields}

    assert result.reviewed_pages == [1]
    assert fields["invoice_id"].value == "INV-42"
    assert fields["invoice_id"].status == "verified"
    assert fields["invoice_id"].evidence
    assert fields["account_secret"].status != "verified"
    assert fields["account_secret"].abstention_reason
    print(json.dumps({"case": "grounded_extraction", "usage": usage.model_dump(mode="json")}))


def test_live_checkbox_state_benchmark(refiner: OpenAIRefiner) -> None:
    image = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(image)
    labels = ["Checked", "Empty", "Indeterminate", "Crossed out"]
    expected = ["CHECKED", "UNCHECKED", "INDETERMINATE", "CROSSED_OUT"]
    blocks: list[Block] = []
    for index, label in enumerate(labels):
        top = 30 + index * 60
        draw.rectangle((30, top, 55, top + 25), outline="black", width=2)
        if index == 0:
            draw.line((34, top + 13, 41, top + 21, 52, top + 5), fill="black", width=3)
        elif index == 2:
            draw.line((35, top + 13, 50, top + 13), fill="black", width=3)
        elif index == 3:
            draw.line((25, top - 5, 60, top + 30), fill="black", width=3)
            draw.line((60, top - 5, 25, top + 30), fill="black", width=3)
        draw.text((75, top + 5), label, fill="black")
        blocks.append(
            Block(
                id=f"p1-b{index + 1}",
                page=1,
                text=label,
                bbox=[0.08, top / 300, 0.3, (top + 25) / 300],
            )
        )
    output = io.BytesIO()
    image.save(output, "PNG")
    page = PageParse(page=1, width=900, height=300, image_bytes=output.getvalue(), blocks=blocks)

    result, usage = refiner.refine([page], {1}, {"Parse"}, [], None)
    observed = {item.label.casefold(): item.state for item in result.checkboxes}

    assert len(result.checkboxes) == len(expected)
    assert [observed[label.casefold()] for label in labels] == expected
    print(json.dumps({"case": "checkbox_states", "usage": usage.model_dump(mode="json")}))
