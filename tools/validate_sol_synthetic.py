"""Explicitly opt-in synthetic live validation with one cumulative $5 ceiling.

Run only with user authorization. Uses real OCR/layout and only synthetic source pixels.
The output directory must be new, preventing accidental overwrite of the cost ledger.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from agentic_extractor.artifacts import build_local_artifacts
from agentic_extractor.budget import RequestBudget
from agentic_extractor.costs import aggregate_usage
from agentic_extractor.layout import create_pp_doclayout_engine
from agentic_extractor.models import Capability, DocumentRequest
from agentic_extractor.ocr import create_rapidocr_engine
from agentic_extractor.openai_refiner import CloudResult, OpenAIRefiner
from agentic_extractor.workflow import run_agent_workflow


def fixture() -> bytes:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 36)
    pages = []
    for number in (1, 2):
        image = Image.new("RGB", (1200, 1600), "white")
        draw = ImageDraw.Draw(image)
        draw.text((80, 60), f"SYNTHETIC VALIDATION - PAGE {number}", font=font, fill="black")
        if number == 1:
            draw.text((80, 180), "Invoice: TEST-0042", font=font, fill="black")
            draw.text((650, 180), "Date: 2026-09-23", font=font, fill="black")
            draw.text((80, 260), "Total: $42.00", font=font, fill="black")
            draw.text((650, 260), "Currency: USD", font=font, fill="black")
            draw.text((80, 380), "Items (continued on page 2)", font=font, fill="black")
            rows = [("Item", "Qty", "Amount"), ("Alpha", "2", "$12.00"), ("Beta", "3", "$30.00")]
            for row, values in enumerate(rows):
                for col, value in enumerate(values):
                    x, y = 80 + col * 330, 460 + row * 100
                    draw.rectangle((x, y, x + 330, y + 100), outline="black", width=3)
                    draw.text((x + 15, y + 20), value, font=font, fill="black")
            draw.text((80, 900), "Equation: E = mc²", font=font, fill="black")
            draw.ellipse((130, 1100, 370, 1340), fill="blue")
            draw.text((80, 1400), "Figure 1: A blue circle", font=font, fill="black")
        else:
            draw.text((80, 200), "Items continued: no additional charges", font=font, fill="black")
            draw.text((80, 340), "Chart: Quarterly units", font=font, fill="black")
            draw.line((100, 900, 100, 450), fill="black", width=4)
            draw.line((100, 900, 950, 900), fill="black", width=4)
            for x, label, height in [(200, "Q1: 10", 180), (500, "Q2: 20", 360)]:
                draw.rectangle((x, 900 - height, x + 150, 900), fill="green")
                draw.text((x, 950), label, font=font, fill="black")
            draw.text((80, 1150), "Account: TEST-ACCOUNT-7", font=font, fill="black")
            draw.text((80, 1300), "No personal or confidential data.", font=font, fill="black")
        pages.append(image)
    output = io.BytesIO()
    pages[0].save(output, "PDF", save_all=True, append_images=pages[1:], resolution=150)
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--authorize-up-to-usd", required=True, type=float)
    args = parser.parse_args()
    budget = RequestBudget(args.authorize_up_to_usd)
    args.output.mkdir(parents=True, exist_ok=False)
    budget.ledger_path = args.output / "billing.json"
    source = fixture()
    args.output.joinpath("synthetic.pdf").write_bytes(source)
    refiner = OpenAIRefiner()
    refiner.budget = budget
    layout = None
    summary = {"budget_usd": float(budget.limit), "status": "not_completed"}
    try:
        # Cheap validation and exact-count endpoint availability precede engine initialization.
        refiner.validate_configuration()
        ocr = create_rapidocr_engine()
        layout = create_pp_doclayout_engine()
        result, workflow = run_agent_workflow(
            DocumentRequest(
                file_name="synthetic.pdf", file_bytes=source, capabilities={Capability.PARSE}
            ),
            ocr_resource=ocr,
            layout_resource=layout,
            refiner=refiner,
        )
        artifacts = build_local_artifacts(result)
        args.output.joinpath("document.md").write_bytes(artifacts.markdown)
        args.output.joinpath("parse-result.json").write_bytes(artifacts.parse_result)
        args.output.joinpath("bundle.zip").write_bytes(artifacts.bundle)
        expected = ["TEST-0042", "42.00", "2026-09-23", "Alpha", "Beta", "TEST-ACCOUNT-7"]
        missing = [value for value in expected if value not in result.markdown]
        summary.update(
            status="completed",
            workflow_state=workflow.current_state.value,
            missing_expected_values=missing,
            failed_pages=result.failed_pages,
            visual_objects=len(CloudResult.model_validate(result.cloud_output).visual_objects),
            review_required=workflow.review_required,
        )
        return 0 if not missing and not result.failed_pages else 1
    except Exception as exc:
        # Do not log provider error bodies or credentials from a configurable gateway.
        summary.update(status="blocked", error_type=type(exc).__name__)
        return 1
    finally:
        usage = aggregate_usage(refiner.request_usage)
        summary.update(
            usage=usage.model_dump(mode="json"),
            reservations=budget.reservations,
            billing_unknown=budget.uncertain,
        )
        args.output.joinpath("summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        if layout is not None:
            layout.client.close()


if __name__ == "__main__":
    raise SystemExit(main())
