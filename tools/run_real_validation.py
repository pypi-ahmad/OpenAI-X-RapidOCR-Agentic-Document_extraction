"""Run the real dual-engine Parse pipeline and compare it with a page-scoped oracle.

Responsible for: executing the live Parse pipeline (RapidOCR, PP-DocLayoutV3,
OpenAI Sol) against a real source PDF, writing generated artifacts, and
comparing output against a page-scoped oracle JSON.

Must not: run without valid OpenAI credentials or fail to record comparison metrics.

Next: `oracle_eval.py` for metric calculation logic.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentic_extractor.artifacts import build_local_artifacts
from agentic_extractor.layout import create_pp_doclayout_engine
from agentic_extractor.models import Capability, DocumentRequest, ProcessingMode
from agentic_extractor.ocr import create_rapidocr_engine
from agentic_extractor.openai_refiner import OpenAIRefiner
from agentic_extractor.oracle_eval import compare_parse_results
from agentic_extractor.workflow import run_agent_workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pages", type=int, nargs="+", required=True)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ProcessingMode],
        default=ProcessingMode.BALANCED.value,
    )
    return parser


def _count_page_objects(parse_result: dict[str, Any]) -> list[dict[str, int]]:
    summaries: list[dict[str, int]] = []
    for page in parse_result.get("structure", {}).get("children", []):
        grounding = page.get("grounding", {})
        children = page.get("children", [])
        summaries.append(
            {
                "page": int(grounding.get("page", 0)),
                "tables": sum(child.get("type") == "table" for child in children),
                "table_cells": sum(
                    len(child.get("children", []))
                    for child in children
                    if child.get("type") == "table"
                ),
            }
        )
    return summaries


def main() -> int:
    args = _parser().parse_args()
    pages = set(args.pages)
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if not args.oracle.is_file():
        raise FileNotFoundError(args.oracle)
    args.output.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    ocr_started = time.perf_counter()
    ocr = create_rapidocr_engine()
    ocr_initialization = time.perf_counter() - ocr_started
    layout_started = time.perf_counter()
    layout = create_pp_doclayout_engine()
    layout_initialization = time.perf_counter() - layout_started
    try:
        result, workflow = run_agent_workflow(
            DocumentRequest(
                file_name=args.source.name,
                file_bytes=args.source.read_bytes(),
                mode=ProcessingMode(args.mode),
                selected_pages=pages,
                capabilities={Capability.PARSE},
            ),
            ocr_resource=ocr,
            layout_resource=layout,
            refiner=OpenAIRefiner(),
            initialization_timings={
                "rapidocr_initialization_seconds": ocr_initialization,
                "layout_initialization_seconds": layout_initialization,
            },
        )
        artifacts = build_local_artifacts(result)
        parse_result = json.loads(artifacts.parse_result)
        oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
        comparison = compare_parse_results(oracle, parse_result, pages=pages)
        summary = {
            "elapsed_seconds": time.perf_counter() - started,
            "selected_pages": sorted(pages),
            "requested_mode": result.requested_mode.value,
            "effective_mode": result.effective_mode.value,
            "engine": asdict(result.engine),
            "layout_engine": asdict(result.layout_engine) if result.layout_engine else None,
            "gpt_calls": result.usage.call_count,
            "input_tokens": result.usage.input_tokens,
            "cached_input_tokens": result.usage.cached_input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_cost_usd": result.usage.total_cost_usd,
            "failed_pages": result.failed_pages,
            "page_objects": _count_page_objects(parse_result),
            "timings": result.timings,
            "warnings": result.warnings,
            "workflow_state": workflow.current_state.value,
        }
        args.output.joinpath("document.md").write_bytes(artifacts.markdown)
        args.output.joinpath("parse-result.json").write_bytes(artifacts.parse_result)
        args.output.joinpath("manifest.json").write_text(
            json.dumps(artifacts.manifest, indent=2) + "\n", encoding="utf-8"
        )
        args.output.joinpath("landingai-comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        args.output.joinpath("run-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"summary": summary, "comparison": comparison}, indent=2))
    finally:
        close = getattr(layout.client, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
