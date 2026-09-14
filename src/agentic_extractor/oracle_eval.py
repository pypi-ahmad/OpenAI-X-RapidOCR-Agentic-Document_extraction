"""Text-free metrics for comparing local Parse artifacts with an approved oracle.

Responsible for: computing text-free, privacy-safe comparison metrics (token
similarity, node type counts, table cell counts, bounding box IoU) between
candidate Parse output and an approved ground-truth oracle.

Must not: copy raw confidential document text into evaluation metric reports.

Next: `tools/run_real_validation.py` and `tools/run_corpus_validation.py`,
which use this module for automated benchmarking.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_CHECKBOX = re.compile(r"\[(?P<state>x|X| |-|~|\?)\]")
_OBJECT_TYPES = {"logo", "attestation", "figure", "scan_code", "table"}
_NORMALIZED_TOKEN = re.compile(r"[a-z0-9]+")


def compare_parse_results(
    oracle: dict[str, Any],
    candidate: dict[str, Any],
    *,
    pages: set[int] | None = None,
) -> dict[str, Any]:
    """Return aggregate accuracy metrics without copying document text into the report."""
    oracle_markdown = _selected_markdown(oracle, pages)
    candidate_markdown = _selected_markdown(candidate, pages)
    oracle_nodes = list(_selected_nodes(oracle, pages))
    candidate_nodes = list(_selected_nodes(candidate, pages))
    oracle_types = [str(node.get("type", "")) for node in oracle_nodes]
    candidate_types = [str(node.get("type", "")) for node in candidate_nodes]
    page_numbers = sorted(
        {
            int(node["grounding"]["page"])
            for document in (oracle, candidate)
            for node in _selected_page_nodes(document, pages)
            if isinstance(node.get("grounding", {}).get("page"), int)
        }
    )
    return {
        "schema_shape_match": set(candidate) == {"markdown", "metadata", "structure"},
        "selected_pages": sorted(pages) if pages else None,
        "markdown": {
            "oracle_characters": len(oracle_markdown),
            "candidate_characters": len(candidate_markdown),
            "normalized_sequence_similarity": _normalized_sequence_similarity(
                oracle_markdown, candidate_markdown
            ),
            "checkbox_count_oracle": len(_CHECKBOX.findall(oracle_markdown)),
            "checkbox_count_candidate": len(_CHECKBOX.findall(candidate_markdown)),
            "html_table_count_oracle": oracle_markdown.lower().count("<table"),
            "html_table_count_candidate": candidate_markdown.lower().count("<table"),
        },
        "structure": {
            "node_types_oracle": dict(Counter(oracle_types)),
            "node_types_candidate": dict(Counter(candidate_types)),
            "type_order_exact": oracle_types == candidate_types,
            "table_cell_count_oracle": oracle_types.count("table_cell"),
            "table_cell_count_candidate": candidate_types.count("table_cell"),
            "ranges_valid": _ranges_valid(candidate, pages=pages),
        },
        "objects": _object_metrics(oracle, candidate, pages),
        "checkbox_states": {
            "oracle": dict(
                Counter(match.group("state") for match in _CHECKBOX.finditer(oracle_markdown))
            ),
            "candidate": dict(
                Counter(match.group("state") for match in _CHECKBOX.finditer(candidate_markdown))
            ),
        },
        "pages": {
            str(page): {
                "oracle": _page_metrics(oracle, page),
                "candidate": _page_metrics(candidate, page),
            }
            for page in page_numbers
        },
    }


def _object_metrics(
    oracle: dict[str, Any], candidate: dict[str, Any], pages: set[int] | None
) -> dict[str, Any]:
    oracle_objects = _objects_by_type_and_page(oracle, pages)
    candidate_objects = _objects_by_type_and_page(candidate, pages)
    metrics: dict[str, Any] = {}
    for node_type in sorted(_OBJECT_TYPES):
        matched = 0
        oracle_count = 0
        candidate_count = 0
        for page in sorted(
            {page for kind, page in {*oracle_objects, *candidate_objects} if kind == node_type}
        ):
            oracle_boxes = oracle_objects.get((node_type, page), [])
            candidate_boxes = candidate_objects.get((node_type, page), [])
            oracle_count += len(oracle_boxes)
            candidate_count += len(candidate_boxes)
            matched += _greedy_box_matches(oracle_boxes, candidate_boxes)
        precision = matched / candidate_count if candidate_count else float(oracle_count == 0)
        recall = matched / oracle_count if oracle_count else float(candidate_count == 0)
        metrics[node_type] = {
            "oracle": oracle_count,
            "candidate": candidate_count,
            "matched": matched,
            "false_positive": candidate_count - matched,
            "false_negative": oracle_count - matched,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4)
            if precision + recall
            else 0.0,
        }
    return metrics


def _objects_by_type_and_page(
    document: dict[str, Any], pages: set[int] | None
) -> dict[tuple[str, int], list[tuple[float, float, float, float]]]:
    objects: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
    for node in _selected_nodes(document, pages):
        node_type = str(node.get("type", ""))
        grounding = node.get("grounding")
        if node_type not in _OBJECT_TYPES or not isinstance(grounding, dict):
            continue
        page = grounding.get("page")
        box = grounding.get("box")
        if not isinstance(page, int) or not isinstance(box, dict):
            continue
        values = tuple(box.get(key) for key in ("xmin", "ymin", "xmax", "ymax"))
        if not all(isinstance(value, (int, float)) for value in values):
            continue
        typed_box: tuple[float, float, float, float] = (
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
        )
        if typed_box[2] <= typed_box[0] or typed_box[3] <= typed_box[1]:
            continue
        objects.setdefault((node_type, page), []).append(typed_box)
    return objects


def _greedy_box_matches(
    oracle_boxes: list[tuple[float, float, float, float]],
    candidate_boxes: list[tuple[float, float, float, float]],
    *,
    minimum_iou: float = 0.25,
) -> int:
    pairs = sorted(
        (
            (_box_iou(oracle_box, candidate_box), oracle_index, candidate_index)
            for oracle_index, oracle_box in enumerate(oracle_boxes)
            for candidate_index, candidate_box in enumerate(candidate_boxes)
        ),
        reverse=True,
    )
    used_oracle: set[int] = set()
    used_candidate: set[int] = set()
    for iou, oracle_index, candidate_index in pairs:
        if iou < minimum_iou:
            break
        if oracle_index in used_oracle or candidate_index in used_candidate:
            continue
        used_oracle.add(oracle_index)
        used_candidate.add(candidate_index)
    return len(used_oracle)


def _box_iou(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _walk(node: Any):
    if not isinstance(node, dict):
        return
    yield node
    for child in node.get("children", []):
        yield from _walk(child)


def _selected_page_nodes(document: dict[str, Any], pages: set[int] | None) -> list[dict[str, Any]]:
    children = document.get("structure", {}).get("children", [])
    if not isinstance(children, list):
        return []
    page_nodes = [node for node in children if isinstance(node, dict)]
    if pages is None:
        return page_nodes
    return [
        node
        for node in page_nodes
        if isinstance(node.get("grounding"), dict) and node["grounding"].get("page") in pages
    ]


def _selected_nodes(document: dict[str, Any], pages: set[int] | None):
    structure = document.get("structure", {})
    if isinstance(structure, dict):
        yield structure
    for page_node in _selected_page_nodes(document, pages):
        yield from _walk(page_node)


def _selected_markdown(document: dict[str, Any], pages: set[int] | None) -> str:
    markdown = str(document.get("markdown", ""))
    if pages is None:
        return markdown
    parts: list[str] = []
    for page_node in _selected_page_nodes(document, pages):
        grounding = page_node.get("grounding", {})
        span = grounding.get("range", {}) if isinstance(grounding, dict) else {}
        start, end = span.get("start"), span.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(markdown):
            parts.append(markdown[start:end])
    return "\n\n".join(parts)


def _page_metrics(document: dict[str, Any], page: int) -> dict[str, Any]:
    page_nodes = _selected_page_nodes(document, {page})
    if not page_nodes:
        return {
            "characters": 0,
            "checkbox_count": 0,
            "html_table_count": 0,
            "node_types": {},
            "table_cell_count": 0,
        }
    markdown = _selected_markdown(document, {page})
    nodes = list(_walk(page_nodes[0]))
    node_types = [str(node.get("type", "")) for node in nodes]
    return {
        "characters": len(markdown),
        "checkbox_count": len(_CHECKBOX.findall(markdown)),
        "html_table_count": markdown.lower().count("<table"),
        "node_types": dict(Counter(node_types)),
        "table_cell_count": node_types.count("table_cell"),
    }


def _normalized_sequence_similarity(oracle: str, candidate: str) -> float:
    """Compare normalized word order without including document text in the report."""
    oracle_tokens = _NORMALIZED_TOKEN.findall(oracle.lower())
    candidate_tokens = _NORMALIZED_TOKEN.findall(candidate.lower())
    return round(SequenceMatcher(None, oracle_tokens, candidate_tokens).ratio(), 6)


def _ranges_valid(candidate: dict[str, Any], *, pages: set[int] | None = None) -> bool:
    markdown = str(candidate.get("markdown", ""))
    for node in _selected_nodes(candidate, pages):
        grounding = node.get("grounding")
        if not isinstance(grounding, dict):
            continue
        span = grounding.get("range")
        if not isinstance(span, dict):
            return False
        start, end = span.get("start"), span.get("end")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start <= end <= len(markdown)
        ):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a local Parse result to an approved oracle"
    )
    parser.add_argument("oracle", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pages", type=int, nargs="+")
    args = parser.parse_args()
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = json.dumps(
        compare_parse_results(oracle, candidate, pages=set(args.pages) if args.pages else None),
        indent=2,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
