from agentic_extractor.oracle_eval import compare_parse_results


def _parse(markdown: str, node_type: str = "text") -> dict:
    return {
        "markdown": markdown,
        "metadata": {},
        "structure": {
            "type": "document",
            "children": [
                {
                    "type": "page",
                    "grounding": {
                        "page": 1,
                        "range": {"start": 0, "end": len(markdown)},
                        "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
                    },
                    "children": [{"type": node_type}],
                }
            ],
        },
    }


def test_oracle_comparison_reports_counts_without_document_text() -> None:
    report = compare_parse_results(_parse("[x] yes\n<table>"), _parse("[ ] no", "marginalia"))

    assert report["markdown"]["checkbox_count_oracle"] == 1
    assert report["markdown"]["html_table_count_candidate"] == 0
    assert report["structure"]["type_order_exact"] is False
    assert report["markdown"]["normalized_sequence_similarity"] < 1.0
    assert "yes" not in str(report)


def test_oracle_comparison_normalizes_markdown_formatting_but_preserves_word_order() -> None:
    report = compare_parse_results(_parse("# Hello, WORLD!"), _parse("hello world"))

    assert report["markdown"]["normalized_sequence_similarity"] == 1.0


def test_oracle_comparison_rejects_out_of_bounds_grounding() -> None:
    candidate = _parse("short")
    candidate["structure"]["children"][0]["grounding"]["range"]["end"] = 99

    assert compare_parse_results(_parse("short"), candidate)["structure"]["ranges_valid"] is False


def test_oracle_comparison_can_limit_a_full_oracle_to_selected_pages() -> None:
    oracle = _parse("page one [ ]\npage two [x]")
    first = oracle["structure"]["children"][0]
    first["grounding"]["range"] = {"start": 0, "end": 12}
    second = {
        **first,
        "grounding": {
            **first["grounding"],
            "page": 2,
            "range": {"start": 13, "end": len(oracle["markdown"])},
        },
    }
    oracle["structure"]["children"].append(second)

    report = compare_parse_results(oracle, oracle, pages={2})

    assert report["selected_pages"] == [2]
    assert report["markdown"]["checkbox_count_oracle"] == 1
    assert report["structure"]["node_types_oracle"]["page"] == 1
    assert report["pages"]["2"]["oracle"]["checkbox_count"] == 1
    assert report["pages"]["2"]["candidate"]["characters"] == 12


def test_oracle_comparison_matches_special_objects_once_by_page_type_and_geometry() -> None:
    oracle = _parse("brand")
    candidate = _parse("brand")
    for document, boxes in (
        (oracle, [(0.1, 0.1, 0.3, 0.2), (0.5, 0.1, 0.7, 0.2)]),
        (candidate, [(0.11, 0.1, 0.31, 0.2)]),
    ):
        children = document["structure"]["children"][0]["children"]
        children.clear()
        for xmin, ymin, xmax, ymax in boxes:
            children.append(
                {
                    "type": "logo",
                    "grounding": {
                        "page": 1,
                        "range": {"start": 0, "end": 5},
                        "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
                    },
                }
            )

    metrics = compare_parse_results(oracle, candidate)["objects"]["logo"]

    assert metrics == {
        "oracle": 2,
        "candidate": 1,
        "matched": 1,
        "false_positive": 0,
        "false_negative": 1,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.6667,
    }


def test_oracle_comparison_reports_checkbox_states_without_labels() -> None:
    report = compare_parse_results(_parse("[x] yes\n[ ] no"), _parse("[ ] yes\n[ ] no"))

    assert report["checkbox_states"]["oracle"] == {"x": 1, " ": 1}
    assert report["checkbox_states"]["candidate"] == {" ": 2}
