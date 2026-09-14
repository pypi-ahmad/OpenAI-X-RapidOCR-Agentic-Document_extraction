from agentic_extractor.timing import stage_timing_summary


def test_stage_timing_summary_identifies_slowest_exclusive_stage() -> None:
    summary = stage_timing_summary(
        {
            "total_seconds": 20.0,
            "layout_seconds": 8.0,  # Aggregate: intentionally excluded from stage ranking.
            "document_ingest_seconds": 1.0,
            "ocr_wall_seconds": 3.0,
            "layout_detection_seconds": 2.0,
            "table_structure_seconds": 4.0,
            "gpt_refinement_seconds": 10.0,
        }
    )

    assert summary["bottleneck"] == {
        "stage": "GPT parse refinement",
        "seconds": 10.0,
        "share_percent": 50.0,
    }
    assert summary["measured_stage_seconds"] == 20.0
    assert [row["stage"] for row in summary["stages"]][:2] == [
        "GPT parse refinement",
        "Table structure",
    ]


def test_stage_timing_summary_ignores_missing_and_invalid_values() -> None:
    summary = stage_timing_summary(
        {
            "ocr_wall_seconds": 0.0,
            "gpt_refinement_seconds": float("nan"),
            "unknown_seconds": 99.0,
        }
    )

    assert summary["bottleneck"] is None
    assert summary["stages"] == []
    assert summary["measured_stage_seconds"] == 0.0
