from __future__ import annotations

from PIL import Image

from agentic_extractor.quality import analyze_page, preprocess_if_improved, run_adaptive_batches


def test_quality_preflight_reports_real_page_diagnostics() -> None:
    image = Image.new("RGB", (1200, 1600), "#777777")
    result = analyze_page(3, image, dpi=(150, 150), source_format="JPEG")
    assert result.page == 3
    assert result.pixel_dimensions == [1200, 1600]
    assert result.estimated_dpi == 150
    assert result.low_contrast is True
    assert any("300–400 DPI" in item for item in result.recommendations)


def test_adaptive_batch_halves_on_memory_pressure_with_bounded_retry() -> None:
    attempts: list[int] = []

    def handler(batch: list[int]) -> list[int]:
        attempts.append(len(batch))
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory")
        return batch

    values, audit = run_adaptive_batches([1, 2, 3, 4], handler, initial_batch_size=4)
    assert values == [1, 2, 3, 4]
    assert attempts == [4, 2, 2]
    assert audit.retries == 1
    assert audit.final_batch_size == 2


def test_contrast_preprocessing_is_retained_only_after_measured_improvement() -> None:
    image = Image.linear_gradient("L").resize((300, 300)).point(lambda value: 100 + value // 12)
    diagnostic = analyze_page(1, image)
    processed, actions = preprocess_if_improved(1, image, diagnostic)
    assert actions
    assert analyze_page(1, processed).contrast_score > diagnostic.contrast_score

    clean = Image.new("L", (100, 100), "white")
    unchanged, clean_actions = preprocess_if_improved(1, clean, analyze_page(1, clean))
    assert clean_actions == []
    assert unchanged is clean


def test_small_skew_is_measured_and_corrected_only_when_projection_improves() -> None:
    from PIL import ImageDraw

    source = Image.new("L", (600, 800), "white")
    draw = ImageDraw.Draw(source)
    for y in range(100, 700, 35):
        draw.rectangle((80, y, 520, y + 5), fill="black")
    skewed = source.rotate(3, resample=Image.Resampling.BICUBIC, fillcolor="white")
    diagnostic = analyze_page(1, skewed)

    corrected, actions = preprocess_if_improved(1, skewed, diagnostic)
    corrected_diagnostic = analyze_page(1, corrected)

    assert diagnostic.skew_degrees == 3.0
    assert diagnostic.skew_degrees is not None
    assert diagnostic.layout_strategy == "local_preprocessing_and_geometry_layout"
    assert any("deskew" in action for action in actions)
    assert abs(corrected_diagnostic.skew_degrees or 0) < abs(diagnostic.skew_degrees)
