import io

import pytest
from PIL import Image

from agentic_extractor.ingest import IngestError, load_document, validate_page_range


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (20, 10), "white").save(output, "PNG")
    return output.getvalue()


def image_bytes(format_name: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (20, 10), "white").save(output, format_name)
    return output.getvalue()


def multi_frame_tiff_bytes() -> bytes:
    output = io.BytesIO()
    first = Image.new("RGB", (20, 10), "white")
    second = Image.new("RGB", (20, 10), "black")
    first.save(output, "TIFF", save_all=True, append_images=[second])
    return output.getvalue()


def test_ingest_uses_content_not_extension() -> None:
    document = load_document("misleading.pdf", png_bytes())
    assert document.mime_type == "image/png"
    assert len(document.pages) == 1


def test_ingest_rejects_unknown_content() -> None:
    with pytest.raises(IngestError, match="unsupported|malformed"):
        load_document("bad.png", b"not an image")


def test_exif_orientation_is_normalized_with_provenance() -> None:
    image = Image.new("RGB", (40, 20), "white")
    exif = Image.Exif()
    exif[274] = 6
    output = io.BytesIO()
    image.save(output, "JPEG", exif=exif)

    document = load_document("rotated.jpg", output.getvalue())

    assert document.pages[0].image.size == (20, 40)
    assert document.pages[0].orientation_correction_degrees == 90


@pytest.mark.parametrize(
    ("file_name", "format_name", "mime_type"),
    [
        ("scan.png", "PNG", "image/png"),
        ("photo.jpg", "JPEG", "image/jpeg"),
        ("multipage.tiff", "TIFF", "image/tiff"),
        ("scan.pdf", "PDF", "application/pdf"),
    ],
)
def test_v1_supported_scans_and_images_are_accepted(
    file_name: str, format_name: str, mime_type: str
) -> None:
    document = load_document(file_name, image_bytes(format_name))
    assert document.mime_type == mime_type
    assert document.page_count == 1


def test_multi_frame_image_requires_pdf_conversion() -> None:
    with pytest.raises(IngestError, match="Multi-frame images.*convert.*PDF"):
        load_document("multipage.tiff", multi_frame_tiff_bytes())


@pytest.mark.parametrize(
    ("start", "end", "total"),
    [(0, 1, 2), (2, 1, 2), (1, 3, 2)],
)
def test_page_range_rejects_values_outside_inclusive_bounds(
    start: int, end: int, total: int
) -> None:
    with pytest.raises(ValueError, match="1 <= start_page"):
        validate_page_range(start, end, total)


def test_page_range_accepts_full_document_and_single_page() -> None:
    assert validate_page_range(1, 7, 7) == range(1, 8)
    assert validate_page_range(1, 1, 1) == range(1, 2)
