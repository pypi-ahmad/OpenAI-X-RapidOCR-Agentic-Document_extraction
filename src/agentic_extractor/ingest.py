"""Bounded, format-aware in-memory document ingestion."""

from __future__ import annotations

import io
from dataclasses import dataclass

import pypdfium2 as pdfium
from PIL import Image, ImageOps, UnidentifiedImageError

from agentic_extractor.config import SETTINGS, Settings


class IngestError(ValueError):
    pass


@dataclass(slots=True)
class InputPage:
    number: int
    image: Image.Image
    source_dpi: tuple[float, float] | None = None
    source_format: str | None = None
    orientation_correction_degrees: int = 0


@dataclass(slots=True)
class IngestedDocument:
    file_name: str
    mime_type: str
    original_bytes: bytes
    pages: list[InputPage]

    @property
    def byte_size(self) -> int:
        return len(self.original_bytes)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def validate_page_range(start_page: int, end_page: int, total_pages: int) -> range:
    """Validate and return an inclusive, one-based page range."""
    if not 1 <= start_page <= end_page <= total_pages:
        raise ValueError("Page range must satisfy 1 <= start_page <= end_page <= total_pages.")
    return range(start_page, end_page + 1)


def load_document(file_name: str, data: bytes, settings: Settings = SETTINGS) -> IngestedDocument:
    if not data or len(data) > settings.max_upload_bytes:
        raise IngestError("Document is empty or exceeds the 50 MiB upload limit.")
    if data.startswith(b"%PDF-"):
        pages = _load_pdf(data, settings)
        mime = "application/pdf"
    else:
        pages, mime = _load_image(data, settings)
    if not pages or len(pages) > settings.max_pages:
        raise IngestError(f"Document must contain 1–{settings.max_pages} pages.")
    return IngestedDocument(file_name=file_name, mime_type=mime, original_bytes=data, pages=pages)


def _check_pixels(image: Image.Image, settings: Settings) -> None:
    if image.width * image.height > settings.max_image_pixels:
        raise IngestError(f"Page exceeds {settings.max_image_pixels:,} decoded pixels.")


def _load_pdf(data: bytes, settings: Settings) -> list[InputPage]:
    try:
        document = pdfium.PdfDocument(data)
        if len(document) > settings.max_pages:
            raise IngestError(f"PDF exceeds the {settings.max_pages}-page limit.")
        scale = settings.render_dpi / 72
        pages = []
        for index in range(len(document)):
            image = document[index].render(scale=scale).to_pil().convert("RGB")
            _check_pixels(image, settings)
            pages.append(
                InputPage(index + 1, image, (settings.render_dpi, settings.render_dpi), "PDF")
            )
        return pages
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError("PDF is encrypted, malformed, or cannot be decoded.") from exc


def _load_image(data: bytes, settings: Settings) -> tuple[list[InputPage], str]:
    try:
        source = Image.open(io.BytesIO(data))
        if source.format not in {"PNG", "JPEG", "TIFF"}:
            raise IngestError("Supported formats: PDF, PNG, JPEG, and TIFF.")
        if getattr(source, "n_frames", 1) > 1:
            raise IngestError("Multi-frame images are not supported; convert the file to PDF.")
        source_dpi = source.info.get("dpi")
        dpi = (
            (float(source_dpi[0]), float(source_dpi[1]))
            if isinstance(source_dpi, tuple) and len(source_dpi) >= 2
            else None
        )
        orientation = int(source.getexif().get(274, 1))
        image = ImageOps.exif_transpose(source).convert("RGB")
        _check_pixels(image, settings)
        correction = {3: 180, 6: 90, 8: -90}.get(orientation, 0)
        mime = {"PNG": "image/png", "JPEG": "image/jpeg", "TIFF": "image/tiff"}[source.format]
        return [InputPage(1, image, dpi, source.format, correction)], mime
    except IngestError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise IngestError("File signature or image data is unsupported or malformed.") from exc
