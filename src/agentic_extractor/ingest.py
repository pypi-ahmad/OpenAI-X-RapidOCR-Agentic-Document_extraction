"""Bounded, format-aware in-memory document ingestion.

Responsible for turning an untrusted upload (arbitrary bytes plus a
caller-supplied file name) into decoded, size/page/pixel-bounded pages ready
for OCR. Must not trust the file name or extension for format decisions, must
not exceed the configured upload/page/pixel limits, and must not let a
malformed or hostile file escape as an unhandled exception — every rejection
path should surface as `IngestError`. Next: `ocr.py`, which OCRs the pages
this module produces."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import pypdfium2 as pdfium
from PIL import Image, ImageOps, UnidentifiedImageError

from agentic_extractor.cache import RENDER_CACHE, page_image_hash
from agentic_extractor.config import SETTINGS, Settings

RENDER_CACHE_VERSION = 1


class IngestError(ValueError):
    pass


@dataclass(slots=True)
class InputPage:
    number: int
    image: Image.Image
    source_dpi: tuple[float, float] | None = None
    source_format: str | None = None
    orientation_correction_degrees: int = 0
    render_cache_hit: bool = False
    page_hash: str = ""


@dataclass(slots=True)
class IngestedDocument:
    file_name: str
    mime_type: str
    original_bytes: bytes
    pages: list[InputPage]
    document_sha256: str = ""
    render_cache_hits: int = 0
    render_cache_misses: int = 0

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
    document_sha256 = hashlib.sha256(data).hexdigest()
    # Format is decided from the file's own byte signature, never from `file_name`'s
    # extension: the upload is untrusted and a renamed file must not bypass validation.
    if data.startswith(b"%PDF-"):
        pages = _load_pdf(data, settings, document_sha256)
        mime = "application/pdf"
    else:
        pages, mime = _load_image(data, settings, document_sha256)
    if not pages or len(pages) > settings.max_pages:
        raise IngestError(f"Document must contain 1–{settings.max_pages} pages.")
    return IngestedDocument(
        file_name=file_name,
        mime_type=mime,
        original_bytes=data,
        pages=pages,
        document_sha256=document_sha256,
        render_cache_hits=sum(page.render_cache_hit for page in pages),
        render_cache_misses=sum(not page.render_cache_hit for page in pages),
    )


def _check_pixels(image: Image.Image, settings: Settings) -> None:
    if image.width * image.height > settings.max_image_pixels:
        raise IngestError(f"Page exceeds {settings.max_image_pixels:,} decoded pixels.")


def _load_pdf(data: bytes, settings: Settings, document_sha256: str) -> list[InputPage]:
    try:
        document = pdfium.PdfDocument(data)
        if len(document) > settings.max_pages:
            raise IngestError(f"PDF exceeds the {settings.max_pages}-page limit.")
        scale = settings.render_dpi / 72
        pages = []
        for index in range(len(document)):
            key = _render_cache_key(document_sha256, index + 1, f"pdf-{settings.render_dpi}")
            cached = _get_cached_render(key)
            image = (
                _decode_cached_image(cached)
                if cached is not None
                else document[index].render(scale=scale).to_pil().convert("RGB")
            )
            _check_pixels(image, settings)
            if cached is None:
                _put_cached_render(key, image)
            pages.append(
                InputPage(
                    index + 1,
                    image,
                    (settings.render_dpi, settings.render_dpi),
                    "PDF",
                    render_cache_hit=cached is not None,
                    page_hash=page_image_hash(image),
                )
            )
        return pages
    except IngestError:
        raise
    # Deliberately broad: pdfium can raise many underlying error types for a
    # corrupt/encrypted file. All of them are remapped to one actionable
    # IngestError rather than leaking a library-specific exception to callers.
    except Exception as exc:
        raise IngestError("PDF is encrypted, malformed, or cannot be decoded.") from exc


def _load_image(
    data: bytes, settings: Settings, document_sha256: str
) -> tuple[list[InputPage], str]:
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
        key = _render_cache_key(document_sha256, 1, "normalized-image")
        cached = _get_cached_render(key)
        image = (
            _decode_cached_image(cached)
            if cached is not None
            else ImageOps.exif_transpose(source).convert("RGB")
        )
        _check_pixels(image, settings)
        if cached is None:
            _put_cached_render(key, image)
        correction = {3: 180, 6: 90, 8: -90}.get(orientation, 0)
        mime = {"PNG": "image/png", "JPEG": "image/jpeg", "TIFF": "image/tiff"}[source.format]
        return [
            InputPage(
                1,
                image,
                dpi,
                source.format,
                correction,
                render_cache_hit=cached is not None,
                page_hash=page_image_hash(image),
            )
        ], mime
    except IngestError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise IngestError("File signature or image data is unsupported or malformed.") from exc


def _render_cache_key(document_sha256: str, page: int, variant: str) -> str:
    source = hashlib.sha256(f"{document_sha256}:{page}:{variant}".encode()).hexdigest()
    return f"render:{RENDER_CACHE_VERSION}:source:{source}"


def _render_content_cache_key(page_hash: str) -> str:
    return f"render:{RENDER_CACHE_VERSION}:page:{page_hash}"


# Rendering is cached through a level of indirection: a "source" key (document
# hash + page + render variant) points at a "content" key (hash of the actual
# decoded pixels). Two different source documents that happen to render to the
# same pixels share one stored image instead of duplicating it, and the source
# key still resolves correctly even though it only points at content.


def _get_cached_render(source_key: str) -> bytes | None:
    content_key = RENDER_CACHE.get(source_key)
    if content_key is None:
        return None
    try:
        key = content_key.decode("ascii")
    except UnicodeDecodeError:
        return None
    return RENDER_CACHE.get(key)


def _put_cached_render(source_key: str, image: Image.Image) -> None:
    content_key = _render_content_cache_key(page_image_hash(image))
    if RENDER_CACHE.get(content_key) is None:
        RENDER_CACHE.put(content_key, _encode_cached_image(image))
    RENDER_CACHE.put(source_key, content_key.encode("ascii"))


def _encode_cached_image(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _decode_cached_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")
