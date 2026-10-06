"""Text extraction from PDF documents."""

from __future__ import annotations

import io
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.ocr.engine import extract_text as recognise_image

logger = get_logger(__name__)

# Below this much embedded text, a page is treated as a scan.
MIN_EMBEDDED_CHARACTERS = 40

PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class PdfPage:
    """One page of a PDF and how its text was obtained."""

    number: int
    text: str
    recognised: bool
    confidence: float


@dataclass(frozen=True)
class PdfExtraction:
    """The text of a PDF, with a record of how each page was handled."""

    text: str
    pages: list[PdfPage]
    page_count: int
    pages_read: int
    pages_recognised: int
    confidence: float
    truncated: bool


def looks_like_pdf(data: bytes) -> bool:
    """Whether the bytes begin with the PDF signature."""
    return data[:5] == PDF_MAGIC


def _render_page(document, index: int, dpi: int) -> bytes:
    """Render one page to PNG bytes at the requested resolution."""
    page = document[index]
    # pypdfium2 works in scale factors against 72 dpi.
    bitmap = page.render(scale=dpi / 72)
    image = bitmap.to_pil()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def extract_pdf(
    data: bytes, language: str | None = None, max_pages: int | None = None
) -> PdfExtraction:
    """Extract the text of a PDF, recognising any page that has no text layer."""
    if not data:
        raise ValidationError("The uploaded PDF is empty.")
    if not looks_like_pdf(data):
        raise ValidationError("The uploaded file is not a PDF.")

    try:
        import pypdf
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - dependencies are declared
        raise ValidationError(f"PDF support is not installed: {exc}") from exc

    settings = get_settings()
    limit = settings.max_pdf_pages if max_pages is None else max_pages
    dpi = settings.pdf_render_dpi

    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        page_count = len(reader.pages)
    except Exception as exc:  # noqa: BLE001 - any parse failure means an unusable file
        raise ValidationError(f"The PDF could not be read: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        raise ValidationError("This PDF is password protected. Remove the password first.")
    if page_count == 0:
        raise ValidationError("This PDF has no pages.")

    pages_to_read = min(page_count, limit) if limit > 0 else page_count
    truncated = pages_to_read < page_count
    if truncated:
        logger.info("Reading %d of %d pages, limited by MAX_PDF_PAGES", pages_to_read, page_count)

    document = None
    pages: list[PdfPage] = []

    try:
        for index in range(pages_to_read):
            embedded = ""
            try:
                embedded = (reader.pages[index].extract_text() or "").strip()
            except Exception as exc:  # noqa: BLE001 - fall through to recognition
                logger.debug("Text layer unreadable on page %d: %s", index + 1, exc)

            if len(embedded) >= MIN_EMBEDDED_CHARACTERS:
                pages.append(
                    PdfPage(number=index + 1, text=embedded, recognised=False, confidence=1.0)
                )
                continue

            # No usable text layer, so this page is a scan.
            if document is None:
                document = pdfium.PdfDocument(io.BytesIO(data))

            try:
                image_bytes = _render_page(document, index, dpi)
                result = recognise_image(image_bytes, language=language)
            except Exception as exc:  # noqa: BLE001 - one bad page must not lose the rest
                logger.warning("Page %d could not be recognised: %s", index + 1, exc)
                pages.append(
                    PdfPage(number=index + 1, text=embedded, recognised=True, confidence=0.0)
                )
                continue

            pages.append(
                PdfPage(
                    number=index + 1,
                    text=result.text,
                    recognised=True,
                    confidence=result.confidence,
                )
            )
    finally:
        if document is not None:
            document.close()

    recognised = [page for page in pages if page.recognised]
    scored = [page.confidence for page in pages if page.text]
    text = "\n\n".join(page.text for page in pages if page.text).strip()

    return PdfExtraction(
        text=text,
        pages=pages,
        page_count=page_count,
        pages_read=len(pages),
        pages_recognised=len(recognised),
        confidence=round(sum(scored) / len(scored), 4) if scored else 0.0,
        truncated=truncated,
    )
