"""Tests for PDF handling."""

from __future__ import annotations

import io

import pypdf
import pytest
from PIL import Image, ImageDraw

from app.core.errors import ValidationError
from app.documents.service import classify_upload
from app.ocr.pdf import MIN_EMBEDDED_CHARACTERS, extract_pdf, looks_like_pdf


def scanned_pdf(pages: int = 1, text: str = "Probation lasts 90 days.") -> bytes:
    """A PDF whose pages are images, with no text layer at all."""
    images = []
    for index in range(pages):
        image = Image.new("RGB", (1000, 260), "white")
        draw = ImageDraw.Draw(image)
        draw.text((40, 100), f"{text} Page {index + 1}.", fill="black")
        images.append(image)

    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def text_layer_pdf(body: str) -> bytes:
    """A minimal PDF built around a single text object."""
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    escaped = body.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    stream = pypdf.generic.DecodedStreamObject()
    stream.set_data(f"BT /F1 18 Tf 50 700 Td ({escaped}) Tj ET".encode("latin-1"))
    font = pypdf.generic.DictionaryObject(
        {
            pypdf.generic.NameObject("/Type"): pypdf.generic.NameObject("/Font"),
            pypdf.generic.NameObject("/Subtype"): pypdf.generic.NameObject("/Type1"),
            pypdf.generic.NameObject("/BaseFont"): pypdf.generic.NameObject("/Helvetica"),
        }
    )
    resources = pypdf.generic.DictionaryObject(
        {
            pypdf.generic.NameObject("/Font"): pypdf.generic.DictionaryObject(
                {pypdf.generic.NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    page[pypdf.generic.NameObject("/Contents")] = writer._add_object(stream)
    page[pypdf.generic.NameObject("/Resources")] = resources

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_looks_like_pdf_checks_the_signature():
    assert looks_like_pdf(b"%PDF-1.7 rest") is True
    assert looks_like_pdf(b"\x89PNG\r\n") is False
    assert looks_like_pdf(b"") is False


def test_pdf_is_classified_by_content_type():
    assert classify_upload("application/pdf", "report.bin") == "pdf"


def test_pdf_is_classified_by_extension():
    assert classify_upload(None, "report.PDF") == "pdf"


def test_pdf_is_classified_when_content_type_is_generic():
    assert classify_upload("application/octet-stream", "scan.pdf") == "pdf"


def test_non_pdf_bytes_are_rejected():
    with pytest.raises(ValidationError, match="not a PDF"):
        extract_pdf(b"\x89PNG\r\n\x1a\n plainly not a pdf")


def test_empty_input_is_rejected():
    with pytest.raises(ValidationError, match="empty"):
        extract_pdf(b"")


def test_corrupt_pdf_is_rejected():
    with pytest.raises(ValidationError):
        extract_pdf(b"%PDF-1.4 truncated and broken")


def test_text_layer_is_read_without_recognition(isolated_settings):
    body = "Annual leave is 24 days per year and the notice period is 60 days."
    assert len(body) >= MIN_EMBEDDED_CHARACTERS

    result = extract_pdf(text_layer_pdf(body))

    assert result.page_count == 1
    assert result.pages_recognised == 0
    assert result.pages[0].recognised is False
    assert "24 days" in result.text
    assert result.confidence == 1.0


def test_page_count_is_reported(isolated_settings):
    result = extract_pdf(scanned_pdf(pages=3))
    assert result.page_count == 3
    assert result.pages_read == 3


def test_scanned_pages_are_sent_to_recognition(isolated_settings):
    result = extract_pdf(scanned_pdf(pages=2))

    assert result.pages_recognised == 2
    assert all(page.recognised for page in result.pages)


def test_page_limit_truncates_and_says_so(isolated_settings):
    result = extract_pdf(scanned_pdf(pages=4), max_pages=2)

    assert result.page_count == 4
    assert result.pages_read == 2
    assert result.truncated is True


def test_no_limit_reads_every_page(isolated_settings):
    result = extract_pdf(scanned_pdf(pages=3), max_pages=0)
    assert result.pages_read == 3
    assert result.truncated is False


def test_pipeline_reports_pdf_pages_in_notes(database):
    from app.documents import service

    result = service.run_pipeline(
        scanned_pdf(pages=2),
        source_kind="pdf",
        apply_correction=False,
        apply_translation=False,
    )

    assert any("pages" in note for note in result.notes)


def test_pipeline_warns_when_pages_are_dropped(database, monkeypatch):
    from app.core.config import get_settings
    from app.documents import service

    monkeypatch.setenv("MAX_PDF_PAGES", "1")
    get_settings.cache_clear()

    result = service.run_pipeline(
        scanned_pdf(pages=3),
        source_kind="pdf",
        apply_correction=False,
        apply_translation=False,
    )

    assert any("Only the first" in note for note in result.notes)
