"""Tests for text extraction. Recognition tests run only when Tesseract is available."""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.core.errors import DependencyUnavailableError, ValidationError
from app.ocr import engine

HAS_TESSERACT = engine.tesseract_version() is not None
needs_tesseract = pytest.mark.skipif(not HAS_TESSERACT, reason="Tesseract is not installed")


def render(text: str, size: int = 40) -> bytes:
    """A plain white image with black Latin text."""
    image = Image.new("L", (1400, 300), 255)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 100), text, fill=0, font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_empty_upload_is_rejected():
    with pytest.raises(ValidationError):
        engine.extract_text(b"", language="eng")


def test_unknown_language_codes_are_rejected():
    with pytest.raises(ValidationError, match="Unknown language code"):
        engine.extract_text(b"data", language="klingon")


@needs_tesseract
def test_non_image_bytes_are_rejected():
    with pytest.raises(ValidationError, match="not a readable image"):
        engine.extract_text(b"definitely not an image", language="eng")


@needs_tesseract
def test_english_text_is_recognised():
    result = engine.extract_text(render("The quick brown fox jumps over the lazy dog"),
                                 language="eng")
    assert "quick brown fox" in result.text
    assert result.languages_requested == "eng"
    assert 0.0 < result.confidence <= 1.0


@needs_tesseract
def test_duplicate_language_codes_are_collapsed():
    result = engine.extract_text(render("Hello world"), language="eng+eng")
    assert result.languages_requested == "eng"


@needs_tesseract
def test_missing_traineddata_is_reported(monkeypatch):
    monkeypatch.setattr(engine, "installed_languages", lambda: ["eng"])
    with pytest.raises(DependencyUnavailableError, match="missing traineddata"):
        engine.extract_text(render("Hello"), language="eng+kan")


@needs_tesseract
def test_installed_languages_include_the_supported_set():
    installed = set(engine.installed_languages())
    assert {"eng", "kan", "hin", "tam"} <= installed
