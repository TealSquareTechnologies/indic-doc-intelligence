"""Tests for upload classification and the text stages of the pipeline."""

from __future__ import annotations

import pytest

from app.core.errors import NotFoundError, ValidationError
from app.documents import service
from app.rag import pipeline


@pytest.mark.parametrize(
    ("content_type", "filename", "expected"),
    [
        ("image/png", "scan.png", "image"),
        ("image/jpeg; charset=binary", "photo.jpg", "image"),
        ("application/pdf", "report.pdf", "pdf"),
        ("text/plain", "notes.txt", "text"),
        ("application/octet-stream", "scan.tiff", "image"),
        ("application/octet-stream", "report.PDF", "pdf"),
        (None, "notes.md", "text"),
    ],
)
def test_uploads_are_classified_by_type_then_extension(content_type, filename, expected):
    assert service.classify_upload(content_type, filename) == expected


def test_unknown_uploads_are_rejected():
    with pytest.raises(ValidationError, match="Unsupported file type"):
        service.classify_upload("application/zip", "archive.zip")


def test_text_uploads_skip_recognition(database):
    result = service.run_pipeline(
        "ಕನ್ನಡ ಪಠ್ಯ ಮತ್ತು English words".encode("utf-8"),
        source_kind="text",
        apply_correction=False,
    )
    assert result.raw_text.startswith("ಕನ್ನಡ")
    assert result.confidence == 1.0
    assert result.ocr_languages == ""
    assert [item.code for item in result.detected] == ["kan", "eng"]


def test_utf16_text_uploads_are_decoded(database):
    result = service.run_pipeline("hello".encode("utf-16"), source_kind="text",
                                  apply_correction=False)
    assert result.raw_text == "hello"


def test_arbitrary_bytes_fall_back_to_latin1(database):
    # Latin-1 accepts every byte, so a text upload is never rejected for its encoding.
    # An odd byte count keeps the UTF-16 attempt from swallowing the input first.
    assert service._decode_text_upload(b"caf\xe9!") == "café!"


def test_create_document_stores_text_and_languages(database, monkeypatch):
    monkeypatch.setattr(service, "index_text", lambda *a, **k: 3)
    row = service.create_document(
        "notes.txt", "text/plain", "यह हिंदी पाठ है".encode("utf-8"), apply_correction=False
    )
    assert row["status"] == "ready"
    assert row["chunk_count"] == 3
    assert "Hindi" in row["detected_languages"]
    assert service.count_documents() == 1


def test_deleting_a_document_removes_it(database, monkeypatch):
    monkeypatch.setattr(service, "index_text", lambda *a, **k: 0)
    row = service.create_document("a.txt", "text/plain", b"text", apply_correction=False)
    service.delete_document(row["id"])
    with pytest.raises(NotFoundError):
        service.get_document(row["id"])


def test_indexing_failure_keeps_the_document(database, monkeypatch):
    from app.core.errors import DependencyUnavailableError

    def failing(*args, **kwargs):
        raise DependencyUnavailableError("no embeddings")

    monkeypatch.setattr(service, "index_text", failing)
    row = service.create_document("a.txt", "text/plain", b"some text", apply_correction=False)
    assert row["status"] == "ready"
    assert row["chunk_count"] == 0
    assert "Indexing skipped" in row["notes"]


def test_pipeline_result_records_elapsed_time(database):
    result = service.run_pipeline(b"plain text", source_kind="text", apply_correction=False)
    assert result.elapsed_ms >= 0
    assert pipeline.NO_ANSWER  # the answering module is importable alongside the service
