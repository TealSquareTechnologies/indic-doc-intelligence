"""Document orchestration. Every entry point runs the pipeline through here."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.errors import AppError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.correct.corrector import correct as correct_text
from app.db.database import get_connection, utc_now
from app.detect.detector import DetectedLanguage, detect
from app.ocr.engine import extract_text
from app.ocr.languages import normalise_language_string
from app.ocr.pdf import extract_pdf
from app.rag import store
from app.rag.chunking import ChunkStrategy
from app.rag.pipeline import index_text
from app.translate.translator import translate as translate_text

logger = get_logger(__name__)

TEXT_CONTENT_TYPES = {"text/plain", "text/markdown", "application/json"}
IMAGE_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp",
                       "image/tiff"}
PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}


@dataclass(frozen=True)
class PipelineResult:
    """Everything one pass of the pipeline produced."""

    raw_text: str
    corrected_text: str
    translated_text: str | None
    detected: list[DetectedLanguage]
    ocr_languages: str
    confidence: float
    correction_applied: bool
    translation_applied: bool
    notes: list[str]
    elapsed_ms: int


def _serialise_languages(detected: list[DetectedLanguage]) -> str:
    return json.dumps(
        [
            {
                "code": item.code,
                "name": item.name,
                "script": item.script,
                "share": item.share,
                "character_count": item.character_count,
                "note": item.note,
            }
            for item in detected
        ]
    )


def _decode_text_upload(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValidationError("The uploaded text file could not be decoded.")


def classify_upload(content_type: str | None, filename: str) -> str:
    """Decide whether an upload is an image to recognise or text to index directly."""
    normalised = (content_type or "").split(";")[0].strip().lower()
    if normalised in IMAGE_CONTENT_TYPES:
        return "image"
    if normalised in PDF_CONTENT_TYPES:
        return "pdf"
    if normalised in TEXT_CONTENT_TYPES:
        return "text"

    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix in {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"}:
        return "image"
    if suffix == "pdf":
        return "pdf"
    if suffix in {"txt", "md", "text"}:
        return "text"

    raise ValidationError(
        f"Unsupported file type '{content_type or suffix or 'unknown'}'. "
        "Upload an image (JPEG, PNG, WebP, BMP, TIFF), a PDF, or a plain text file."
    )


def run_pipeline(
    data: bytes,
    source_kind: str,
    language: str | None = None,
    apply_correction: bool = True,
    apply_translation: bool = False,
    target_language: str = "English",
) -> PipelineResult:
    """Run the text stages of the pipeline over one upload."""
    started = time.perf_counter()
    notes: list[str] = []

    if source_kind == "image":
        extraction = extract_text(data, language=language)
        raw_text = extraction.text
        confidence = extraction.confidence
        ocr_languages = extraction.languages_requested
        if not raw_text:
            notes.append("No text was recognised in the image.")
    elif source_kind == "pdf":
        pdf = extract_pdf(data, language=language)
        raw_text = pdf.text
        confidence = pdf.confidence
        ocr_languages = normalise_language_string(language or get_settings().default_ocr_languages)

        read = f"Read {pdf.pages_read} of {pdf.page_count} pages"
        if pdf.pages_recognised:
            read += f", {pdf.pages_recognised} by recognition"
        notes.append(f"{read}.")
        if pdf.truncated:
            notes.append(
                f"Only the first {pdf.pages_read} pages were read. "
                "Raise MAX_PDF_PAGES to read more."
            )
        if not raw_text:
            notes.append("No text could be read from this PDF.")
    else:
        raw_text = _decode_text_upload(data).strip()
        confidence = 1.0
        ocr_languages = ""

    detected = detect(raw_text)

    corrected_text = raw_text
    correction_applied = False
    if apply_correction and raw_text:
        result = correct_text(raw_text)
        corrected_text = result.text
        correction_applied = result.applied
        if not result.applied and result.detail:
            notes.append(f"Correction skipped: {result.detail}")

    translated_text: str | None = None
    translation_applied = False
    if apply_translation and corrected_text:
        result = translate_text(corrected_text, target_language=target_language)
        translation_applied = result.applied
        if result.applied:
            translated_text = result.text
        elif result.detail:
            # Left as None: the source text must not be shown as a translation.
            notes.append(f"Translation failed: {result.detail}")

    return PipelineResult(
        raw_text=raw_text,
        corrected_text=corrected_text,
        translated_text=translated_text,
        detected=detected,
        ocr_languages=ocr_languages,
        confidence=confidence,
        correction_applied=correction_applied,
        translation_applied=translation_applied,
        notes=notes,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


def create_document(
    filename: str,
    content_type: str | None,
    data: bytes,
    language: str | None = None,
    apply_correction: bool = True,
    apply_translation: bool = False,
    target_language: str = "English",
    index: bool = True,
    strategy: ChunkStrategy = ChunkStrategy.WORDS,
) -> sqlite3.Row:
    """Process an upload and persist it, indexing the result for retrieval."""
    if not data:
        raise ValidationError("The uploaded file is empty.")

    source_kind = classify_upload(content_type, filename)
    timestamp = utc_now()

    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO documents (filename, content_type, source_kind, status, created_at, "
            "updated_at) VALUES (?, ?, ?, 'processing', ?, ?)",
            (filename, content_type, source_kind, timestamp, timestamp),
        )
        document_id = int(cursor.lastrowid)

    try:
        result = run_pipeline(
            data,
            source_kind=source_kind,
            language=language,
            apply_correction=apply_correction,
            apply_translation=apply_translation,
            target_language=target_language,
        )
    except AppError as exc:
        _mark_failed(document_id, exc.message)
        raise

    chunk_count = 0
    if index and result.corrected_text:
        try:
            chunk_count = index_text(document_id, result.corrected_text, strategy=strategy)
        except AppError as exc:
            # A document that cannot be indexed is still worth keeping for its text.
            logger.warning("Indexing failed for document %d: %s", document_id, exc.message)
            result.notes.append(f"Indexing skipped: {exc.message}")

    with get_connection() as connection:
        connection.execute(
            "UPDATE documents SET status = 'ready', ocr_languages = ?, detected_languages = ?, "
            "raw_text = ?, corrected_text = ?, translated_text = ?, notes = ?, "
            "chunk_count = ?, processing_time_ms = ?, updated_at = ? WHERE id = ?",
            (
                result.ocr_languages,
                _serialise_languages(result.detected),
                result.raw_text,
                result.corrected_text,
                result.translated_text,
                json.dumps(result.notes),
                chunk_count,
                result.elapsed_ms,
                utc_now(),
                document_id,
            ),
        )

    logger.info("Document %d ready: %s", document_id, filename)
    return get_document(document_id)


def _mark_failed(document_id: int, message: str) -> None:
    with get_connection() as connection:
        connection.execute(
            "UPDATE documents SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
            (message, utc_now(), document_id),
        )


def get_document(document_id: int) -> sqlite3.Row:
    """Fetch one document, raising NotFoundError when it does not exist."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    if row is None:
        raise NotFoundError(f"No document with id {document_id}.")
    return row


def list_documents(limit: int = 100, offset: int = 0) -> list[sqlite3.Row]:
    """List documents, newest first, without their text."""
    with get_connection() as connection:
        return connection.execute(
            "SELECT id, filename, content_type, source_kind, status, error, detected_languages, "
            "notes, chunk_count, processing_time_ms, created_at, updated_at FROM documents "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()


def delete_document(document_id: int) -> None:
    """Remove a document and everything indexed from it."""
    get_document(document_id)
    with get_connection() as connection:
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    store.invalidate_cache()
    logger.info("Deleted document %d", document_id)


def reindex_document(
    document_id: int,
    strategy: ChunkStrategy = ChunkStrategy.WORDS,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    """Rebuild a document's chunks, for example after changing the chunking settings."""
    row = get_document(document_id)
    text = row["corrected_text"] or row["raw_text"]
    if not text:
        raise ValidationError("This document has no text to index.")

    count = index_text(document_id, text, strategy=strategy, size=chunk_size,
                       overlap=chunk_overlap)
    with get_connection() as connection:
        connection.execute(
            "UPDATE documents SET chunk_count = ?, updated_at = ? WHERE id = ?",
            (count, utc_now(), document_id),
        )
    return count


def count_documents() -> int:
    """Total number of stored documents."""
    with get_connection() as connection:
        row = connection.execute("SELECT COUNT(*) AS total FROM documents").fetchone()
    return int(row["total"])
