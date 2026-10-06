"""Conversion between database rows, pipeline results and API models."""

from __future__ import annotations

import json
import sqlite3

from app.detect.detector import DetectedLanguage
from app.documents.schemas import (
    DetectedLanguageOut,
    Document,
    DocumentSummary,
    SourceOut,
)
from app.rag.pipeline import Answer, Source


def detected_languages(items: list[DetectedLanguage]) -> list[DetectedLanguageOut]:
    """Convert detector output into API models."""
    return [
        DetectedLanguageOut(
            code=item.code,
            name=item.name,
            script=item.script,
            share=item.share,
            character_count=item.character_count,
            note=item.note,
        )
        for item in items
    ]


def _stored_languages(raw: str | None) -> list[DetectedLanguageOut]:
    """Rebuild detection results from the JSON column, tolerating older rows."""
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except (TypeError, ValueError):
        return []

    return [
        DetectedLanguageOut(
            code=entry.get("code", ""),
            name=entry.get("name", ""),
            script=entry.get("script", ""),
            share=float(entry.get("share", 0.0)),
            character_count=int(entry.get("character_count", 0)),
            note=entry.get("note", ""),
        )
        for entry in entries
        if isinstance(entry, dict)
    ]


def _stored_notes(raw: str | None) -> list[str]:
    """Rebuild processing notes from the JSON column, tolerating older rows."""
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(entry) for entry in entries] if isinstance(entries, list) else []


def document_summary(row: sqlite3.Row) -> DocumentSummary:
    """Convert a listing row into a summary model."""
    return DocumentSummary(
        id=row["id"],
        filename=row["filename"],
        content_type=row["content_type"],
        source_kind=row["source_kind"],
        status=row["status"],
        error=row["error"],
        detected_languages=_stored_languages(row["detected_languages"]),
        notes=_stored_notes(row["notes"]),
        chunk_count=row["chunk_count"],
        processing_time_ms=row["processing_time_ms"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def document_detail(row: sqlite3.Row) -> Document:
    """Convert a full row into a detail model."""
    return Document(
        **document_summary(row).model_dump(),
        ocr_languages=row["ocr_languages"],
        raw_text=row["raw_text"] or "",
        corrected_text=row["corrected_text"] or "",
        translated_text=row["translated_text"],
    )


def sources(items: list[Source]) -> list[SourceOut]:
    """Convert retrieved chunks into API models."""
    return [
        SourceOut(
            document_id=item.document_id,
            chunk_id=item.chunk_id,
            ordinal=item.ordinal,
            score=item.score,
            text=item.text,
        )
        for item in items
    ]


def answer_payload(answer: Answer) -> dict:
    """Flatten an answer into the shape the API returns."""
    return {
        "question": answer.question,
        "answer": answer.answer,
        "grounded": answer.grounded,
        "best_score": answer.best_score,
        "sources": sources(answer.sources),
    }
