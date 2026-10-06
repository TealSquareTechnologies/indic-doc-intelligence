"""Splitting documents into retrievable pieces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from app.core.config import get_settings
from app.core.errors import ValidationError


class ChunkStrategy(StrEnum):
    """How a document is divided before it is embedded."""

    WORDS = "words"
    PARAGRAPHS = "paragraphs"
    SENTENCES = "sentences"


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of a document."""

    ordinal: int
    text: str
    word_count: int


# Sentence terminators, including the danda and double danda used by Indic scripts.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।॥])\s+")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")


def _split_words(words: list[str], size: int, overlap: int) -> list[str]:
    step = size - overlap
    pieces: list[str] = []
    start = 0
    while start < len(words):
        pieces.append(" ".join(words[start : start + size]))
        start += step
    return pieces


def _pack_units(units: list[str], size: int, overlap: int) -> list[str]:
    """Group units into chunks of at most size words, splitting any oversized unit."""
    pieces: list[str] = []
    buffer: list[str] = []
    buffer_words = 0

    for unit in units:
        unit_words = len(unit.split())

        if unit_words > size:
            if buffer:
                pieces.append(" ".join(buffer))
                buffer, buffer_words = [], 0
            pieces.extend(_split_words(unit.split(), size, overlap))
            continue

        if buffer_words + unit_words > size and buffer:
            pieces.append(" ".join(buffer))
            buffer, buffer_words = [], 0

        buffer.append(unit)
        buffer_words += unit_words

    if buffer:
        pieces.append(" ".join(buffer))
    return pieces


def chunk_text(
    text: str,
    strategy: ChunkStrategy = ChunkStrategy.WORDS,
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Divide text into chunks of at most size words."""
    settings = get_settings()
    size = settings.chunk_size if size is None else size
    overlap = settings.chunk_overlap if overlap is None else overlap

    if size < 1:
        raise ValidationError("Chunk size must be at least one word.")
    if overlap < 0:
        raise ValidationError("Chunk overlap cannot be negative.")
    if overlap >= size:
        raise ValidationError("Chunk overlap must be smaller than chunk size.")

    if not text or not text.strip():
        return []

    if strategy is ChunkStrategy.WORDS:
        pieces = _split_words(text.split(), size, overlap)
    elif strategy is ChunkStrategy.PARAGRAPHS:
        units = [part.strip() for part in _PARAGRAPH_BOUNDARY.split(text) if part.strip()]
        pieces = _pack_units(units, size, overlap)
    else:
        units = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
        pieces = _pack_units(units, size, overlap)

    return [
        Chunk(ordinal=index, text=piece, word_count=len(piece.split()))
        for index, piece in enumerate(pieces)
        if piece.strip()
    ]
