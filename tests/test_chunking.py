"""Tests for document chunking."""

from __future__ import annotations

import pytest

from app.core.errors import ValidationError
from app.rag.chunking import ChunkStrategy, chunk_text


def words(count: int) -> str:
    return " ".join(f"w{index}" for index in range(count))


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_fixed_windows_respect_the_size_limit():
    chunks = chunk_text(words(1000), size=250, overlap=50)
    assert all(chunk.word_count <= 250 for chunk in chunks)
    assert chunks[0].ordinal == 0


def test_fixed_windows_overlap_by_the_requested_amount():
    chunks = chunk_text(words(500), size=100, overlap=20)
    first = chunks[0].text.split()
    second = chunks[1].text.split()
    assert first[-20:] == second[:20]


def test_every_word_is_covered():
    source = words(430)
    chunks = chunk_text(source, size=100, overlap=25)
    recovered = " ".join(chunk.text for chunk in chunks).split()
    assert set(recovered) == set(source.split())


def test_ordinals_are_sequential():
    chunks = chunk_text(words(700), size=100, overlap=10)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_overlap_equal_to_size_is_rejected():
    with pytest.raises(ValidationError, match="overlap must be smaller"):
        chunk_text(words(100), size=50, overlap=50)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValidationError):
        chunk_text(words(100), size=50, overlap=-1)


def test_zero_size_is_rejected():
    with pytest.raises(ValidationError):
        chunk_text(words(100), size=0)


def test_paragraph_strategy_keeps_paragraphs_together():
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = chunk_text(text, strategy=ChunkStrategy.PARAGRAPHS, size=250, overlap=0)
    assert len(chunks) == 1
    assert "First" in chunks[0].text and "Third" in chunks[0].text


def test_paragraph_strategy_splits_an_oversized_paragraph():
    chunks = chunk_text(words(600), strategy=ChunkStrategy.PARAGRAPHS, size=100, overlap=10)
    assert len(chunks) > 1
    assert all(chunk.word_count <= 100 for chunk in chunks)


def test_sentence_strategy_splits_on_the_danda():
    text = "ಇದು ಮೊದಲ ವಾಕ್ಯ। ಇದು ಎರಡನೇ ವಾಕ್ಯ। ಇದು ಮೂರನೇ ವಾಕ್ಯ।"
    chunks = chunk_text(text, strategy=ChunkStrategy.SENTENCES, size=4, overlap=0)
    assert len(chunks) > 1


def test_sentence_strategy_splits_on_full_stops():
    text = "One sentence. Two sentence. Three sentence. Four sentence."
    chunks = chunk_text(text, strategy=ChunkStrategy.SENTENCES, size=2, overlap=0)
    assert len(chunks) >= 2
