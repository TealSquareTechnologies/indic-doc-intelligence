"""Tests for indexing and grounded answering, with the model and embeddings stubbed."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import DependencyUnavailableError, ValidationError
from app.db.database import get_connection, utc_now
from app.rag import embeddings, pipeline, store


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic two-dimensional embeddings keyed on whether text mentions leave."""
    rows = []
    for text in texts:
        rows.append([1.0, 0.0] if "leave" in text.lower() else [0.0, 1.0])
    return np.asarray(rows, dtype=np.float32)


def insert_document() -> int:
    timestamp = utc_now()
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO documents (filename, source_kind, status, created_at, updated_at) "
            "VALUES ('policy.txt', 'text', 'ready', ?, ?)",
            (timestamp, timestamp),
        )
        return int(cursor.lastrowid)


@pytest.fixture
def stubbed(database, monkeypatch):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    monkeypatch.setattr(embeddings, "embed_one", lambda text: fake_embed([text])[0])
    return database


def test_a_blank_question_is_rejected(stubbed):
    with pytest.raises(ValidationError):
        pipeline.ask("   ")


def test_an_empty_corpus_says_so(stubbed):
    answer = pipeline.ask("What is the leave policy?")
    assert answer.grounded is False
    assert "No documents" in answer.answer
    assert answer.sources == []


def test_index_text_stores_one_chunk_per_piece(stubbed):
    document_id = insert_document()
    stored = pipeline.index_text(document_id, "Annual leave is 20 days.\n\nSick leave is 10 days.")
    assert stored == 1
    assert store.count_chunks(document_id) == 1


def test_indexing_empty_text_clears_the_document(stubbed):
    document_id = insert_document()
    pipeline.index_text(document_id, "Annual leave is 20 days.")
    assert pipeline.index_text(document_id, "   ") == 0
    assert store.count_chunks(document_id) == 0


def test_weak_retrieval_refuses_instead_of_guessing(stubbed, monkeypatch):
    document_id = insert_document()
    pipeline.index_text(document_id, "Office hours are nine to five.")
    called = []
    monkeypatch.setattr(pipeline, "complete", lambda *a, **k: called.append(1) or "guess")

    answer = pipeline.ask("What is the leave policy?", threshold=0.5)

    assert answer.answer == pipeline.NO_ANSWER
    assert answer.grounded is False
    assert called == []
    assert answer.sources  # the weak hits are still reported


def test_strong_retrieval_calls_the_model_with_the_sources(stubbed, monkeypatch):
    document_id = insert_document()
    pipeline.index_text(document_id, "Annual leave is 20 days per year.")
    prompts = []

    def fake_complete(prompt, *, system=None, temperature=0.1, model=None):
        prompts.append(prompt)
        return "20 days [Source 1]"

    monkeypatch.setattr(pipeline, "complete", fake_complete)
    answer = pipeline.ask("How much annual leave is there?", threshold=0.2)

    assert answer.grounded is True
    assert answer.answer.startswith("20 days")
    assert "Source 1:" in prompts[0]
    assert "Annual leave is 20 days" in prompts[0]
    assert answer.sources[0].document_id == document_id


def test_the_model_saying_no_answer_is_not_grounded(stubbed, monkeypatch):
    document_id = insert_document()
    pipeline.index_text(document_id, "Annual leave is 20 days per year.")
    monkeypatch.setattr(pipeline, "complete", lambda *a, **k: pipeline.NO_ANSWER)

    answer = pipeline.ask("How much leave?", threshold=0.2)
    assert answer.grounded is False
    assert answer.answer == pipeline.NO_ANSWER


def test_a_model_failure_keeps_the_sources(stubbed, monkeypatch):
    document_id = insert_document()
    pipeline.index_text(document_id, "Annual leave is 20 days per year.")

    def failing(*args, **kwargs):
        raise DependencyUnavailableError("provider down")

    monkeypatch.setattr(pipeline, "complete", failing)
    answer = pipeline.ask("How much leave?", threshold=0.2)
    assert answer.grounded is False
    assert "provider down" in answer.answer
    assert len(answer.sources) == 1


def test_search_can_be_limited_to_chosen_documents(stubbed, monkeypatch):
    first = insert_document()
    second = insert_document()
    pipeline.index_text(first, "Leave policy from the first document.")
    pipeline.index_text(second, "Leave policy from the second document.")
    monkeypatch.setattr(pipeline, "complete", lambda *a, **k: "ok")

    answer = pipeline.ask("leave", document_ids=[second], threshold=0.0)
    assert {source.document_id for source in answer.sources} == {second}
