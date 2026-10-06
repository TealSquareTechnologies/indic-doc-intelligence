"""Tests for the SQLite backed vector store."""

from __future__ import annotations

import numpy as np

from app.db.database import get_connection, utc_now
from app.rag import store
from app.rag.chunking import Chunk


def make_chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(ordinal=index, text=text, word_count=len(text.split()))
        for index, text in enumerate(texts)
    ]


def unit_vectors(rows: list[list[float]]) -> np.ndarray:
    matrix = np.array(rows, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def insert_document(filename: str = "sample.txt") -> int:
    timestamp = utc_now()
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO documents (filename, source_kind, status, created_at, updated_at) "
            "VALUES (?, 'text', 'ready', ?, ?)",
            (filename, timestamp, timestamp),
        )
        return int(cursor.lastrowid)


def test_search_on_an_empty_store_returns_nothing(database):
    assert store.search(np.array([1.0, 0.0], dtype=np.float32)) == []


def test_chunks_round_trip(database):
    document_id = insert_document()
    chunks = make_chunks(["first chunk", "second chunk"])
    vectors = unit_vectors([[1.0, 0.0], [0.0, 1.0]])

    assert store.replace_document_chunks(document_id, chunks, vectors) == 2
    assert store.count_chunks(document_id) == 2

    hits = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert hits[0].text == "first chunk"
    assert hits[0].score > hits[1].score


def test_scores_are_cosine_similarities(database):
    document_id = insert_document()
    store.replace_document_chunks(
        document_id, make_chunks(["a"]), unit_vectors([[1.0, 0.0]])
    )
    hits = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert hits[0].score == 1.0


def test_replacing_chunks_discards_the_previous_index(database):
    document_id = insert_document()
    store.replace_document_chunks(
        document_id, make_chunks(["old one", "old two"]), unit_vectors([[1.0, 0.0], [0.0, 1.0]])
    )
    store.replace_document_chunks(
        document_id, make_chunks(["new only"]), unit_vectors([[1.0, 0.0]])
    )

    assert store.count_chunks(document_id) == 1
    hits = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=5)
    assert [hit.text for hit in hits] == ["new only"]


def test_search_can_be_restricted_to_one_document(database):
    first = insert_document("first.txt")
    second = insert_document("second.txt")
    store.replace_document_chunks(first, make_chunks(["from first"]), unit_vectors([[1.0, 0.0]]))
    store.replace_document_chunks(second, make_chunks(["from second"]), unit_vectors([[0.9, 0.1]]))

    hits = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=5, document_ids=[second])
    assert [hit.document_id for hit in hits] == [second]


def test_filtering_to_a_document_with_no_chunks_returns_nothing(database):
    first = insert_document("first.txt")
    empty = insert_document("empty.txt")
    store.replace_document_chunks(first, make_chunks(["only one"]), unit_vectors([[1.0, 0.0]]))

    assert store.search(np.array([1.0, 0.0], dtype=np.float32), document_ids=[empty]) == []


def test_top_k_larger_than_the_corpus_is_safe(database):
    document_id = insert_document()
    store.replace_document_chunks(document_id, make_chunks(["only"]), unit_vectors([[1.0, 0.0]]))
    assert len(store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=50)) == 1


def test_deleting_a_document_removes_its_chunks(database):
    document_id = insert_document()
    store.replace_document_chunks(document_id, make_chunks(["gone"]), unit_vectors([[1.0, 0.0]]))
    store.delete_document_chunks(document_id)

    assert store.count_chunks(document_id) == 0
    assert store.search(np.array([1.0, 0.0], dtype=np.float32)) == []


def test_deleting_a_document_cascades_to_its_chunks(database):
    document_id = insert_document()
    store.replace_document_chunks(document_id, make_chunks(["gone"]), unit_vectors([[1.0, 0.0]]))

    with get_connection() as connection:
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    assert store.count_chunks(document_id) == 0
