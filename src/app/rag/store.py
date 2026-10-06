"""Vector storage on top of SQLite."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass

import numpy as np

from app.core.logging import get_logger
from app.db.database import get_connection, utc_now
from app.rag.chunking import Chunk

logger = get_logger(__name__)

_cache_lock = threading.Lock()
_cached_matrix: np.ndarray | None = None
_cached_rows: list[tuple[int, int, int, str]] = []


@dataclass(frozen=True)
class SearchHit:
    """One chunk returned by a similarity search."""

    chunk_id: int
    document_id: int
    ordinal: int
    text: str
    score: float


def invalidate_cache() -> None:
    """Drop the cached matrix so the next search reloads it."""
    global _cached_matrix, _cached_rows
    with _cache_lock:
        _cached_matrix = None
        _cached_rows = []


def _load_matrix() -> tuple[np.ndarray, list[tuple[int, int, int, str]]]:
    """Return every stored embedding as a matrix, with the rows describing each vector."""
    global _cached_matrix, _cached_rows
    with _cache_lock:
        if _cached_matrix is not None:
            return _cached_matrix, _cached_rows

        with get_connection() as connection:
            records = connection.execute(
                "SELECT id, document_id, ordinal, text, embedding "
                "FROM chunks ORDER BY document_id, ordinal"
            ).fetchall()

        if not records:
            _cached_matrix = np.zeros((0, 0), dtype=np.float32)
            _cached_rows = []
            return _cached_matrix, _cached_rows

        vectors = [np.frombuffer(record["embedding"], dtype=np.float32) for record in records]
        _cached_matrix = np.vstack(vectors)
        _cached_rows = [
            (record["id"], record["document_id"], record["ordinal"], record["text"])
            for record in records
        ]
        logger.debug("Loaded %d embeddings into memory", len(_cached_rows))
        return _cached_matrix, _cached_rows


def replace_document_chunks(
    document_id: int, chunks: list[Chunk], embeddings: np.ndarray
) -> int:
    """Store the chunks for a document, discarding anything indexed for it before."""
    if len(chunks) != len(embeddings):
        raise ValueError("The number of chunks and embeddings must match.")

    timestamp = utc_now()
    rows = [
        (
            document_id,
            chunk.ordinal,
            chunk.text,
            embeddings[index].astype(np.float32).tobytes(),
            int(embeddings.shape[1]),
            timestamp,
        )
        for index, chunk in enumerate(chunks)
    ]

    with get_connection() as connection:
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        if rows:
            connection.executemany(
                "INSERT INTO chunks (document_id, ordinal, text, embedding, dimension, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    invalidate_cache()
    return len(rows)


def delete_document_chunks(document_id: int) -> None:
    """Remove every chunk belonging to a document."""
    with get_connection() as connection:
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    invalidate_cache()


def count_chunks(document_id: int | None = None) -> int:
    """Count stored chunks, either in total or for one document."""
    with get_connection() as connection:
        if document_id is None:
            row = connection.execute("SELECT COUNT(*) AS total FROM chunks").fetchone()
        else:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchone()
    return int(row["total"])


def search(
    query_vector: np.ndarray, top_k: int = 3, document_ids: list[int] | None = None
) -> list[SearchHit]:
    """Return the closest chunks to a query vector, highest score first."""
    matrix, rows = _load_matrix()
    if matrix.size == 0 or top_k < 1:
        return []

    scores = matrix @ query_vector.astype(np.float32)

    if document_ids:
        wanted = set(document_ids)
        mask = np.array([row[1] in wanted for row in rows], dtype=bool)
        if not mask.any():
            return []
        # Push filtered rows below any real score so they can never be selected.
        scores = np.where(mask, scores, -np.inf)

    count = min(top_k, int(np.isfinite(scores).sum()))
    if count < 1:
        return []

    top_indices = np.argpartition(-scores, count - 1)[:count]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    return [
        SearchHit(
            chunk_id=rows[index][0],
            document_id=rows[index][1],
            ordinal=rows[index][2],
            text=rows[index][3],
            score=round(float(scores[index]), 4),
        )
        for index in top_indices
    ]


def chunks_for_document(document_id: int) -> list[sqlite3.Row]:
    """Return the stored chunks for a document in order, without their vectors."""
    with get_connection() as connection:
        return connection.execute(
            "SELECT id, ordinal, text FROM chunks WHERE document_id = ? ORDER BY ordinal",
            (document_id,),
        ).fetchall()
