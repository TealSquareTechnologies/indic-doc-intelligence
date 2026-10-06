"""Retrieval and grounded answering."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.core.errors import AppError, ValidationError
from app.core.llm import complete
from app.core.logging import get_logger
from app.rag import embeddings, store
from app.rag.chunking import ChunkStrategy, chunk_text

logger = get_logger(__name__)

NO_ANSWER = "The answer is not available in the retrieved context."

SYSTEM_PROMPT = (
    "You answer questions strictly from supplied source material. You never draw on "
    "outside knowledge and you never speculate."
)

USER_PROMPT = """Answer the question using only the context below.

Rules:
1. Use only facts stated in the context.
2. If the context does not contain the answer, reply exactly: {no_answer}
3. Cite the source numbers you relied on, like [Source 2].
4. Be concise and do not restate the question.

Context:
------------------
{context}
------------------

Question: {question}"""


@dataclass(frozen=True)
class Source:
    """A retrieved chunk offered as evidence for an answer."""

    document_id: int
    chunk_id: int
    ordinal: int
    score: float
    text: str


@dataclass(frozen=True)
class Answer:
    """The result of asking a question against the indexed corpus."""

    question: str
    answer: str
    grounded: bool
    best_score: float
    sources: list[Source]


def index_text(
    document_id: int,
    text: str,
    strategy: ChunkStrategy = ChunkStrategy.WORDS,
    size: int | None = None,
    overlap: int | None = None,
) -> int:
    """Chunk, embed and store a document's text. Returns the number of chunks stored."""
    chunks = chunk_text(text, strategy=strategy, size=size, overlap=overlap)
    if not chunks:
        store.delete_document_chunks(document_id)
        return 0

    vectors = embeddings.embed([chunk.text for chunk in chunks])
    stored = store.replace_document_chunks(document_id, chunks, vectors)
    logger.info("Indexed document %d into %d chunks", document_id, stored)
    return stored


def _build_context(hits: list[store.SearchHit]) -> str:
    return "\n\n".join(
        f"Source {position}:\n{hit.text}" for position, hit in enumerate(hits, start=1)
    )


def ask(
    question: str,
    document_ids: list[int] | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
) -> Answer:
    """Answer a question from the indexed corpus, restricted to the retrieved context."""
    if not question or not question.strip():
        raise ValidationError("A question is required.")

    settings = get_settings()
    top_k = settings.retrieval_top_k if top_k is None else top_k
    threshold = settings.similarity_threshold if threshold is None else threshold

    question = question.strip()
    query_vector = embeddings.embed_one(question)
    hits = store.search(query_vector, top_k=top_k, document_ids=document_ids)

    if not hits:
        empty_corpus = store.count_chunks() == 0
        return Answer(
            question=question,
            answer="No documents have been indexed yet." if empty_corpus else NO_ANSWER,
            grounded=False,
            best_score=0.0,
            sources=[],
        )

    sources = [
        Source(
            document_id=hit.document_id,
            chunk_id=hit.chunk_id,
            ordinal=hit.ordinal,
            score=hit.score,
            text=hit.text,
        )
        for hit in hits
    ]
    best_score = hits[0].score

    # Retrieval that is this weak means the corpus does not cover the question.
    if best_score < threshold:
        return Answer(
            question=question,
            answer=NO_ANSWER,
            grounded=False,
            best_score=best_score,
            sources=sources,
        )

    prompt = USER_PROMPT.format(
        no_answer=NO_ANSWER, context=_build_context(hits), question=question
    )

    try:
        answer = complete(prompt, system=SYSTEM_PROMPT, temperature=0.1)
    except AppError as exc:
        logger.warning("Answer generation failed: %s", exc.message)
        return Answer(
            question=question,
            answer=f"Retrieval succeeded but the answer could not be generated: {exc.message}",
            grounded=False,
            best_score=best_score,
            sources=sources,
        )

    return Answer(
        question=question,
        answer=answer or NO_ANSWER,
        grounded=bool(answer) and answer != NO_ANSWER,
        best_score=best_score,
        sources=sources,
    )
