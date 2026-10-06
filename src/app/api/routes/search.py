"""Question answering over the indexed corpus."""

from __future__ import annotations

from fastapi import APIRouter

from app.api import serializers
from app.documents.schemas import AskRequest, AskResponse
from app.rag.pipeline import ask

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Answer a question using only the text retrieved from stored documents."""
    answer = ask(
        request.question,
        document_ids=request.document_ids,
        top_k=request.top_k,
        threshold=request.threshold,
    )
    return AskResponse(**serializers.answer_payload(answer))
