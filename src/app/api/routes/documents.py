"""Document endpoints: upload, inspect, reindex and remove."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Query, UploadFile

from app.api import serializers
from app.api.deps import read_upload
from app.documents import service
from app.documents.schemas import (
    DeleteResponse,
    Document,
    DocumentSummary,
    ReindexRequest,
)
from app.rag.chunking import ChunkStrategy

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=Document, status_code=201)
async def upload_document(
    file: UploadFile = File(..., description="Image or plain text file"),
    language: str | None = Form(default=None, description="Language string such as 'kan+eng'"),
    correct: bool = Form(default=True),
    translate: bool = Form(default=False),
    target_language: str = Form(default="English"),
    index: bool = Form(default=True, description="Make the document searchable"),
    strategy: ChunkStrategy = Form(default=ChunkStrategy.WORDS),
) -> Document:
    """Run the full pipeline over an upload and store the result."""
    data = await read_upload(file)
    row = service.create_document(
        filename=file.filename or "upload",
        content_type=file.content_type,
        data=data,
        language=language,
        apply_correction=correct,
        apply_translation=translate,
        target_language=target_language,
        index=index,
        strategy=strategy,
    )
    return serializers.document_detail(row)


@router.get("", response_model=list[DocumentSummary])
def list_documents(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[DocumentSummary]:
    """List stored documents, newest first."""
    return [serializers.document_summary(row) for row in service.list_documents(limit, offset)]


@router.get("/{document_id}", response_model=Document)
def get_document(document_id: int) -> Document:
    """Fetch one document with every text stage it passed through."""
    return serializers.document_detail(service.get_document(document_id))


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: int) -> DeleteResponse:
    """Remove a document and everything indexed from it."""
    service.delete_document(document_id)
    return DeleteResponse(id=document_id, deleted=True)


@router.post("/{document_id}/reindex", response_model=Document)
def reindex_document(document_id: int, request: ReindexRequest) -> Document:
    """Rebuild a document's chunks with different chunking settings."""
    service.reindex_document(
        document_id,
        strategy=request.strategy,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    return serializers.document_detail(service.get_document(document_id))
