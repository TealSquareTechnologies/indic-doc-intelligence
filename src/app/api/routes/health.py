"""Health and capability reporting."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.llm import is_configured
from app.documents.schemas import HealthResponse, LanguageOut
from app.documents.service import count_documents
from app.ocr.engine import installed_languages, tesseract_version
from app.ocr.languages import LANGUAGES
from app.rag import embeddings, store

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the application and its dependencies are ready."""
    version = tesseract_version()
    installed = installed_languages()

    return HealthResponse(
        status="healthy" if version else "degraded",
        tesseract_available=version is not None,
        tesseract_version=version,
        installed_languages=installed,
        model_provider_configured=is_configured(),
        embedding_model_ready=embeddings.is_ready(),
        document_count=count_documents(),
        chunk_count=store.count_chunks(),
    )


@router.get("/languages", response_model=list[LanguageOut], tags=["languages"])
def languages() -> list[LanguageOut]:
    """List supported languages and whether Tesseract can currently extract them."""
    installed = set(installed_languages())
    return [
        LanguageOut(
            code=language.code,
            name=language.name,
            script=language.script,
            installed=language.code in installed,
        )
        for language in LANGUAGES
    ]
