"""Standalone translation endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.api import serializers
from app.detect.detector import detect
from app.documents.schemas import TranslateRequest, TranslateResponse
from app.translate.translator import translate

router = APIRouter(prefix="/translate", tags=["translation"])


@router.post("", response_model=TranslateResponse)
def translate_text(request: TranslateRequest) -> TranslateResponse:
    """Translate arbitrary text, reporting the languages found in the source."""
    result = translate(request.text, target_language=request.target_language)
    return TranslateResponse(
        source_text=request.text,
        translated_text=result.text,
        target_language=result.target_language,
        detected_languages=serializers.detected_languages(detect(request.text)),
        applied=result.applied,
        detail=result.detail,
    )
