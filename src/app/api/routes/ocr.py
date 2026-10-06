"""Stateless recognition endpoints."""

from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, File, Form, UploadFile

from app.api import serializers
from app.api.deps import read_upload
from app.core.errors import ValidationError
from app.documents.schemas import OCRRequest, OCRResponse
from app.documents.service import classify_upload, run_pipeline

router = APIRouter(prefix="/ocr", tags=["recognition"])


def _to_response(result, success: bool = True) -> OCRResponse:
    return OCRResponse(
        success=success,
        languages=serializers.detected_languages(result.detected),
        ocr_languages=result.ocr_languages,
        raw_text=result.raw_text,
        corrected_text=result.corrected_text,
        translated_text=result.translated_text,
        confidence=result.confidence,
        correction_applied=result.correction_applied,
        translation_applied=result.translation_applied,
        notes=result.notes,
        processing_time_ms=result.elapsed_ms,
    )


def _decode_base64(payload: str) -> bytes:
    """Decode base64 image data, accepting a data URL prefix."""
    data = payload.split(",", 1)[1] if payload.startswith("data:") else payload
    try:
        decoded = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError(f"The base64 payload could not be decoded: {exc}") from exc

    if not decoded:
        raise ValidationError("The base64 payload is empty.")
    return decoded


@router.post("", response_model=OCRResponse)
def recognise_base64(request: OCRRequest) -> OCRResponse:
    """Extract, correct and optionally translate text from inline base64 image data."""
    result = run_pipeline(
        _decode_base64(request.image_base64),
        source_kind="image",
        language=request.language,
        apply_correction=request.correct,
        apply_translation=request.translate,
        target_language=request.target_language,
    )
    return _to_response(result)


@router.post("/file", response_model=OCRResponse)
async def recognise_upload(
    file: UploadFile = File(..., description="Image or plain text file"),
    language: str | None = Form(default=None),
    correct: bool = Form(default=True),
    translate: bool = Form(default=False),
    target_language: str = Form(default="English"),
) -> OCRResponse:
    """Extract, correct and optionally translate text from an uploaded file."""
    data = await read_upload(file)
    source_kind = classify_upload(file.content_type, file.filename or "upload")

    result = run_pipeline(
        data,
        source_kind=source_kind,
        language=language,
        apply_correction=correct,
        apply_translation=translate,
        target_language=target_language,
    )
    return _to_response(result)
