"""Request and response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.rag.chunking import ChunkStrategy


class DetectedLanguageOut(BaseModel):
    """One language identified in a document."""

    code: str = Field(description="Tesseract language code")
    name: str = Field(description="Language name")
    script: str = Field(description="Unicode script the language is written in")
    share: float = Field(description="Fraction of script bearing characters, 0 to 1")
    character_count: int = Field(description="Number of characters attributed to this script")
    note: str = Field(default="", description="Caveats about the identification")


class LanguageOut(BaseModel):
    """A language the platform supports."""

    code: str
    name: str
    script: str
    installed: bool = Field(description="Whether Tesseract has the traineddata for it")


class DocumentSummary(BaseModel):
    """A document without its text, for listings."""

    id: int
    filename: str
    content_type: str | None = None
    source_kind: str
    status: str
    error: str | None = None
    detected_languages: list[DetectedLanguageOut] = Field(default_factory=list)
    notes: list[str] = Field(
        default_factory=list, description="What happened while processing, such as pages read"
    )
    chunk_count: int = 0
    processing_time_ms: int = 0
    created_at: str
    updated_at: str


class Document(DocumentSummary):
    """A document with every text stage it passed through."""

    ocr_languages: str | None = None
    raw_text: str = ""
    corrected_text: str = ""
    translated_text: str | None = None


class OCRRequest(BaseModel):
    """A stateless recognition request carrying an image inline."""

    image_base64: str = Field(description="Base64 image data, with or without a data URL prefix")
    language: str | None = Field(
        default=None, description="Tesseract language string such as 'kan+eng'"
    )
    correct: bool = Field(default=True, description="Repair recognition errors with the model")
    translate: bool = Field(default=False, description="Also translate the corrected text")
    target_language: str = Field(default="English", description="Translation target")


class OCRResponse(BaseModel):
    """The result of a stateless recognition request."""

    success: bool
    languages: list[DetectedLanguageOut]
    ocr_languages: str
    raw_text: str
    corrected_text: str
    translated_text: str | None = None
    confidence: float = Field(description="Mean per-word recognition confidence, 0 to 1")
    correction_applied: bool
    translation_applied: bool = False
    notes: list[str] = Field(default_factory=list)
    processing_time_ms: int


class TranslateRequest(BaseModel):
    """A standalone translation request."""

    text: str = Field(min_length=1)
    target_language: str = Field(default="English")


class TranslateResponse(BaseModel):
    """The result of a standalone translation request."""

    source_text: str
    translated_text: str
    target_language: str
    detected_languages: list[DetectedLanguageOut]
    applied: bool
    detail: str = ""


class AskRequest(BaseModel):
    """A question put to the indexed corpus."""

    question: str = Field(min_length=1)
    document_ids: list[int] | None = Field(
        default=None, description="Restrict retrieval to these documents"
    )
    top_k: int | None = Field(default=None, ge=1, le=20)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class SourceOut(BaseModel):
    """A retrieved chunk cited in support of an answer."""

    document_id: int
    chunk_id: int
    ordinal: int
    score: float
    text: str


class AskResponse(BaseModel):
    """An answer with the sources it was drawn from."""

    question: str
    answer: str
    grounded: bool
    best_score: float
    sources: list[SourceOut]


class ReindexRequest(BaseModel):
    """Options for rebuilding a document's index."""

    strategy: ChunkStrategy = ChunkStrategy.WORDS
    chunk_size: int | None = Field(default=None, ge=20, le=2000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)


class HealthResponse(BaseModel):
    """Readiness of the application and everything it depends on."""

    status: str
    tesseract_available: bool
    tesseract_version: str | None = None
    installed_languages: list[str]
    model_provider_configured: bool
    embedding_model_ready: bool
    document_count: int
    chunk_count: int


class DeleteResponse(BaseModel):
    """Confirmation that a document was removed."""

    id: int
    deleted: bool
