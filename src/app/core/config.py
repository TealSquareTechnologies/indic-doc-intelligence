"""Application settings, loaded once from the environment and a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]

# The value .env.example ships with. Treated as no key at all.
PLACEHOLDER_API_KEY = "your-consciousnx-api-key-here"


class Settings(BaseSettings):
    """Every tunable value in the application, in one place."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Language model provider
    consciousnx_api_key: str = Field(default="", alias="CONSCIOUSNX_API_KEY")
    consciousnx_base_url: str = Field(
        default="https://backend.consciousnx.com/raw/v1", alias="CONSCIOUSNX_BASE_URL"
    )
    consciousnx_model: str = Field(default="cx-medium", alias="CONSCIOUSNX_MODEL")
    # The provider can be slow. The interface shows elapsed time and a stop control.
    llm_timeout_seconds: float = Field(default=1200.0, alias="LLM_TIMEOUT_SECONDS")
    # A retry repeats the full timeout, so none by default.
    llm_max_retries: int = Field(default=0, alias="LLM_MAX_RETRIES")

    # HTTP server
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    api_key: str = Field(default="", alias="API_KEY")
    open_browser: bool = Field(default=False, alias="OPEN_BROWSER")

    # Optical character recognition
    tesseract_cmd: str = Field(default="", alias="TESSERACT_CMD")
    tessdata_dir: str = Field(default="tessdata", alias="TESSDATA_DIR")
    default_ocr_languages: str = Field(default="eng+kan+tam+tel", alias="DEFAULT_OCR_LANGUAGES")
    preprocess_images: bool = Field(default=True, alias="PREPROCESS_IMAGES")
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")

    # Pages without a text layer are rendered at this resolution and recognised.
    pdf_render_dpi: int = Field(default=200, alias="PDF_RENDER_DPI")
    max_pdf_pages: int = Field(default=50, alias="MAX_PDF_PAGES")

    # Storage
    database_path: str = Field(default="data/app.db", alias="DATABASE_PATH")

    # Retrieval
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    chunk_size: int = Field(default=250, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=3, alias="RETRIEVAL_TOP_K")
    similarity_threshold: float = Field(default=0.20, alias="SIMILARITY_THRESHOLD")

    # Language detection
    detection_min_share: float = Field(default=0.05, alias="DETECTION_MIN_SHARE")

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_below_size(cls, value: int, info: ValidationInfo) -> int:
        size = info.data.get("chunk_size", 250)
        if value >= size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return max(value, 0)

    @property
    def has_provider_key(self) -> bool:
        """Whether a real provider key is set, ignoring the shipped placeholder."""
        key = self.consciousnx_api_key.strip()
        return bool(key) and key != PLACEHOLDER_API_KEY

    @property
    def database_file(self) -> Path:
        """Absolute path to the SQLite file. Creates nothing on access."""
        path = Path(self.database_path)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def browser_url(self) -> str:
        host = "localhost" if self.host in {"0.0.0.0", "127.0.0.1", "::"} else self.host
        return f"http://{host}:{self.port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
