"""Text extraction through Tesseract."""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings
from app.core.errors import DependencyUnavailableError, ValidationError
from app.core.logging import get_logger
from app.ocr.languages import normalise_language_string
from app.ocr.preprocess import prepare_image
from app.ocr.tessdata import configure_environment

logger = get_logger(__name__)

_binary_configured = False


@dataclass(frozen=True)
class ExtractionResult:
    """Text recovered from one image, with the mean per-word confidence."""

    text: str
    confidence: float
    languages_requested: str


# Where installers place the binary on Windows when it is not added to PATH.
_WINDOWS_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def _discover_binary() -> str | None:
    """Find the Tesseract binary without any configuration."""
    found = shutil.which("tesseract")
    if found:
        return found

    for candidate in _WINDOWS_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _configure_binary() -> None:
    """Point pytesseract at the binary and the project's language data, once per process."""
    global _binary_configured
    if _binary_configured:
        return

    configured = get_settings().tesseract_cmd.strip()
    if configured:
        path = Path(configured)
        if not path.exists():
            raise DependencyUnavailableError(
                f"TESSERACT_CMD points at '{configured}', which does not exist."
            )
        pytesseract.pytesseract.tesseract_cmd = str(path)
    else:
        discovered = _discover_binary()
        if discovered:
            pytesseract.pytesseract.tesseract_cmd = discovered

    # Prefer the project's own language data when present.
    directory = configure_environment()
    if directory:
        logger.debug("Using language data from %s", directory)

    _binary_configured = True


def tesseract_version() -> str | None:
    """Return the installed Tesseract version, or None when it cannot be reached."""
    try:
        _configure_binary()
        return str(pytesseract.get_tesseract_version())
    except Exception:  # noqa: BLE001 - any failure means the binary is unusable
        return None


def installed_languages() -> list[str]:
    """Return the traineddata packs available to the installed binary."""
    try:
        _configure_binary()
        return sorted(pytesseract.get_languages(config=""))
    except Exception:  # noqa: BLE001 - treat any failure as nothing installed
        return []


def _mean_confidence(image: Image.Image, language: str) -> float:
    """Average the per-word confidences Tesseract reports for a page."""
    try:
        data = pytesseract.image_to_data(
            image, lang=language, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractError:
        return 0.0

    scores = [float(value) for value in data.get("conf", []) if float(value) >= 0]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores) / 100, 4)


def extract_text(image_bytes: bytes, language: str | None = None) -> ExtractionResult:
    """Extract text from image bytes using the given Tesseract language string."""
    if not image_bytes:
        raise ValidationError("No image data was supplied.")

    settings = get_settings()
    requested = normalise_language_string(language or settings.default_ocr_languages)
    _configure_binary()

    missing = set(requested.split("+")) - set(installed_languages())
    if missing:
        raise DependencyUnavailableError(
            "Tesseract is missing traineddata for: "
            f"{', '.join(sorted(missing))}. Install the matching language packs."
        )

    try:
        if settings.preprocess_images:
            image = Image.fromarray(prepare_image(image_bytes))
        else:
            image = Image.open(io.BytesIO(image_bytes))
    except (UnidentifiedImageError, ValueError) as exc:
        raise ValidationError(f"The uploaded file is not a readable image: {exc}") from exc

    with image:
        try:
            text = pytesseract.image_to_string(image, lang=requested)
        except pytesseract.TesseractNotFoundError as exc:
            raise DependencyUnavailableError(
                "The Tesseract binary was not found. Install it, or set TESSERACT_CMD."
            ) from exc
        except pytesseract.TesseractError as exc:
            raise DependencyUnavailableError(f"Tesseract failed: {exc}") from exc

        confidence = _mean_confidence(image, requested)

    return ExtractionResult(
        text=text.strip(),
        confidence=confidence,
        languages_requested=requested,
    )
