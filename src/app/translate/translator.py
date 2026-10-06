"""Translation of extracted text through a language model."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import AppError, ValidationError
from app.core.llm import complete
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_INPUT_CHARACTERS = 20000

SYSTEM_PROMPT = (
    "You are a translator. You return only the translation, with no preamble, notes or "
    "alternative renderings."
)

USER_PROMPT = """Translate the text below into {target}.

Rules:
1. Produce natural, fluent {target} rather than a word for word rendering.
2. Preserve paragraph breaks and the order of the content.
3. Leave proper nouns, numbers, dates and identifiers unchanged.
4. Never add content that is not present in the source.
5. Return the translation only.

Text:
{text}"""


@dataclass(frozen=True)
class TranslationResult:
    """The outcome of a translation pass."""

    text: str
    target_language: str
    applied: bool
    detail: str = ""


def translate(text: str, target_language: str = "English") -> TranslationResult:
    """Translate text into the target language, defaulting to English."""
    target = (target_language or "English").strip() or "English"

    if not text or not text.strip():
        return TranslationResult(
            text="", target_language=target, applied=False, detail="There was no text to translate."
        )

    if len(text) > MAX_INPUT_CHARACTERS:
        raise ValidationError(
            f"Text is {len(text)} characters, above the {MAX_INPUT_CHARACTERS} character limit."
        )

    try:
        translated = complete(
            USER_PROMPT.format(target=target, text=text),
            system=SYSTEM_PROMPT,
            temperature=0.2,
        )
    except AppError as exc:
        logger.warning("Translation failed: %s", exc.message)
        return TranslationResult(
            text="", target_language=target, applied=False, detail=exc.message
        )

    if not translated:
        return TranslationResult(
            text="", target_language=target, applied=False, detail="The model returned no text."
        )

    return TranslationResult(text=translated, target_language=target, applied=True)
