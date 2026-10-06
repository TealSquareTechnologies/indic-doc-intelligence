"""Cleanup of recognised text through a language model."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import AppError
from app.core.llm import complete
from app.core.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You repair text produced by optical character recognition. You never add, remove "
    "or reinterpret content, and you never explain what you changed."
)

USER_PROMPT = """Correct the recognition errors in the text below.

Rules:
1. Fix only characters that were misrecognised. Preserve the original wording.
2. Keep the original language. Do not translate.
3. Preserve line breaks, spacing, indentation and the order of the content.
4. Never invent words, sentences or details that are not in the input.
5. Where a passage is too damaged to recover, leave it as it is.
6. Return the corrected text only, with no commentary or formatting markers.

Text:
{text}"""


@dataclass(frozen=True)
class CorrectionResult:
    """The outcome of a correction pass."""

    text: str
    applied: bool
    detail: str = ""


def correct(text: str) -> CorrectionResult:
    """Repair recognition errors in text."""
    if not text or not text.strip():
        return CorrectionResult(text="", applied=False, detail="There was no text to correct.")

    try:
        corrected = complete(USER_PROMPT.format(text=text), system=SYSTEM_PROMPT, temperature=0.1)
    except AppError as exc:
        logger.warning("Correction skipped: %s", exc.message)
        return CorrectionResult(text=text, applied=False, detail=exc.message)

    if not corrected:
        return CorrectionResult(text=text, applied=False, detail="The model returned no text.")

    return CorrectionResult(text=corrected, applied=True)
