"""Language detection by script."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.core.config import get_settings
from app.ocr.languages import Language, language_for_codepoint


@dataclass(frozen=True)
class DetectedLanguage:
    """One language found in a piece of text, with the share of characters it covers."""

    code: str
    name: str
    script: str
    share: float
    character_count: int
    note: str = ""


def _count_scripts(text: str) -> tuple[Counter[str], dict[str, Language], int]:
    counts: Counter[str] = Counter()
    languages: dict[str, Language] = {}
    total = 0

    for character in text:
        # Whitespace, digits, punctuation and control characters belong to no script.
        if character.isspace() or character.isdigit() or not character.isprintable():
            continue

        language = language_for_codepoint(ord(character))
        if language is None:
            continue

        counts[language.code] += 1
        languages[language.code] = language
        total += 1

    return counts, languages, total


def detect(text: str, min_share: float | None = None) -> list[DetectedLanguage]:
    """Identify the languages present in text, most prominent first."""
    if not text or not text.strip():
        return []

    threshold = get_settings().detection_min_share if min_share is None else min_share
    counts, languages, total = _count_scripts(text)
    if total == 0:
        return []

    detected = [
        DetectedLanguage(
            code=code,
            name=languages[code].name,
            script=languages[code].script,
            share=round(count / total, 4),
            character_count=count,
            note=languages[code].note,
        )
        for code, count in counts.items()
        if count / total >= threshold
    ]

    # Fall back to the most common script so short strings still report something.
    if not detected:
        code, count = counts.most_common(1)[0]
        detected = [
            DetectedLanguage(
                code=code,
                name=languages[code].name,
                script=languages[code].script,
                share=round(count / total, 4),
                character_count=count,
                note=languages[code].note,
            )
        ]

    return sorted(detected, key=lambda item: (-item.share, item.name))


def detect_names(text: str, min_share: float | None = None) -> list[str]:
    """Return just the language names, most prominent first."""
    return [item.name for item in detect(text, min_share)]


def suggest_ocr_languages(text: str, min_share: float | None = None) -> str:
    """Build a Tesseract language string from what a first pass recovered."""
    detected = detect(text, min_share)
    if not detected:
        return get_settings().default_ocr_languages
    return "+".join(item.code for item in detected)
