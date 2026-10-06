"""Supported languages."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.errors import ValidationError


@dataclass(frozen=True)
class Language:
    """A language the platform can extract and identify."""

    code: str
    name: str
    script: str
    ranges: tuple[tuple[int, int], ...]
    note: str = field(default="")


LANGUAGES: tuple[Language, ...] = (
    Language(
        code="eng",
        name="English",
        script="Latin",
        ranges=((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)),
    ),
    Language(
        code="hin",
        name="Hindi",
        script="Devanagari",
        ranges=((0x0900, 0x097F),),
        note="Devanagari is also used by Marathi, Nepali and Sanskrit.",
    ),
    Language(code="ben", name="Bengali", script="Bengali", ranges=((0x0980, 0x09FF),)),
    Language(
        code="pan",
        name="Punjabi",
        script="Gurmukhi",
        ranges=((0x0A00, 0x0A7F),),
    ),
    Language(code="guj", name="Gujarati", script="Gujarati", ranges=((0x0A80, 0x0AFF),)),
    Language(code="ori", name="Odia", script="Odia", ranges=((0x0B00, 0x0B7F),)),
    Language(code="tam", name="Tamil", script="Tamil", ranges=((0x0B80, 0x0BFF),)),
    Language(code="tel", name="Telugu", script="Telugu", ranges=((0x0C00, 0x0C7F),)),
    Language(code="kan", name="Kannada", script="Kannada", ranges=((0x0C80, 0x0CFF),)),
    Language(code="mal", name="Malayalam", script="Malayalam", ranges=((0x0D00, 0x0D7F),)),
)

_BY_CODE: dict[str, Language] = {language.code: language for language in LANGUAGES}


def supported_codes() -> list[str]:
    """Every Tesseract code the platform knows about."""
    return [language.code for language in LANGUAGES]


def by_code(code: str) -> Language:
    """Look up a language by its Tesseract code."""
    try:
        return _BY_CODE[code]
    except KeyError:
        raise ValidationError(
            f"Unknown language code '{code}'. Supported codes: {', '.join(supported_codes())}"
        ) from None


def language_for_codepoint(code_point: int) -> Language | None:
    """Return the language whose script covers this code point, if any."""
    for language in LANGUAGES:
        for start, end in language.ranges:
            if start <= code_point <= end:
                return language
    return None


def parse_language_string(value: str) -> list[Language]:
    """Validate a Tesseract language string such as 'kan+eng' into language entries."""
    codes = [part.strip() for part in value.split("+") if part.strip()]
    if not codes:
        raise ValidationError("A language string must name at least one language code.")

    seen: set[str] = set()
    languages: list[Language] = []
    for code in codes:
        if code in seen:
            continue
        seen.add(code)
        languages.append(by_code(code))
    return languages


def normalise_language_string(value: str) -> str:
    """Validate and canonicalise a language string, removing duplicates."""
    return "+".join(language.code for language in parse_language_string(value))
