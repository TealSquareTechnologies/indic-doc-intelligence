"""Tests for the language table and language string parsing."""

from __future__ import annotations

import pytest

from app.core.errors import ValidationError
from app.ocr import languages


def test_every_language_has_a_unique_code():
    codes = [language.code for language in languages.LANGUAGES]
    assert len(codes) == len(set(codes))


def test_script_ranges_do_not_overlap():
    seen: list[tuple[int, int, str]] = []
    for language in languages.LANGUAGES:
        for start, end in language.ranges:
            for other_start, other_end, other_code in seen:
                assert end < other_start or start > other_end, (
                    f"{language.code} overlaps {other_code}"
                )
            seen.append((start, end, language.code))


def test_parse_language_string_accepts_combinations():
    parsed = languages.parse_language_string("kan+eng")
    assert [language.code for language in parsed] == ["kan", "eng"]


def test_parse_language_string_removes_duplicates():
    assert languages.normalise_language_string("eng+kan+eng") == "eng+kan"


def test_parse_language_string_rejects_unknown_codes():
    with pytest.raises(ValidationError, match="Unknown language code"):
        languages.parse_language_string("kan+xyz")


def test_parse_language_string_rejects_empty_input():
    with pytest.raises(ValidationError):
        languages.parse_language_string("  ")


@pytest.mark.parametrize(
    ("character", "expected"),
    [
        ("A", "eng"),
        ("ಕ", "kan"),
        ("த", "tam"),
        ("తె", "tel"),
        ("മ", "mal"),
        ("क", "hin"),
        ("ব", "ben"),
        ("ગ", "guj"),
        ("ਪ", "pan"),
        ("ଓ", "ori"),
    ],
)
def test_language_for_codepoint(character, expected):
    found = languages.language_for_codepoint(ord(character[0]))
    assert found is not None
    assert found.code == expected


def test_language_for_codepoint_returns_none_outside_known_scripts():
    assert languages.language_for_codepoint(ord("漢")) is None
