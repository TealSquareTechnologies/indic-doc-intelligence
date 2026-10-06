"""Tests for translation behaviour when the provider is unavailable."""

from __future__ import annotations

import pytest

from app.core.errors import ValidationError
from app.documents import service
from app.translate.translator import MAX_INPUT_CHARACTERS, translate

SOURCE = "ಇದು ಒಂದು ಪರೀಕ್ಷಾ ವಾಕ್ಯ"


def test_failure_never_returns_the_source_text(isolated_settings):
    result = translate(SOURCE)

    assert result.applied is False
    assert result.text == ""
    assert SOURCE not in result.text
    assert result.detail


def test_failure_reports_a_reason(isolated_settings):
    result = translate(SOURCE)
    assert "CONSCIOUSNX_API_KEY" in result.detail


def test_target_language_is_preserved_on_failure(isolated_settings):
    result = translate(SOURCE, target_language="Hindi")
    assert result.target_language == "Hindi"
    assert result.text == ""


def test_empty_input_is_not_an_error(isolated_settings):
    result = translate("   ")
    assert result.applied is False
    assert result.text == ""


def test_target_language_defaults_to_english(isolated_settings):
    assert translate(SOURCE, target_language="").target_language == "English"


def test_oversized_input_is_rejected_before_calling_the_provider(isolated_settings):
    with pytest.raises(ValidationError, match="character limit"):
        translate("x" * (MAX_INPUT_CHARACTERS + 1))


def test_pipeline_leaves_translation_unset_when_it_fails(database):
    result = service.run_pipeline(
        SOURCE.encode("utf-8"),
        source_kind="text",
        apply_correction=False,
        apply_translation=True,
    )

    assert result.translated_text is None
    assert result.translation_applied is False
    assert any("Translation failed" in note for note in result.notes)
    assert result.raw_text == SOURCE


def test_pipeline_keeps_text_when_correction_fails(database):
    result = service.run_pipeline(
        SOURCE.encode("utf-8"),
        source_kind="text",
        apply_correction=True,
        apply_translation=False,
    )

    assert result.corrected_text == SOURCE
    assert result.correction_applied is False
