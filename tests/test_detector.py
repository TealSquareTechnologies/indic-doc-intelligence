"""Tests for script based language detection."""

from __future__ import annotations

from app.detect.detector import detect, detect_names, suggest_ocr_languages


def test_empty_text_detects_nothing():
    assert detect("") == []
    assert detect("   \n  ") == []


def test_single_script_is_identified():
    detected = detect("This is an English sentence.")
    assert [item.code for item in detected] == ["eng"]
    assert detected[0].share == 1.0


def test_mixed_scripts_are_ordered_by_share():
    text = "ಕನ್ನಡ ಪಠ್ಯ ಇಲ್ಲಿದೆ ಮತ್ತು ಇನ್ನಷ್ಟು ಕನ್ನಡ hello"
    names = detect_names(text)
    assert names[0] == "Kannada"
    assert "English" in names


def test_stray_characters_below_the_threshold_are_dropped():
    text = "ಕನ್ನಡ" * 40 + "x"
    assert detect_names(text) == ["Kannada"]


def test_threshold_can_be_relaxed():
    text = "ಕನ್ನಡ" * 40 + "x"
    assert "English" in detect_names(text, min_share=0.0)


def test_shares_sum_to_one_for_a_two_script_document():
    detected = detect("ಕನ್ನಡ hello", min_share=0.0)
    assert round(sum(item.share for item in detected), 4) == 1.0


def test_digits_and_punctuation_are_ignored():
    detected = detect("12345 !@#$ ....")
    assert detected == []


def test_devanagari_carries_a_note_about_shared_use():
    detected = detect("यह हिंदी पाठ है")
    assert detected[0].code == "hin"
    assert "Marathi" in detected[0].note


def test_short_text_falls_back_to_the_dominant_script():
    detected = detect("ಕ", min_share=0.99)
    assert [item.code for item in detected] == ["kan"]


def test_suggest_ocr_languages_builds_a_tesseract_string():
    suggestion = suggest_ocr_languages("ಕನ್ನಡ ಪಠ್ಯ hello world")
    assert set(suggestion.split("+")) == {"kan", "eng"}
