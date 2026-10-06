"""Tests for image preparation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ocr import preprocess


def text_like_page(height: int = 600, width: int = 800) -> np.ndarray:
    """A white page with black horizontal bars standing in for lines of text."""
    page = np.full((height, width), 255, dtype=np.uint8)
    for top in range(80, height - 80, 40):
        cv2.rectangle(page, (100, top), (width - 100, top + 14), 0, -1)
    return page


def encode(image: np.ndarray, extension: str = ".png") -> bytes:
    ok, buffer = cv2.imencode(extension, image)
    assert ok
    return buffer.tobytes()


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), borderValue=255)


def test_output_is_a_binary_grayscale_image():
    result = preprocess.prepare_image(encode(text_like_page()))
    assert result.ndim == 2
    assert set(np.unique(result)).issubset({0, 255})


def test_colour_images_are_converted():
    colour = cv2.cvtColor(text_like_page(), cv2.COLOR_GRAY2BGR)
    result = preprocess.prepare_image(encode(colour))
    assert result.ndim == 2


def test_images_with_an_alpha_channel_are_converted():
    rgba = cv2.cvtColor(text_like_page(), cv2.COLOR_GRAY2BGRA)
    result = preprocess.prepare_image(encode(rgba))
    assert result.ndim == 2


def test_small_images_are_upscaled_to_the_minimum_height():
    small = text_like_page(height=500, width=400)
    result = preprocess.prepare_image(encode(small))
    assert result.shape[0] >= preprocess.MIN_HEIGHT_PIXELS


def test_upscaling_is_capped():
    tiny = text_like_page(height=100, width=200)
    result = preprocess.prepare_image(encode(tiny))
    assert result.shape[0] <= 100 * preprocess.MAX_SCALE + 1


def test_large_images_keep_their_size():
    large = text_like_page(height=1400, width=1000)
    result = preprocess.prepare_image(encode(large))
    assert result.shape == large.shape


@pytest.mark.parametrize("degrees", [2.0, -3.5, 6.0])
def test_skew_is_estimated_within_half_a_degree(degrees):
    page = rotate(text_like_page(height=1200, width=900), degrees)
    _, ink = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    estimate = preprocess._estimate_skew_degrees(ink)
    assert abs(estimate - degrees) < 0.5


def test_deskew_straightens_a_rotated_page():
    page = rotate(text_like_page(height=1200, width=900), 5.0)
    _, ink = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    straightened = preprocess._deskew(page, ink)
    _, ink_after = cv2.threshold(straightened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    assert abs(preprocess._estimate_skew_degrees(ink_after)) < 0.5


def test_rotations_beyond_the_limit_are_left_alone():
    page = rotate(text_like_page(height=1200, width=900), 30.0)
    _, ink = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    assert preprocess._deskew(page, ink) is page


def test_nearly_blank_pages_report_no_skew():
    blank = np.zeros((500, 500), dtype=np.uint8)
    assert preprocess._estimate_skew_degrees(blank) == 0.0


def test_undecodable_bytes_are_rejected():
    with pytest.raises(ValueError):
        preprocess.prepare_image(b"this is not an image")
