"""Image preparation that runs before text extraction."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)

# Below this height, upscaling helps Tesseract resolve small glyph detail.
MIN_HEIGHT_PIXELS = 1000
MAX_SCALE = 3.0
MAX_DESKEW_DEGREES = 15.0


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _upscale_if_small(image: np.ndarray) -> np.ndarray:
    height = image.shape[0]
    if height >= MIN_HEIGHT_PIXELS:
        return image

    scale = min(MIN_HEIGHT_PIXELS / height, MAX_SCALE)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _estimate_skew_degrees(binary: np.ndarray) -> float:
    """Estimate page rotation from the minimum area rectangle around the ink."""
    coordinates = np.column_stack(np.where(binary > 0))
    if coordinates.shape[0] < 100:
        return 0.0

    angle = cv2.minAreaRect(coordinates.astype(np.float32))[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return float(angle)


def _deskew(image: np.ndarray, binary: np.ndarray) -> np.ndarray:
    angle = _estimate_skew_degrees(binary)
    # Large estimates usually mean the heuristic failed rather than a badly rotated page.
    if abs(angle) < 0.25 or abs(angle) > MAX_DESKEW_DEGREES:
        return image

    height, width = image.shape[:2]
    centre = (width / 2, height / 2)
    # getRotationMatrix2D rotates counter-clockwise by a positive angle, and the
    # estimate above is the page's own counter-clockwise tilt, so undo it.
    matrix = cv2.getRotationMatrix2D(centre, -angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def prepare_image(image_bytes: bytes) -> np.ndarray:
    """Decode and clean an image, returning a grayscale array ready for extraction."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("The uploaded data could not be decoded as an image.")

    grayscale = _to_grayscale(decoded)

    try:
        scaled = _upscale_if_small(grayscale)
        denoised = cv2.fastNlMeansDenoising(scaled, None, h=10, templateWindowSize=7,
                                            searchWindowSize=21)
        # The inverted copy is only used to locate the ink.
        _, ink = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        straightened = _deskew(denoised, ink)
        _, binary = cv2.threshold(straightened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except cv2.error as exc:
        logger.warning("Image cleanup failed, using the decoded image as is: %s", exc)
        return grayscale
