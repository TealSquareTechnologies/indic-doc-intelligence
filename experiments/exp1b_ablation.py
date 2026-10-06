"""Experiment 1b: which preparation step helps and which hurts.

Each variant rebuilds the preparation pipeline from its parts (with the deskew fix in
place) and hands the result to the recogniser with the application's own preparation
switched off, so the recognition path is otherwise identical to experiment 1.

  grayscale      grayscale and upscaling only
  denoise        grayscale, upscaling, non-local means denoising (no binarisation)
  binarise       grayscale, upscaling, deskew, Otsu binarisation (no denoising)
  adaptive       grayscale, upscaling, denoising, deskew, adaptive Gaussian threshold
                 instead of Otsu (a candidate fix for uneven lighting)
The full pipeline (denoise + deskew + Otsu) is the preparation arm of experiment 1.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, cer, save_json, wer  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.ocr import preprocess  # noqa: E402
from app.ocr.engine import extract_text  # noqa: E402

PAGES_DIR = REPO_ROOT / "experiments" / "pages"
CONDITIONS = ["clean", "noisy", "skewed", "photo"]
PAGES_PER_LANGUAGE = 3
VARIANTS = ["grayscale", "denoise", "binarise", "adaptive"]


def prepare(image_bytes: bytes, variant: str) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    gray = preprocess._upscale_if_small(preprocess._to_grayscale(decoded))
    if variant == "grayscale":
        return gray
    if variant == "denoise":
        return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    if variant == "binarise":
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        straight = preprocess._deskew(gray, ink)
        _, binary = cv2.threshold(straight, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    if variant == "adaptive":
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        ink = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 51, 15)
        straight = preprocess._deskew(denoised, ink)
        return cv2.adaptiveThreshold(straight, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, 15)
    raise ValueError(variant)


def main() -> None:
    manifest = json.loads((PAGES_DIR / "manifest.json").read_text(encoding="utf-8"))
    settings = get_settings()
    settings.preprocess_images = False
    rows = []
    for entry in manifest:
        if entry["page"] >= PAGES_PER_LANGUAGE:
            continue
        for condition in CONDITIONS:
            data = (PAGES_DIR / f"{entry['stem']}_{condition}.png").read_bytes()
            for variant in VARIANTS:
                started = time.perf_counter()
                prepared = prepare(data, variant)
                ok, encoded = cv2.imencode(".png", prepared)
                result = extract_text(encoded.tobytes(), language=entry["code"])
                elapsed = time.perf_counter() - started
                rows.append({
                    "code": entry["code"], "page": entry["page"], "condition": condition,
                    "variant": variant, "cer": cer(entry["text"], result.text),
                    "wer": wer(entry["text"], result.text), "confidence": result.confidence,
                    "seconds": round(elapsed, 3),
                })
            print(entry["stem"], condition,
                  {r["variant"]: round(100 * r["cer"], 1) for r in rows[-len(VARIANTS):]}, flush=True)
    print("saved", save_json("exp1b_ablation.json", rows))


if __name__ == "__main__":
    main()
