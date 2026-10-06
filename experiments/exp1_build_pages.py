"""Experiment 1, data preparation: render ground-truth pages and degrade them.

For each language, FLORES-200 dev sentences are grouped into pages of roughly
PAGE_WORDS words, rendered with the Nirmala UI font at the size a 200 dpi scan of
12 pt type would give, then degraded in six ways that stand in for the conditions
a scanned or photographed document arrives in.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LANGUAGES, REPO_ROOT, flores_sentences  # noqa: E402

PAGES_DIR = REPO_ROOT / "experiments" / "pages"
PAGES_PER_LANGUAGE = 6
PAGE_WORDS = 110          # about half an A4 page of 12 pt text
CONDITIONS = ["clean", "small", "noisy", "skewed", "jpeg", "photo"]
SEED = 20260911


def build_page_texts(code: str) -> list[str]:
    sentences = flores_sentences(code)
    pages: list[str] = []
    buffer: list[str] = []
    words = 0
    for sentence in sentences:
        buffer.append(sentence.strip())
        words += len(sentence.split())
        if words >= PAGE_WORDS:
            pages.append(" ".join(buffer))
            buffer, words = [], 0
            if len(pages) == PAGES_PER_LANGUAGE:
                break
    return pages


def degrade(clean: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    if condition == "clean":
        return clean
    if condition == "small":
        # A phone photo taken from too far away: about 90 dpi effective.
        return cv2.resize(clean, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)
    if condition == "noisy":
        blurred = cv2.GaussianBlur(clean, (3, 3), 0)
        noise = rng.normal(0, 25, blurred.shape)
        return np.clip(blurred.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    if condition == "skewed":
        return rotate(clean, 4.0)
    if condition == "jpeg":
        ok, encoded = cv2.imencode(".jpg", clean, [cv2.IMWRITE_JPEG_QUALITY, 20])
        return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if condition == "photo":
        # Skew, uneven lighting, sensor noise and mild blur together.
        rotated = rotate(clean, -3.0)
        height, width = rotated.shape
        gradient = np.linspace(0.55, 1.0, width, dtype=np.float64)[None, :]
        vertical = np.linspace(1.0, 0.75, height, dtype=np.float64)[:, None]
        lit = rotated.astype(np.float64) * gradient * vertical
        lit = cv2.GaussianBlur(lit, (5, 5), 0)
        noise = rng.normal(0, 12, lit.shape)
        return np.clip(lit + noise, 0, 255).astype(np.uint8)
    raise ValueError(condition)


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def main() -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    spec = []
    manifest = []
    for code in LANGUAGES:
        for index, text in enumerate(build_page_texts(code)):
            stem = f"{code}_{index:02d}"
            clean_path = PAGES_DIR / f"{stem}_clean.png"
            spec.append({"path": str(clean_path), "text": text})
            manifest.append({"code": code, "page": index, "stem": stem, "text": text})

    spec_path = PAGES_DIR / "render_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    script = REPO_ROOT / "experiments" / "render_pages.ps1"
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
         "-Spec", str(spec_path)],
        check=True,
    )

    rng = np.random.default_rng(SEED)
    for entry in manifest:
        clean = cv2.imread(str(PAGES_DIR / f"{entry['stem']}_clean.png"), cv2.IMREAD_GRAYSCALE)
        for condition in CONDITIONS[1:]:
            image = degrade(clean, condition, rng)
            cv2.imwrite(str(PAGES_DIR / f"{entry['stem']}_{condition}.png"), image)

    (PAGES_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"{len(manifest)} pages x {len(CONDITIONS)} conditions written to {PAGES_DIR}")


if __name__ == "__main__":
    main()
