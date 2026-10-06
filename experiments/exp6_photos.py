"""Experiment 6: two real photographs of Kannada documents through the whole pipeline.

There is no ground truth for these, so the record is qualitative: what the
recogniser produced, how confident it was, what the detector reported, and what
the correction and translation stages made of it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import route_llm_through_ollama, save_json  # noqa: E402

route_llm_through_ollama()
from app.core.config import get_settings  # noqa: E402
from app.documents.service import run_pipeline  # noqa: E402

PHOTOS = Path(__file__).resolve().parent / "data" / "photos"


def main() -> None:
    settings = get_settings()
    rows = []
    for path in sorted(PHOTOS.glob("*.jpg")):
        for preprocess in (True, False):
            settings.preprocess_images = preprocess
            started = time.perf_counter()
            result = run_pipeline(path.read_bytes(), source_kind="image", language="kan+eng",
                                  apply_correction=True, apply_translation=True)
            rows.append({
                "photo": path.name, "preprocess": preprocess,
                "confidence": result.confidence, "elapsed_ms": result.elapsed_ms,
                "detected": [(d.code, d.share) for d in result.detected],
                "raw_text": result.raw_text, "corrected_text": result.corrected_text,
                "translated_text": result.translated_text, "notes": result.notes,
                "seconds": round(time.perf_counter() - started, 1),
            })
            print(path.name, preprocess, result.confidence, rows[-1]["detected"],
                  len(result.raw_text), "chars", flush=True)
    print("saved", save_json("exp6_photos.json", rows))


if __name__ == "__main__":
    main()
