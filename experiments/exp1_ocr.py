"""Experiment 1: recognition accuracy across ten languages and six image conditions,
with and without the image preparation step.

Every page is recognised with the correct single-language string, so that this
experiment isolates the recogniser and the preprocessing. Language selection is
the subject of experiment 2.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, cer, save_json, wer  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.ocr.engine import extract_text  # noqa: E402

PAGES_DIR = REPO_ROOT / "experiments" / "pages"
CONDITIONS = ["clean", "small", "noisy", "skewed", "jpeg", "photo"]


def main() -> None:
    # Usage: exp1_ocr.py [output name] [--preprocess-only]
    output = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "exp1_ocr.json"
    arms = (True,) if "--preprocess-only" in sys.argv else (True, False)
    manifest = json.loads((PAGES_DIR / "manifest.json").read_text(encoding="utf-8"))
    settings = get_settings()
    rows = []
    total = len(manifest) * len(CONDITIONS) * len(arms)
    done = 0
    for entry in manifest:
        for condition in CONDITIONS:
            data = (PAGES_DIR / f"{entry['stem']}_{condition}.png").read_bytes()
            for preprocess in arms:
                settings.preprocess_images = preprocess
                started = time.perf_counter()
                result = extract_text(data, language=entry["code"])
                elapsed = time.perf_counter() - started
                rows.append({
                    "code": entry["code"],
                    "page": entry["page"],
                    "condition": condition,
                    "preprocess": preprocess,
                    "cer": cer(entry["text"], result.text),
                    "wer": wer(entry["text"], result.text),
                    "confidence": result.confidence,
                    "seconds": round(elapsed, 3),
                    "text": result.text,
                })
                done += 1
                if done % 20 == 0:
                    print(f"{done}/{total}  {entry['stem']} {condition} pre={preprocess} "
                          f"cer={rows[-1]['cer']:.3f} {elapsed:.1f}s", flush=True)
    path = save_json(output, rows)
    print("saved", path)


if __name__ == "__main__":
    main()
