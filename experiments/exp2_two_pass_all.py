"""Experiment 2d addendum: a two-pass scheme whose first pass uses all ten languages.

The first pass in exp2_ocr_languages.py used the application's default string, and
the detector then suggested whatever scripts the garbage output happened to contain.
Here the first pass names every supported language, detection narrows it, and the
second pass runs with the narrowed string. Rows are appended to exp2_languages_d.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LANGUAGES, REPO_ROOT, cer, load_json, save_json  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.detect.detector import suggest_ocr_languages  # noqa: E402
from app.ocr.engine import extract_text  # noqa: E402

PAGES_DIR = REPO_ROOT / "experiments" / "pages"
ALL_TEN = "+".join(LANGUAGES)


def main() -> None:
    rows = [r for r in load_json("exp2_languages_d.json") if r["config"] != "two_pass_all"]
    manifest = json.loads((PAGES_DIR / "manifest.json").read_text(encoding="utf-8"))
    get_settings().preprocess_images = True
    for entry in manifest:
        if entry["page"] >= 3:
            continue
        data = (PAGES_DIR / f"{entry['stem']}_clean.png").read_bytes()
        started = time.perf_counter()
        first = extract_text(data, language=ALL_TEN)
        suggested = suggest_ocr_languages(first.text)
        second = extract_text(data, language=suggested)
        elapsed = time.perf_counter() - started
        rows.append({
            "code": entry["code"], "page": entry["page"], "config": "two_pass_all",
            "language_string": suggested, "cer": cer(entry["text"], second.text),
            "confidence": second.confidence, "seconds": round(elapsed, 3),
        })
        print(entry["stem"], suggested, round(100 * rows[-1]["cer"], 1), flush=True)
    save_json("exp2_languages_d.json", rows)


if __name__ == "__main__":
    main()
