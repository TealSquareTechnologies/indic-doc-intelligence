"""Experiment 2, parts b and d: language selection for the recogniser.

Part d compares four ways of choosing the Tesseract language string for a page:
the correct single language (an oracle), the application's default string, a
string naming all ten supported languages, and the two-pass scheme the detector
makes possible (recognise with the default string, detect the scripts, recognise
again with the suggested string).

Part b reads the recognised text from experiment 1 and checks whether script
detection still names the right language on noisy output.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LANGUAGES, REPO_ROOT, cer, load_json, save_json  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.detect.detector import detect, suggest_ocr_languages  # noqa: E402
from app.ocr.engine import extract_text  # noqa: E402

PAGES_DIR = REPO_ROOT / "experiments" / "pages"
PAGES_PER_LANGUAGE = 3
ALL_TEN = "+".join(LANGUAGES)


def part_d() -> list[dict]:
    manifest = json.loads((PAGES_DIR / "manifest.json").read_text(encoding="utf-8"))
    settings = get_settings()
    settings.preprocess_images = True
    default = settings.default_ocr_languages
    rows = []
    for entry in manifest:
        if entry["page"] >= PAGES_PER_LANGUAGE:
            continue
        data = (PAGES_DIR / f"{entry['stem']}_clean.png").read_bytes()
        configs = {"oracle": entry["code"], "default": default, "all_ten": ALL_TEN}
        outputs = {}
        for label, language in configs.items():
            started = time.perf_counter()
            result = extract_text(data, language=language)
            elapsed = time.perf_counter() - started
            outputs[label] = result.text
            rows.append({
                "code": entry["code"], "page": entry["page"], "config": label,
                "language_string": language, "cer": cer(entry["text"], result.text),
                "confidence": result.confidence, "seconds": round(elapsed, 3),
            })
        # Two-pass: detect on the default-string output, then recognise again.
        started = time.perf_counter()
        suggested = suggest_ocr_languages(outputs["default"])
        second = extract_text(data, language=suggested)
        elapsed = time.perf_counter() - started
        first_pass_seconds = next(r["seconds"] for r in rows
                                  if r["code"] == entry["code"] and r["page"] == entry["page"]
                                  and r["config"] == "default")
        rows.append({
            "code": entry["code"], "page": entry["page"], "config": "two_pass",
            "language_string": suggested, "cer": cer(entry["text"], second.text),
            "confidence": second.confidence,
            "seconds": round(elapsed + first_pass_seconds, 3),
        })
        print(entry["stem"], {r["config"]: round(r["cer"], 3) for r in rows[-4:]},
              "suggested:", suggested, flush=True)
    return rows


def part_b() -> list[dict]:
    rows = []
    for row in load_json("exp1_ocr.json"):
        if not row["preprocess"]:
            continue
        detected = detect(row["text"])
        rows.append({
            "code": row["code"], "condition": row["condition"], "cer": row["cer"],
            "detected": [item.code for item in detected],
            "top1": detected[0].code if detected else None,
            "top1_share": detected[0].share if detected else 0.0,
            "correct": bool(detected) and detected[0].code == row["code"],
        })
    return rows


if __name__ == "__main__":
    save_json("exp2_languages_d.json", part_d())
    save_json("exp2_detection_b.json", part_b())
    print("done")
