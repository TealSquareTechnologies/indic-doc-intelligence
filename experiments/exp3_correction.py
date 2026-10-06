"""Experiment 3: does model-based correction repair recognition errors?

The recognised text from experiment 1 (preprocessing on) is passed through the
application's correct() function for three image conditions, and the character
error rate against the ground truth is compared before and after.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cer, load_json, route_llm_through_ollama, save_json, wer  # noqa: E402

route_llm_through_ollama()
from app.correct.corrector import correct  # noqa: E402

CONDITIONS = ["clean", "noisy", "small"]
PAGES_PER_LANGUAGE = 4
PAGES_DIR = Path(__file__).resolve().parent / "pages"


def main() -> None:
    import json
    manifest = {(m["code"], m["page"]): m["text"]
                for m in json.loads((PAGES_DIR / "manifest.json").read_text(encoding="utf-8"))}
    rows = []
    for row in load_json("exp1_ocr.json"):
        if not row["preprocess"] or row["condition"] not in CONDITIONS:
            continue
        if row["page"] >= PAGES_PER_LANGUAGE:
            continue
        truth = manifest[(row["code"], row["page"])]
        started = time.perf_counter()
        result = correct(row["text"])
        elapsed = time.perf_counter() - started
        rows.append({
            "code": row["code"], "page": row["page"], "condition": row["condition"],
            "cer_before": row["cer"], "cer_after": cer(truth, result.text),
            "wer_before": row["wer"], "wer_after": wer(truth, result.text),
            "applied": result.applied, "chars_in": len(row["text"]),
            "chars_out": len(result.text), "seconds": round(elapsed, 2),
            "corrected": result.text,
        })
        print(f"{row['code']} p{row['page']} {row['condition']}: "
              f"{row['cer']:.3f} -> {rows[-1]['cer_after']:.3f} ({elapsed:.0f}s)", flush=True)
    print("saved", save_json("exp3_correction.json", rows))


if __name__ == "__main__":
    main()
