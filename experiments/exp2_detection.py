"""Experiment 2 (parts a and c): script-based language detection.

Part a: top-1 detection accuracy on every FLORES-200 dev sentence of each language.
Part c: two-script documents mixed at known ratios, to see how well the reported
share tracks the true share and where the 5% reporting threshold cuts in.
Part b (detection on recognised text) and part d (language strings) are in
exp2_ocr_languages.py because they need Tesseract output.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LANGUAGES, flores_sentences, save_json  # noqa: E402

from app.detect.detector import detect  # noqa: E402


def part_a() -> list[dict]:
    rows = []
    for code in LANGUAGES:
        sentences = flores_sentences(code)
        correct = 0
        secondary = {}
        for sentence in sentences:
            detected = detect(sentence)
            if detected and detected[0].code == code:
                correct += 1
            for item in detected[1:]:
                secondary[item.code] = secondary.get(item.code, 0) + 1
        rows.append({
            "code": code, "sentences": len(sentences), "top1_correct": correct,
            "top1_accuracy": round(correct / len(sentences), 4),
            "secondary_scripts": secondary,
        })
        print(rows[-1])
    return rows


def part_c() -> list[dict]:
    rng = random.Random(7)
    rows = []
    pairs = [("kan", "eng"), ("hin", "eng"), ("tam", "hin"), ("ben", "eng")]
    for primary, secondary in pairs:
        first = flores_sentences(primary)
        second = flores_sentences(secondary)
        for ratio in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
            for trial in range(5):
                pool_a = rng.sample(first, 20)
                pool_b = rng.sample(second, 20)
                # Build a document whose character budget splits at the ratio.
                budget = 3000
                text_a = " ".join(pool_a)[: int(budget * (1 - ratio))]
                text_b = " ".join(pool_b)[: int(budget * ratio)]
                text = text_a + " " + text_b
                detected = detect(text)
                shares = {item.code: item.share for item in detected}
                rows.append({
                    "primary": primary, "secondary": secondary, "ratio": ratio, "trial": trial,
                    "detected": [item.code for item in detected],
                    "share_primary": shares.get(primary, 0.0),
                    "share_secondary": shares.get(secondary, 0.0),
                })
    return rows


if __name__ == "__main__":
    save_json("exp2_detection_a.json", part_a())
    save_json("exp2_detection_c.json", part_c())
    print("done")
