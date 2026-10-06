"""Experiment 4: translation quality of the 4B model on clean text.

Forty FLORES-200 dev sentences per Indic language are translated into English with
the application's own translate() function and scored against the English side of
the corpus with chrF and BLEU. A smaller reverse run, English into Hindi and Kannada,
checks the direction users take when they want a document in their own language.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LANGUAGES, chrf, flores_sentences, route_llm_through_ollama, save_json  # noqa: E402

route_llm_through_ollama()
from app.translate.translator import translate  # noqa: E402

START, COUNT = 200, 40
REVERSE = {"hin": "Hindi", "kan": "Kannada"}
REVERSE_COUNT = 20


def main() -> None:
    english = flores_sentences("eng")
    rows = []
    for code, (name, _) in LANGUAGES.items():
        if code == "eng":
            continue
        source = flores_sentences(code)
        for index in range(START, START + COUNT):
            started = time.perf_counter()
            result = translate(source[index], target_language="English")
            elapsed = time.perf_counter() - started
            rows.append({
                "direction": f"{code}->eng", "code": code, "index": index,
                "source": source[index], "reference": english[index],
                "hypothesis": result.text, "applied": result.applied,
                "chrf": chrf(english[index], result.text) if result.applied else 0.0,
                "seconds": round(elapsed, 2),
            })
            print(f"{code}->eng {index} chrf={rows[-1]['chrf']:.1f} {elapsed:.1f}s", flush=True)

    for code, name in REVERSE.items():
        target = flores_sentences(code)
        for index in range(START, START + REVERSE_COUNT):
            started = time.perf_counter()
            result = translate(english[index], target_language=name)
            elapsed = time.perf_counter() - started
            rows.append({
                "direction": f"eng->{code}", "code": code, "index": index,
                "source": english[index], "reference": target[index],
                "hypothesis": result.text, "applied": result.applied,
                "chrf": chrf(target[index], result.text) if result.applied else 0.0,
                "seconds": round(elapsed, 2),
            })
            print(f"eng->{code} {index} chrf={rows[-1]['chrf']:.1f} {elapsed:.1f}s", flush=True)

    print("saved", save_json("exp4_translation.json", rows))


if __name__ == "__main__":
    main()
