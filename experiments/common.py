"""Shared helpers for the experiments reported in TealSquare technical report TSQ-TR-2026-02.

Every script in this directory imports the application modules directly, so the code
measured is the code that ships. The one substitution is the model transport: the
Consciousnx endpoint routes through the Cx desktop application, which was not
running on the benchmark machine, so `route_llm_through_ollama()` replaces
`app.core.llm.complete` with a call to a local Ollama server running the same
Qwen3.5 4B weights. Prompts, system messages and temperatures are untouched.
"""

from __future__ import annotations

import json
import os
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_DIR = REPO_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))

# Language code -> (name, FLORES-200 file stem)
LANGUAGES = {
    "eng": ("English", "eng_Latn"),
    "hin": ("Hindi", "hin_Deva"),
    "ben": ("Bengali", "ben_Beng"),
    "pan": ("Punjabi", "pan_Guru"),
    "guj": ("Gujarati", "guj_Gujr"),
    "ori": ("Odia", "ory_Orya"),
    "tam": ("Tamil", "tam_Taml"),
    "tel": ("Telugu", "tel_Telu"),
    "kan": ("Kannada", "kan_Knda"),
    "mal": ("Malayalam", "mal_Mlym"),
}

FLORES_DIR = Path(os.environ.get("FLORES_DIR", REPO_ROOT / "experiments" / "data" / "flores200_dataset"))


def ollama_complete(prompt: str, *, system: str | None = None, temperature: float = 0.1,
                    model: str | None = None) -> str:
    """Same signature as app.core.llm.complete, served by a local Ollama."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model or OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX},
        "messages": messages,
    }
    request = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        payload = json.load(response)
    return (payload["message"]["content"] or "").strip()


def route_llm_through_ollama() -> None:
    """Point every application module that calls the model at the local Ollama."""
    import app.core.llm as llm
    import app.correct.corrector as corrector
    import app.rag.pipeline as pipeline
    import app.translate.translator as translator

    llm.complete = ollama_complete
    corrector.complete = ollama_complete
    translator.complete = ollama_complete
    pipeline.complete = ollama_complete


def flores_sentences(code: str, split: str = "dev") -> list[str]:
    stem = LANGUAGES[code][1]
    path = FLORES_DIR / split / f"{stem}.{split}"
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def normalise(text: str) -> str:
    """NFC normalisation and whitespace collapsing before any string comparison."""
    text = unicodedata.normalize("NFC", text)
    # Joiners are invisible and never come back out of an OCR engine.
    text = text.replace("‌", "").replace("‍", "")
    return " ".join(text.split())


def cer(reference: str, hypothesis: str) -> float:
    import jiwer
    reference, hypothesis = normalise(reference), normalise(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return float(jiwer.cer(reference, hypothesis))


def wer(reference: str, hypothesis: str) -> float:
    import jiwer
    reference, hypothesis = normalise(reference), normalise(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return float(jiwer.wer(reference, hypothesis))


def chrf(reference: str, hypothesis: str) -> float:
    from sacrebleu.metrics import CHRF
    return float(CHRF().sentence_score(normalise(hypothesis), [normalise(reference)]).score)


def save_json(name: str, payload) -> Path:
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(name: str):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
