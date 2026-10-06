"""Experiment 5: retrieval, refusal and grounded answering.

The corpus is the first CORPUS_SIZE English sentences of FLORES-200 dev, indexed as
one document with the application's own chunker, embedder and store. Questions are
written by the same 4B model from individual corpus sentences, so each question has
a known source sentence and a short gold answer.

Part a: hit rate of the source sentence in the top k for several chunkers.
Part b: the same questions against the Kannada and Hindi text of the same sentences,
        indexed in the original script and indexed after machine translation.
Part c: best retrieval score for in-corpus questions, questions about similar news
        that is not in the corpus, and unrelated general-knowledge questions.
Part d: answer accuracy of the full ask() path, with refusals counted.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FLORES_DIR,
    flores_sentences,
    normalise,
    ollama_complete,
    route_llm_through_ollama,
    save_json,
)

route_llm_through_ollama()

from app.core.config import get_settings  # noqa: E402
from app.db.database import get_connection, init_database, utc_now  # noqa: E402
from app.rag import embeddings, pipeline, store  # noqa: E402
from app.rag.chunking import ChunkStrategy  # noqa: E402

CORPUS_SIZE = 240
QUESTIONS = 60
DISTRACTOR_QUESTIONS = 30
SEED = 11

GENERAL_QUESTIONS = [
    "What is the capital of France?", "How many legs does a spider have?",
    "Who wrote Hamlet?", "What is the boiling point of water in Celsius?",
    "Which planet is closest to the Sun?", "What year did the Berlin Wall fall?",
    "What is the chemical symbol for gold?", "How many days are in a leap year?",
    "Who painted the Mona Lisa?", "What is the largest ocean on Earth?",
    "What is the square root of 144?", "Which language has the most native speakers?",
    "What is the tallest mountain in Africa?", "How many strings does a violin have?",
    "What does DNA stand for?", "Who discovered penicillin?",
    "What is the currency of Japan?", "How many players are on a football team?",
    "What is the freezing point of water in Fahrenheit?", "Which gas do plants absorb?",
]

QUESTION_PROMPT = (
    "Write one short factual question that is answered by the sentence below, and the "
    "short answer (a few words at most, copied from the sentence).\n"
    'Return JSON only, in the form {"question": "...", "answer": "..."}.\n\n'
    "Sentence: {sentence}"
)

JUDGE_PROMPT = (
    "Question: {question}\nGold answer: {gold}\nCandidate answer: {candidate}\n\n"
    "Does the candidate answer state the same fact as the gold answer? "
    "Reply with exactly one word: yes or no."
)


def make_questions(sentences: list[str], count: int, rng: random.Random, label: str) -> list[dict]:
    chosen = rng.sample(range(len(sentences)), count)
    items = []
    for index in chosen:
        raw = ollama_complete(QUESTION_PROMPT.replace("{sentence}", sentences[index]),
                              temperature=0.1)
        match = re.search(r"\{.*\}", raw, re.S)
        try:
            parsed = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            parsed = {}
        if not parsed.get("question") or not parsed.get("answer"):
            continue
        items.append({"index": index, "sentence": sentences[index],
                      "question": parsed["question"].strip(),
                      "answer": str(parsed["answer"]).strip(), "kind": label})
        print(label, index, items[-1]["question"], "->", items[-1]["answer"], flush=True)
    return items


def fresh_document(name: str) -> int:
    timestamp = utc_now()
    with get_connection() as connection:
        connection.execute("DELETE FROM documents")
        cursor = connection.execute(
            "INSERT INTO documents (filename, source_kind, status, created_at, updated_at) "
            "VALUES (?, 'text', 'ready', ?, ?)", (name, timestamp, timestamp))
    store.invalidate_cache()
    return int(cursor.lastrowid)


def hits(question: str, target: str, k: int) -> bool:
    vector = embeddings.embed_one(question)
    found = store.search(vector, top_k=k)
    target = normalise(target)
    return any(target in normalise(hit.text) for hit in found)


def part_a(questions: list[dict], text: str) -> list[dict]:
    rows = []
    configs = [("words", 40, 8), ("words", 80, 16), ("words", 160, 32), ("words", 250, 50),
               ("sentences", 40, 0), ("sentences", 80, 0), ("sentences", 160, 0)]
    for strategy, size, overlap in configs:
        document_id = fresh_document("english")
        started = time.perf_counter()
        chunks = pipeline.index_text(document_id, text, strategy=ChunkStrategy(strategy),
                                     size=size, overlap=overlap)
        index_seconds = time.perf_counter() - started
        for k in (1, 3, 5):
            hit = sum(hits(q["question"], q["sentence"], k) for q in questions)
            rows.append({"strategy": strategy, "size": size, "overlap": overlap, "chunks": chunks,
                         "k": k, "hits": hit, "hit_rate": round(hit / len(questions), 4),
                         "index_seconds": round(index_seconds, 2)})
        print(strategy, size, [r["hit_rate"] for r in rows[-3:]], flush=True)
    return rows


def part_b(questions: list[dict], translations: dict[str, list[str]]) -> list[dict]:
    rows = []
    english = flores_sentences("eng")[:CORPUS_SIZE]
    for code in ("kan", "hin"):
        native = flores_sentences(code)[:CORPUS_SIZE]
        variants = {"native_script": native, "translated": translations[code]}
        for variant, sentences in variants.items():
            document_id = fresh_document(f"{code}-{variant}")
            pipeline.index_text(document_id, " ".join(sentences),
                                strategy=ChunkStrategy.SENTENCES, size=80, overlap=0)
            for k in (1, 3, 5):
                hit = 0
                for q in questions:
                    target = sentences[q["index"]]
                    hit += bool(target) and hits(q["question"], target, k)
                rows.append({"code": code, "variant": variant, "k": k, "hits": hit,
                             "hit_rate": round(hit / len(questions), 4)})
            print(code, variant, [r["hit_rate"] for r in rows[-3:]], flush=True)
    document_id = fresh_document("english-reference")
    pipeline.index_text(document_id, " ".join(english), strategy=ChunkStrategy.SENTENCES,
                        size=80, overlap=0)
    for k in (1, 3, 5):
        hit = sum(hits(q["question"], q["sentence"], k) for q in questions)
        rows.append({"code": "eng", "variant": "reference", "k": k, "hits": hit,
                     "hit_rate": round(hit / len(questions), 4)})
    return rows


def part_c(in_corpus: list[dict], distractors: list[dict], text: str) -> list[dict]:
    document_id = fresh_document("english")
    pipeline.index_text(document_id, text, strategy=ChunkStrategy.SENTENCES, size=80, overlap=0)
    rows = []
    groups = [("in_corpus", [q["question"] for q in in_corpus]),
              ("similar_news", [q["question"] for q in distractors]),
              ("general", GENERAL_QUESTIONS)]
    for label, items in groups:
        for question in items:
            found = store.search(embeddings.embed_one(question), top_k=1)
            rows.append({"group": label, "question": question,
                         "best_score": found[0].score if found else 0.0})
    return rows


def part_d(in_corpus: list[dict], distractors: list[dict], text: str) -> list[dict]:
    document_id = fresh_document("english")
    pipeline.index_text(document_id, text, strategy=ChunkStrategy.SENTENCES, size=80, overlap=0)
    settings = get_settings()
    rows = []
    groups = [("in_corpus", in_corpus), ("similar_news", distractors),
              ("general", [{"question": q, "answer": ""} for q in GENERAL_QUESTIONS])]
    for label, items in groups:
        for q in items:
            started = time.perf_counter()
            answer = pipeline.ask(q["question"], top_k=settings.retrieval_top_k,
                                  threshold=settings.similarity_threshold)
            elapsed = time.perf_counter() - started
            refused = answer.answer == pipeline.NO_ANSWER or not answer.grounded
            contains = bool(q["answer"]) and (
                normalise(q["answer"]).lower() in normalise(answer.answer).lower())
            judged = None
            if label == "in_corpus" and not refused:
                verdict = ollama_complete(
                    JUDGE_PROMPT.format(question=q["question"], gold=q["answer"],
                                        candidate=answer.answer), temperature=0.0)
                judged = verdict.strip().lower().startswith("yes")
            cited = bool(re.search(r"\[Source \d+\]", answer.answer))
            rows.append({"group": label, "question": q["question"], "gold": q["answer"],
                         "answer": answer.answer, "grounded": answer.grounded,
                         "refused": refused, "best_score": answer.best_score,
                         "contains_gold": contains, "judge_correct": judged, "cited": cited,
                         "seconds": round(elapsed, 2)})
            status = "refused" if refused else ("ok" if (judged or contains) else "wrong")
            print(label, status, f"{elapsed:.1f}s", q["question"][:60], "->",
                  answer.answer[:60], flush=True)
    return rows


def translate_corpus(code: str, sentences: list[str]) -> list[str]:
    from app.translate.translator import translate
    out = []
    for i, sentence in enumerate(sentences):
        result = translate(sentence, target_language="English")
        out.append(result.text if result.applied else "")
        if i % 20 == 0:
            print(code, "translated", i, flush=True)
    return out


def main() -> None:
    settings = get_settings()
    settings.database_path = str(Path(__file__).resolve().parent / "results" / "exp5.db")
    init_database()
    rng = random.Random(SEED)

    english = flores_sentences("eng")
    corpus = english[:CORPUS_SIZE]
    text = " ".join(corpus)

    devtest_path = FLORES_DIR / "devtest" / "eng_Latn.devtest"
    devtest = devtest_path.read_text(encoding="utf-8").splitlines()

    questions = make_questions(corpus, QUESTIONS, rng, "in_corpus")
    distractors = make_questions(devtest, DISTRACTOR_QUESTIONS, rng, "similar_news")
    save_json("exp5_questions.json", {"in_corpus": questions, "similar_news": distractors})

    save_json("exp5_a_chunking.json", part_a(questions, text))
    save_json("exp5_c_scores.json", part_c(questions, distractors, text))
    save_json("exp5_d_answers.json", part_d(questions, distractors, text))

    translations = {}
    for code in ("kan", "hin"):
        translations[code] = translate_corpus(code, flores_sentences(code)[:CORPUS_SIZE])
    save_json("exp5_translations.json", translations)
    save_json("exp5_b_crosslingual.json", part_b(questions, translations))
    print("done")


if __name__ == "__main__":
    main()
