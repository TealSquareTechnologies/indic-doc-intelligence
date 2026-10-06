"""Experiment 7: cost of the stages that do not call the language model.

Detection, chunking, embedding and search are timed on their own so that the
latency table can attribute each part of a request to the stage that caused it.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, flores_sentences, save_json  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.database import init_database  # noqa: E402
from app.detect.detector import detect  # noqa: E402
from app.rag import embeddings, pipeline, store  # noqa: E402
from app.rag.chunking import ChunkStrategy, chunk_text  # noqa: E402

PAGES_DIR = REPO_ROOT / "experiments" / "pages"


def timed(fn, repeats=20):
    fn()  # warm up
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> None:
    settings = get_settings()
    settings.database_path = str(Path(__file__).resolve().parent / "results" / "exp7.db")
    init_database()
    manifest = json.loads((PAGES_DIR / "manifest.json").read_text(encoding="utf-8"))
    page = next(m["text"] for m in manifest if m["code"] == "kan")
    english = flores_sentences("eng")

    out = {}
    out["Detection, one page"] = timed(lambda: detect(page))
    out["Chunking, 240 sentences"] = timed(lambda: chunk_text(" ".join(english[:240]),
                                                                strategy=ChunkStrategy.SENTENCES, size=80))
    embeddings.get_model()
    out["Embedding, one 80-word chunk"] = timed(lambda: embeddings.embed_one(" ".join(english[:5])), 10)
    out["Embedding, one question"] = timed(lambda: embeddings.embed_one("When did the protest start?"), 10)

    from app.db.database import get_connection, utc_now
    timestamp = utc_now()
    with get_connection() as connection:
        connection.execute("DELETE FROM documents")
        cursor = connection.execute(
            "INSERT INTO documents (filename, source_kind, status, created_at, updated_at) "
            "VALUES ('t', 'text', 'ready', ?, ?)", (timestamp, timestamp))
        document_id = int(cursor.lastrowid)
    store.invalidate_cache()
    started = time.perf_counter()
    count = pipeline.index_text(document_id, " ".join(english), strategy=ChunkStrategy.SENTENCES, size=80)
    out[f"Indexing {count} chunks (all 997 sentences)"] = time.perf_counter() - started
    query = embeddings.embed_one("When did the protest start?")
    store.search(query, top_k=3)
    out[f"Search over {count} chunks"] = timed(lambda: store.search(query, top_k=3), 50)
    for k, v in out.items():
        print(f"{k:45} {v:.4f} s")
    save_json("exp_stage_timing.json", out)


if __name__ == "__main__":
    main()
