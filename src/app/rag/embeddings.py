"""Sentence embeddings."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.core.errors import DependencyUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)

_model: Any | None = None
_lock = threading.Lock()


def get_model() -> Any:
    """Return the shared embedding model, loading it on first use."""
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:
            return _model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DependencyUnavailableError(
                "sentence-transformers is not installed, so search is unavailable. "
                "Install the project dependencies to enable it."
            ) from exc

        name = get_settings().embedding_model
        logger.info("Loading embedding model %s", name)
        try:
            _model = SentenceTransformer(name)
        except Exception as exc:  # noqa: BLE001 - surfaces download and load failures alike
            raise DependencyUnavailableError(
                f"The embedding model '{name}' could not be loaded: {exc}"
            ) from exc

        logger.info("Embedding model ready")
        return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into normalised float32 vectors, one row each."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    vectors = get_model().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_one(text: str) -> np.ndarray:
    """Embed a single text into one normalised float32 vector."""
    return embed([text])[0]


def warm_up() -> None:
    """Load the model in a background thread so the first search is not the slow one."""

    def _load() -> None:
        try:
            get_model()
        except DependencyUnavailableError as exc:
            logger.warning("Embedding model unavailable: %s", exc.message)

    threading.Thread(target=_load, name="embedding-warmup", daemon=True).start()


def is_ready() -> bool:
    """Report whether the model is already loaded, without triggering a load."""
    return _model is not None
