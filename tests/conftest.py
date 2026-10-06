"""Shared test fixtures."""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.core.llm import get_client
from app.db.database import init_database
from app.rag import store


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point every setting that touches disk or the network at test-local values."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("CONSCIOUSNX_API_KEY", "")
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("PREPROCESS_IMAGES", "false")

    get_settings.cache_clear()
    get_client.cache_clear()
    store.invalidate_cache()

    yield get_settings()

    get_settings.cache_clear()
    get_client.cache_clear()
    store.invalidate_cache()


@pytest.fixture
def database(isolated_settings):
    """A migrated, empty database."""
    init_database()
    return isolated_settings.database_file
