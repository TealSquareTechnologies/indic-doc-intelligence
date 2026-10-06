"""SQLite connections and schema setup."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def utc_now() -> str:
    """Current UTC time as an ISO 8601 string, the format used for every timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    database_file = settings.database_file
    database_file.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_file, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a connection that commits on success and rolls back on failure."""
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# Columns added later. CREATE TABLE IF NOT EXISTS leaves existing tables untouched.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("documents", "notes", "TEXT NOT NULL DEFAULT '[]'"),
)


def _apply_added_columns(connection: sqlite3.Connection) -> None:
    """Add any column the schema gained since the database was created."""
    for table, column, definition in _ADDED_COLUMNS:
        existing = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            logger.info("Added column %s.%s", table, column)


def init_database() -> None:
    """Create the schema if it does not exist yet, then bring it up to date."""
    schema = SCHEMA_FILE.read_text(encoding="utf-8")
    with get_connection() as connection:
        connection.executescript(schema)
        _apply_added_columns(connection)
    logger.info("Database ready at %s", get_settings().database_file)
