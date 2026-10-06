-- Schema for the local SQLite store.
-- Applied on every startup; all statements are idempotent.

CREATE TABLE IF NOT EXISTS documents (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    filename           TEXT    NOT NULL,
    content_type       TEXT,
    source_kind        TEXT    NOT NULL DEFAULT 'image',
    status             TEXT    NOT NULL DEFAULT 'pending',
    error              TEXT,
    ocr_languages      TEXT,
    detected_languages TEXT    NOT NULL DEFAULT '[]',
    raw_text           TEXT    NOT NULL DEFAULT '',
    corrected_text     TEXT    NOT NULL DEFAULT '',
    translated_text    TEXT,
    notes              TEXT    NOT NULL DEFAULT '[]',
    chunk_count        INTEGER NOT NULL DEFAULT 0,
    processing_time_ms INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT    NOT NULL,
    updated_at         TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    embedding   BLOB    NOT NULL,
    dimension   INTEGER NOT NULL,
    created_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
