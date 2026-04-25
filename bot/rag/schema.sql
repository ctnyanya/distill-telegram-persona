-- RAG store schema: chunks + FTS5 mirror for BM25 fallback.

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,      -- 'historical' | 'runtime'
    text         TEXT NOT NULL,
    speakers     TEXT NOT NULL,      -- JSON array of display names
    timestamp    INTEGER NOT NULL,   -- unix epoch seconds
    chat_id      INTEGER,
    msg_id_start INTEGER,
    msg_id_end   INTEGER,
    embedding    BLOB                -- float32 little-endian, shape (D,)
);

CREATE INDEX IF NOT EXISTS idx_source_ts ON chunks(source, timestamp);

-- FTS5 mirror, content-less so we drive it from triggers below.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
