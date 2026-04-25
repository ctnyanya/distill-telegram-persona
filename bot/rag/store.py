"""SQLite-backed chunk store.

Single-file DB, no server. FTS5 mirror is populated by triggers defined in
schema.sql, so `search_fts` works out of the box. Embeddings are stored as raw
float32 blobs; `load_all_embeddings` returns the full matrix for brute-force
cosine search — fine for Phase A (well below the ANN-payoff threshold).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@dataclass
class Chunk:
    source: str                     # 'historical' | 'runtime'
    text: str
    speakers: list[str]
    timestamp: int                  # unix epoch seconds
    chat_id: int | None = None
    msg_id_start: int | None = None
    msg_id_end: int | None = None
    embedding: np.ndarray | None = None
    id: int | None = None


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so a single Connection can be shared with the
        # rag executor thread; concurrency is serialized by the single-worker
        # ThreadPoolExecutor in Retriever, so this is safe.
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    # ── writes ──────────────────────────────────────────────────────────────

    def add_chunks_batch(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        rows = []
        for c in chunks:
            emb_blob: bytes | None = None
            if c.embedding is not None:
                arr = np.asarray(c.embedding, dtype=np.float32)
                emb_blob = arr.tobytes()
            rows.append((
                c.source,
                c.text,
                json.dumps(c.speakers, ensure_ascii=False),
                c.timestamp,
                c.chat_id,
                c.msg_id_start,
                c.msg_id_end,
                emb_blob,
            ))
        self.conn.executemany(
            "INSERT INTO chunks(source, text, speakers, timestamp, chat_id, "
            "msg_id_start, msg_id_end, embedding) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def add_chunk_one(self, chunk: Chunk) -> int:
        """Insert a single chunk and return its new rowid.

        Used by runtime ingestion: the caller needs the id immediately to
        append the row's embedding to the in-memory cache.
        """
        emb_blob: bytes | None = None
        if chunk.embedding is not None:
            arr = np.asarray(chunk.embedding, dtype=np.float32)
            emb_blob = arr.tobytes()
        cur = self.conn.execute(
            "INSERT INTO chunks(source, text, speakers, timestamp, chat_id, "
            "msg_id_start, msg_id_end, embedding) VALUES (?,?,?,?,?,?,?,?)",
            (
                chunk.source,
                chunk.text,
                json.dumps(chunk.speakers, ensure_ascii=False),
                chunk.timestamp,
                chunk.chat_id,
                chunk.msg_id_start,
                chunk.msg_id_end,
                emb_blob,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # ── reads ───────────────────────────────────────────────────────────────

    def load_all_embeddings(
        self, source: str | None = None
    ) -> tuple[list[int], np.ndarray]:
        """Return (ids, matrix[N, D]) for every chunk that has an embedding."""
        q = "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
        params: tuple = ()
        if source:
            q += " AND source = ?"
            params = (source,)
        rows = self.conn.execute(q, params).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [r[0] for r in rows]
        mats = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
        return ids, np.stack(mats)

    def get_by_ids(self, ids: list[int]) -> list[Chunk]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, source, text, speakers, timestamp, chat_id, "
            f"msg_id_start, msg_id_end, embedding FROM chunks "
            f"WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {r[0]: r for r in rows}
        out: list[Chunk] = []
        for i in ids:
            r = by_id.get(i)
            if not r:
                continue
            emb = (
                np.frombuffer(r[8], dtype=np.float32).copy() if r[8] else None
            )
            out.append(Chunk(
                id=r[0],
                source=r[1],
                text=r[2],
                speakers=json.loads(r[3]),
                timestamp=r[4],
                chat_id=r[5],
                msg_id_start=r[6],
                msg_id_end=r[7],
                embedding=emb,
            ))
        return out

    def search_fts(self, query: str, k: int = 5) -> list[Chunk]:
        """BM25 search via FTS5 — used in bm25 fallback mode (Phase B / OOM)."""
        # Quote the whole query as a phrase to avoid MATCH syntax errors on
        # arbitrary user input (punctuation, spaces, etc.).
        safe = query.replace('"', '""')
        rows = self.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (f'"{safe}"', k),
        ).fetchall()
        ids = [r[0] for r in rows]
        # Preserve ranked order
        chunks = {c.id: c for c in self.get_by_ids(ids)}
        return [chunks[i] for i in ids if i in chunks]

    def count(self, source: str | None = None) -> int:
        if source:
            return self.conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE source = ?", (source,)
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
