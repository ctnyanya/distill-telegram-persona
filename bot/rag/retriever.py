"""Retriever: in-memory cosine top-k + async wrappers + runtime ingest.

Glue between Store/Embedder and the bot's LLM call. Heavy work (embed,
sqlite, matmul) runs on a single-thread executor so the asyncio loop never
blocks on these blocking calls. The historical+runtime embedding matrices
are loaded once at startup and kept resident; runtime additions are
appended in place, so we never reload the whole table.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from bot.rag.store import Chunk, Store

if TYPE_CHECKING:
    from bot.rag.embedder import Embedder

log = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: "Embedder | None",
        mode: str = "vector",
    ):
        if mode not in ("vector", "bm25"):
            raise ValueError(f"unknown rag mode: {mode}")
        if mode == "vector" and embedder is None:
            raise ValueError("vector mode requires an Embedder")

        self.store = store
        self.embedder = embedder
        self.mode = mode
        # Single worker: serializes sqlite writes and embed calls. Safe with
        # store's check_same_thread=False since there is no concurrency here.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rag"
        )

        if mode == "vector":
            self._hist_ids, self._hist_mat = store.load_all_embeddings(
                source="historical"
            )
            self._runtime_ids, self._runtime_mat = store.load_all_embeddings(
                source="runtime"
            )
            dim = (
                self._hist_mat.shape[1]
                if self._hist_mat.size
                else (self._runtime_mat.shape[1] if self._runtime_mat.size else "?")
            )
            log.info(
                "[rag] cache loaded: historical=%d runtime=%d dim=%s",
                len(self._hist_ids),
                len(self._runtime_ids),
                dim,
            )
        else:
            self._hist_ids: list[int] = []
            self._runtime_ids: list[int] = []
            self._hist_mat = np.zeros((0, 0), dtype=np.float32)
            self._runtime_mat = np.zeros((0, 0), dtype=np.float32)

    # ── retrieval ────────────────────────────────────────────────────────────

    @staticmethod
    def _topk(
        q_vec: np.ndarray, mat: np.ndarray, ids: list[int], k: int
    ) -> list[tuple[int, float]]:
        if not ids or mat.size == 0 or k <= 0:
            return []
        sims = mat @ q_vec  # both L2-normalized → dot == cosine
        k = min(k, len(ids))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(ids[i], float(sims[i])) for i in idx]

    def retrieve(
        self, query: str, k_hist: int = 3, k_runtime: int = 2
    ) -> tuple[list[Chunk], list[Chunk]]:
        if not query or not query.strip():
            return [], []

        if self.mode == "bm25":
            rows = self.store.search_fts(query, k=k_hist + k_runtime + 4)
            hist = [c for c in rows if c.source == "historical"][:k_hist]
            runtime = [c for c in rows if c.source == "runtime"][:k_runtime]
            return hist, runtime

        q_vec = self.embedder.embed_one(query)  # already L2-normalized
        hist_top = self._topk(q_vec, self._hist_mat, self._hist_ids, k_hist)
        runtime_top = self._topk(q_vec, self._runtime_mat, self._runtime_ids, k_runtime)
        return self._fetch(hist_top), self._fetch(runtime_top)

    def _fetch(self, ranked: list[tuple[int, float]]) -> list[Chunk]:
        if not ranked:
            return []
        ids = [i for i, _ in ranked]
        by_id = {c.id: c for c in self.store.get_by_ids(ids)}
        out: list[Chunk] = []
        for cid, score in ranked:
            c = by_id.get(cid)
            if c is not None:
                # Stash for callers that want to log it; not part of schema.
                setattr(c, "_score", score)
                out.append(c)
        return out

    async def retrieve_async(
        self, query: str, k_hist: int = 3, k_runtime: int = 2
    ) -> tuple[list[Chunk], list[Chunk]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self.retrieve, query, k_hist, k_runtime
        )

    # ── runtime ingestion ────────────────────────────────────────────────────

    def add_runtime(self, chunk: Chunk) -> int:
        if self.mode == "vector" and chunk.embedding is None:
            chunk.embedding = self.embedder.embed_one(chunk.text)

        new_id = self.store.add_chunk_one(chunk)
        chunk.id = new_id

        if self.mode == "vector" and chunk.embedding is not None:
            vec = np.asarray(chunk.embedding, dtype=np.float32).reshape(1, -1)
            if self._runtime_mat.size == 0:
                self._runtime_mat = vec.copy()
            else:
                self._runtime_mat = np.vstack([self._runtime_mat, vec])
            self._runtime_ids.append(new_id)

        log.info(
            "[rag] runtime_chunk id=%d speakers=%s text_preview=%r",
            new_id,
            chunk.speakers,
            chunk.text[:60],
        )
        return new_id

    async def add_runtime_async(self, chunk: Chunk) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.add_runtime, chunk)

    # ── prompt formatting ────────────────────────────────────────────────────

    @staticmethod
    def format_for_prompt(hist: list[Chunk], runtime: list[Chunk]) -> str:
        if not hist and not runtime:
            return ""

        parts: list[str] = []
        if hist:
            parts.append("## 相关历史对话（来自你过去的真实聊天记录）")
            for c in hist:
                parts.append(_render_block(c))

        if runtime:
            parts.append("## 最近的相关互动（你和群友）")
            for c in runtime:
                parts.append(_render_block(c))

        return "\n\n".join(parts)


def _render_block(c: Chunk) -> str:
    bits: list[str] = []
    if c.timestamp:
        try:
            bits.append(datetime.fromtimestamp(c.timestamp).strftime("%Y-%m-%d"))
        except (ValueError, OSError, OverflowError):
            pass
    if c.chat_id is not None:
        bits.append(f"chat={c.chat_id}")
    header = f"[{' '.join(bits)}]" if bits else ""
    return f"{header}\n{c.text}" if header else c.text
