"""Phase-1 verification CLI: query the RAG index by vector or BM25.

Usage:
    python scripts/query_rag.py "奶茶"
    python scripts/query_rag.py "奶茶" --k 10 --mode bm25
    python scripts/query_rag.py "奶茶" --target 张三
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.rag.embedder import Embedder  # noqa: E402
from bot.rag.store import Chunk, Store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="search query (free-form text)")
    ap.add_argument("--target", default="张三",
                    help="target name under data/ (default: 张三)")
    ap.add_argument("--db", default=None,
                    help="override db path (default: data/<target>/rag.db)")
    ap.add_argument("--k", type=int, default=5, help="top-k (default 5)")
    ap.add_argument("--mode", choices=["vector", "bm25"], default="vector")
    ap.add_argument("--source", choices=["historical", "runtime"], default=None,
                    help="restrict to one source (vector mode only)")
    ap.add_argument("--preview", type=int, default=600,
                    help="chunk text preview length (default 600)")
    args = ap.parse_args()

    db_path = Path(args.db) if args.db else ROOT / "data" / args.target / "rag.db"
    if not db_path.exists():
        sys.exit(f"db not found: {db_path}  (run build_rag_index.py first)")

    store = Store(db_path)

    if args.mode == "bm25":
        _run_bm25(store, args.query, args.k, args.preview)
    else:
        _run_vector(store, args.query, args.k, args.source, args.preview)


def _run_bm25(store: Store, query: str, k: int, preview: int) -> None:
    t0 = time.time()
    results = store.search_fts(query, k=k)
    elapsed_ms = (time.time() - t0) * 1000
    print(f"[bm25] {len(results)} results  ({elapsed_ms:.1f} ms)\n")
    for rank, c in enumerate(results, 1):
        _print_chunk(c, rank=rank, preview=preview)


def _run_vector(
    store: Store, query: str, k: int, source: str | None, preview: int
) -> None:
    print("loading embedder...")
    embedder = Embedder()

    t0 = time.time()
    q_emb = embedder.embed_one(query)
    ids, mat = store.load_all_embeddings(source=source)
    if len(ids) == 0:
        sys.exit("no embeddings in db")

    sims = mat @ q_emb  # both L2-normalized → cosine
    top_idx = np.argsort(-sims)[:k]
    top_ids = [ids[i] for i in top_idx]
    top_scores = [float(sims[i]) for i in top_idx]

    chunks = {c.id: c for c in store.get_by_ids(top_ids)}
    elapsed_ms = (time.time() - t0) * 1000

    src_note = f", source={source}" if source else ""
    print(f"[vector] top-{k} of {len(ids)} chunks "
          f"({elapsed_ms:.1f} ms{src_note})\n")
    for rank, (cid, score) in enumerate(zip(top_ids, top_scores), 1):
        c = chunks.get(cid)
        if c:
            _print_chunk(c, rank=rank, score=score, preview=preview)


def _print_chunk(
    c: Chunk,
    rank: int | None = None,
    score: float | None = None,
    preview: int = 600,
) -> None:
    ts = (
        datetime.fromtimestamp(c.timestamp).strftime("%Y-%m-%d %H:%M")
        if c.timestamp else "-"
    )
    parts = []
    if rank is not None:
        parts.append(f"#{rank}")
    if score is not None:
        parts.append(f"score={score:.3f}")
    parts.append(f"id={c.id}")
    parts.append(f"src={c.source}")
    parts.append(f"ts={ts}")
    parts.append(f"speakers={','.join(c.speakers)}")
    if c.msg_id_start is not None:
        parts.append(f"msgs={c.msg_id_start}-{c.msg_id_end}")
    print("─── " + "  ".join(parts))

    text = c.text if len(c.text) <= preview else c.text[:preview] + "..."
    print(text)
    print()


if __name__ == "__main__":
    main()
