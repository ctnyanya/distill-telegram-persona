"""Build historical RAG index from preprocessor jsonl files.

Usage:
    python scripts/build_rag_index.py --target 张三

Output:
    data/<target>/rag.db  (SQLite, single file)

Deletes the existing rag.db first, so this script is the reset button for the
historical index. Runtime chunks (written live by bot.py in Phase 2) would be
lost; acceptable at this stage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.rag.chunker import chunk_historical  # noqa: E402
from bot.rag.embedder import Embedder  # noqa: E402
from bot.rag.store import Store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    help="target name under data/ (e.g. 张三)")
    ap.add_argument("--db", default=None,
                    help="override output path (default: data/<target>/rag.db)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="embedding batch size (default 32)")
    args = ap.parse_args()

    target_dir = ROOT / "data" / args.target
    progress_dir = target_dir / "progress"
    if not progress_dir.exists():
        sys.exit(f"no progress dir: {progress_dir}")

    db_path = Path(args.db) if args.db else target_dir / "rag.db"
    if db_path.exists():
        print(f"removing existing db: {db_path}")
        db_path.unlink()

    jsonl_files = sorted(progress_dir.glob("*.jsonl"))
    if not jsonl_files:
        sys.exit(f"no jsonl files in {progress_dir}")
    print(f"found {len(jsonl_files)} jsonl file(s)")

    all_chunks = []
    for p in jsonl_files:
        t0 = time.time()
        chunks = chunk_historical(p)
        print(f"  {p.name}: {len(chunks)} chunks  ({time.time()-t0:.1f}s)")
        all_chunks.extend(chunks)

    if not all_chunks:
        sys.exit("no chunks produced")

    print(f"total chunks: {len(all_chunks)}")

    print("loading fastembed model (first run will download ONNX)...")
    embedder = Embedder()
    texts = [c.text for c in all_chunks]

    print(f"embedding {len(texts)} chunks (batch_size={args.batch_size})...")
    t0 = time.time()
    emb = embedder.embed_batch(texts, batch_size=args.batch_size)
    print(f"  done in {time.time()-t0:.1f}s, shape={emb.shape}")

    for c, e in zip(all_chunks, emb):
        c.embedding = e

    print("writing to sqlite...")
    store = Store(db_path)
    t0 = time.time()
    insert_batch = 500
    for i in range(0, len(all_chunks), insert_batch):
        store.add_chunks_batch(all_chunks[i : i + insert_batch])
    store.close()
    print(f"  done in {time.time()-t0:.1f}s")

    size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"\ndone: {db_path}")
    print(f"  size: {size_mb:.1f} MB")
    print(f"  chunks: {len(all_chunks)}")


if __name__ == "__main__":
    main()
