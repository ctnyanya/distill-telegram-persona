"""fastembed wrapper: local ONNX, lazy-loaded, L2-normalized float32 output."""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"  # 512-dim, Chinese, ~100 MB ONNX


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None  # lazy

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self.model_name)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return (N, D) float32 matrix, L2-normalized so dot == cosine."""
        self._ensure_loaded()
        embs = list(self._model.embed(texts, batch_size=batch_size))
        arr = np.asarray(embs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
