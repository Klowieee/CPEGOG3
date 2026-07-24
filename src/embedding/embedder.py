"""
embedder.py — Local embedding of chunks and queries with bge-small-en-v1.5.

Purpose:
    Provide ONE wrapper that both the ingestion pipeline and the query-time
    retriever use to turn text into vectors. Centralizing this guarantees
    passages and queries are embedded by the identical model with identical
    normalization — a divergence here (e.g. normalizing one side only, or
    prefixing passages) silently wrecks retrieval and is a classic RAG bug.

Inputs:
    Model name and query prefix (from EmbeddingSettings); text to embed.

Outputs:
    L2-normalized float32 vectors as numpy arrays. With normalized vectors,
    a dot product equals cosine similarity — which is what the vector store
    is configured to use.

Dependencies:
    numpy (always). sentence-transformers is imported lazily only when the
    real model is needed, so this module (and its tests) import without any
    heavyweight ML libraries present.

Why this file exists:
    docs/embedding_strategy.md selects bge-small-en-v1.5 and specifies two
    usage rules that must be enforced in exactly one place: (1) queries get
    the model's instruction prefix, passages do not; (2) all vectors are
    L2-normalized. This wrapper is that place.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Any object with a compatible .encode(list[str]) -> array method can back the
# Embedder (the real SentenceTransformer, or a fake in tests). This is what
# makes the wrapper's logic testable without downloading the model.
DEFAULT_BATCH_SIZE = 32


class Embedder:
    """Embeds passages and queries into normalized vectors.

    The real sentence-transformers model is loaded lazily on first use, so
    constructing an Embedder is cheap and import-safe. An alternate encoder
    may be injected (used by tests) to exercise all wrapper logic without the
    model.
    """

    def __init__(self, model_name: str, query_prefix: str, encoder=None):
        """
        Args:
            model_name: HuggingFace id of the embedding model.
            query_prefix: Instruction prefix prepended to QUERIES only
                (bge models are trained to expect it; passages get none).
            encoder: Optional pre-built encoder exposing
                .encode(list[str]) -> np.ndarray. If None, a
                SentenceTransformer is loaded lazily from model_name.
        """
        self.model_name = model_name
        self.query_prefix = query_prefix
        self._encoder = encoder

    def _get_encoder(self):
        """Load the SentenceTransformer on first use (lazy, cached)."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # heavy, lazy

            log.info("Loading embedding model %s ...", self.model_name)
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed passages (no prefix). Returns an (n, dim) normalized array.

        Used by ingestion to embed chunk text. Order is preserved so vectors
        align with the input chunks.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._encode(texts)
        return _l2_normalize(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query (with the instruction prefix). Returns (dim,)."""
        prefixed = f"{self.query_prefix}{query}"
        vector = self._encode([prefixed])[0]
        return _l2_normalize(vector[np.newaxis, :])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Call the backing encoder and return a float32 numpy array."""
        encoder = self._get_encoder()
        vectors = encoder.encode(
            texts,
            batch_size=DEFAULT_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so dot product == cosine similarity.

    Rows with zero norm (empty/degenerate text) are left as zeros rather than
    dividing by zero.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return (vectors / safe).astype(np.float32)
