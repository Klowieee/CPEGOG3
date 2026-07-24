"""
bm25_index.py — In-memory BM25 keyword index over the stored chunks.

Purpose:
    Give retrieval a second, lexical opinion. Dense embeddings match on
    meaning but blur exact terms: "Is plagiarism a major offense?" failed to
    retrieve provision 5.3.1.1.6 even at top_k=8, because that chunk is a
    bare fragment split from its "major offenses" parent heading and shares
    little sentence-level meaning with the question. BM25 finds it instantly
    on the word "plagiarism". The Retriever fuses both rankings.

Inputs:
    All stored chunks plus their embeddings (VectorStore.load_all).

Outputs:
    chunk_ids ranked by keyword score (search), and full RetrievedChunk
    records scored with their true cosine similarity (lookup).

Dependencies:
    rank_bm25 (BM25Okapi), numpy, src.retrieval.vector_store.

Why this file exists:
    Keeping the keyword side in its own module means the Retriever stays a
    small fusion step, and the index can be tested without ChromaDB or an
    embedding model. The whole handbook is ~374 chunks, so holding the index
    and the embedding matrix in memory costs well under a megabyte.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import RetrievedChunk, VectorStore

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    No stemming and no stopword list: handbook queries are short and the
    terms that matter ("plagiarism", "honors", "5.3.1.1.6" -> 5 3 1 1 6) are
    already literal. BM25's own IDF weighting demotes common words without
    needing a curated list.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Keyword search over every stored chunk, with cosine rescoring."""

    def __init__(self, chunks: list[RetrievedChunk], embeddings: np.ndarray):
        """
        Args:
            chunks: All stored chunks (similarity values are ignored).
            embeddings: (n, dim) matrix aligned row-for-row with `chunks`,
                already L2-normalized by the embedder at ingestion time.
        """
        self._chunks = chunks
        self._embeddings = embeddings
        # Chunk text carries its breadcrumb ("Undergraduate > Section 10:
        # GRADING\n..."), and that is kept in the index on purpose: section
        # titles are exactly the formal vocabulary a keyword query should hit.
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self._rows = {c.chunk_id: i for i, c in enumerate(chunks)}
        log.debug("BM25 index built over %d chunks", len(chunks))

    @classmethod
    def from_store(cls, store: VectorStore) -> "BM25Index":
        """Build the index from everything in the vector store."""
        return cls(*store.load_all())

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, question: str, k: int) -> list[str]:
        """Return the chunk_ids of the top-k keyword matches, best first.

        Chunks scoring zero (no query term appears in them) are dropped
        rather than padded in: a zero-score chunk carries no keyword signal,
        and feeding it into rank fusion would reward an arbitrary tie order.
        """
        tokens = tokenize(question)
        if not tokens or not self._chunks:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:k]
        return [self._chunks[i].chunk_id for i in ranked if scores[i] > 0.0]

    def lookup(self, chunk_id: str, query_vector: np.ndarray) -> RetrievedChunk:
        """Return the chunk scored with its true cosine similarity.

        Both the stored vectors and the query vector are L2-normalized, so
        the dot product IS cosine similarity — the same number the vector
        store would have reported had this chunk placed in the semantic
        top-k. That keeps every similarity in a fused result set comparable
        and the similarity floor honest.
        """
        row = self._rows[chunk_id]
        similarity = float(np.dot(self._embeddings[row], np.asarray(
            query_vector, dtype=np.float32)))
        # Copy, so the cached record never carries a per-query score.
        return replace(self._chunks[row], similarity=similarity)
