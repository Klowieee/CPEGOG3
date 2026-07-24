"""
retriever.py — Turn a question into relevant handbook chunks.

Purpose:
    Compose the Embedder and VectorStore into the one call the chat layer
    needs: retrieve(question) -> ranked chunks. Also owns the similarity
    floor that drives the no-API-call refusal path (docs/prompting.md §4):
    if nothing clears the floor, the chat layer refuses without ever calling
    the LLM — cost-free and hallucination-proof.

    When a BM25 index is supplied, retrieval is hybrid: the semantic ranking
    and the keyword ranking are combined by reciprocal rank fusion, which
    recovers chunks whose exact terms match but whose sentence meaning does
    not (see src/retrieval/bm25_index.py for the motivating case).

Inputs:
    A natural-language question; k (defaults to settings.top_k).

Outputs:
    list[RetrievedChunk], best first. Every chunk carries a true cosine
    similarity, whichever ranking surfaced it.

Dependencies:
    src.embedding.embedder, src.retrieval.vector_store,
    src.retrieval.bm25_index, config.

Why this file exists:
    Retrieval is the RAG step that decides answer quality; keeping it in one
    small, injectable class lets us evaluate it in isolation (no LLM needed)
    via scripts/eval_retrieval.py and the golden set.
"""

from __future__ import annotations

import logging

from src.embedding.embedder import Embedder
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_store import RetrievedChunk, VectorStore
from src.utils.config import Settings

log = logging.getLogger(__name__)


class Retriever:
    """Embeds a query and returns the most similar stored chunks."""

    def __init__(self, embedder: Embedder, store: VectorStore,
                 top_k: int, similarity_floor: float,
                 bm25_index: BM25Index | None = None, rrf_k: int = 60,
                 keyword_weight: float = 0.5):
        """
        Args:
            bm25_index: Keyword index for hybrid retrieval. None means
                semantic-only, the original behavior.
            rrf_k: Reciprocal rank fusion constant. Larger values flatten the
                weighting between ranks; 60 is the standard default.
            keyword_weight: How much a keyword rank counts against a semantic
                one. Below 1.0 because with only top_k slots to fill, equal
                weighting evicts good semantic results — measured to cost
                both hit@k and answer quality on this corpus.
        """
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.similarity_floor = similarity_floor
        self.bm25_index = bm25_index
        self.rrf_k = rrf_k
        self.keyword_weight = keyword_weight

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks for a question, best first.

        The result is NOT floor-filtered — callers get the full ranked list
        for the prompt, and use meets_floor() to decide whether to answer or
        refuse. Returning everything keeps the retriever a pure ranking step.

        With a BM25 index attached the order is by fused rank, so results[0]
        is the best overall match but not necessarily the highest cosine one;
        use best_similarity() rather than results[0].similarity for scoring.
        """
        k = k or self.top_k
        query_vector = self.embedder.embed_query(question)
        semantic = self.store.query(query_vector, k)

        if self.bm25_index is None:
            if semantic:
                log.debug("Top similarity for %r: %.3f",
                          question, semantic[0].similarity)
            return semantic

        keyword_ids = self.bm25_index.search(question, k)
        results = self._fuse(semantic, keyword_ids, query_vector, k)
        if results:
            log.debug("Hybrid top for %r: fused=%s best_sim=%.3f",
                      question, results[0].chunk_id, best_similarity(results))
        return results

    def _fuse(self, semantic: list[RetrievedChunk], keyword_ids: list[str],
              query_vector, k: int) -> list[RetrievedChunk]:
        """Combine the two rankings by reciprocal rank fusion.

        RRF scores a chunk as the sum of 1/(rrf_k + rank) over the lists it
        appears in, so agreement between the two retrievers outranks a strong
        showing in either alone. Ranks are used rather than raw scores because
        BM25 scores are unbounded and corpus-dependent while cosine sits in a
        narrow band — there is no stable way to put them on one scale.
        """
        scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(semantic, start=1):
            scores[chunk.chunk_id] = 1.0 / (self.rrf_k + rank)
            chunks[chunk.chunk_id] = chunk

        for rank, chunk_id in enumerate(keyword_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                self.keyword_weight / (self.rrf_k + rank))
            if chunk_id not in chunks:
                # Keyword-only hit: materialize it with its real cosine score.
                chunks[chunk_id] = self.bm25_index.lookup(chunk_id, query_vector)

        ordered = sorted(
            chunks.values(),
            key=lambda c: (scores[c.chunk_id], c.similarity),
            reverse=True,
        )
        return ordered[:k]

    def meets_floor(self, results: list[RetrievedChunk]) -> bool:
        """True if the best result is relevant enough to attempt an answer.

        This is the retrieval-layer half of the two-layer refusal design: a
        False here means "not covered" and the chat layer refuses without
        calling the LLM.

        The check is on the highest cosine similarity in the set, not on
        results[0], because fusion reorders by rank. That keeps the decision
        identical to the semantic-only one: the top semantic chunk always
        survives fusion (it holds rank 1 in one of the two lists), and no
        keyword-only chunk can out-score it on cosine — it would have been in
        the semantic top-k if it could. So hybrid changes which excerpts the
        model sees, never whether the system answers at all.
        """
        return bool(results) and best_similarity(results) >= self.similarity_floor


def best_similarity(results: list[RetrievedChunk]) -> float:
    """Highest cosine similarity in a result set (0.0 if empty)."""
    return max((r.similarity for r in results), default=0.0)


def build_retriever(settings: Settings, store: VectorStore | None = None,
                    embedder: Embedder | None = None) -> Retriever:
    """Construct the configured retriever, hybrid index included when enabled.

    Shared by the chat app, the evaluation script, and the integration tests
    so they cannot drift into retrieving differently from each other.

    Args:
        store: Pre-built (and ideally already validated) store; created from
            settings when omitted.
        embedder: Pre-built embedder; created from settings when omitted.
    """
    store = store or VectorStore(settings.paths.vector_db_dir, settings.document.id)
    embedder = embedder or Embedder(settings.embedding.model,
                                    settings.embedding.query_prefix)

    bm25_index = None
    if settings.retrieval.hybrid.enabled:
        bm25_index = BM25Index.from_store(store)
        log.info("Hybrid retrieval enabled (BM25 over %d chunks)", len(bm25_index))

    return Retriever(embedder, store, settings.retrieval.top_k,
                     settings.retrieval.similarity_floor,
                     bm25_index=bm25_index,
                     rrf_k=settings.retrieval.hybrid.rrf_k,
                     keyword_weight=settings.retrieval.hybrid.keyword_weight)
