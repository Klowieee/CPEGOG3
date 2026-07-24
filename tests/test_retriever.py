"""
test_retriever.py — Tests for src/retrieval/retriever.py (Phase 7).

Uses a fake embedder whose vectors follow the SAME deterministic hash scheme
the chunks are stored with, so a query for a chunk's text retrieves that
chunk — exercising the real embedder→store wiring without the model. Covers
ranking, k, and the meets_floor refusal decision.

Dependencies:
    pytest, numpy, chromadb, src.retrieval.*, src.chunking.chunker.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.chunker import Chunk                       # noqa: E402
from src.retrieval.bm25_index import BM25Index               # noqa: E402
from src.retrieval.retriever import Retriever, best_similarity  # noqa: E402
from src.retrieval.vector_store import VectorStore           # noqa: E402

DIM = 32


def _vec(text: str) -> np.ndarray:
    h = hashlib.sha256(text.encode()).digest()
    v = np.array([b - 128 for b in h[:DIM]], dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    """Embeds with the same hash scheme used to store chunks."""

    def embed_texts(self, texts):
        return np.stack([_vec(t) for t in texts])

    def embed_query(self, query):
        return _vec(query)


def _chunks():
    return [
        Chunk("d_1", "Plagiarism is a major offense under discipline rules.",
              "doc", "General Provisions", "5", "STANDARDS", ["5.3"], [60], 200),
        Chunk("d_2", "Undergraduate grading and credit rules and GPA.",
              "doc", "Undergraduate", "10", "GRADING", ["10.1"], [101], 200),
        Chunk("d_3", "Graduate comprehensive examination requirements.",
              "doc", "Graduate", "16", "COMPS", ["16.1"], [118], 200),
    ]


@pytest.fixture()
def retriever(tmp_path):
    chunks = _chunks()
    store = VectorStore(tmp_path / "db", "doc")
    store.rebuild(chunks, np.stack([_vec(c.text) for c in chunks]), "fake")
    return Retriever(FakeEmbedder(), store, top_k=3, similarity_floor=0.5)


def test_retrieve_ranks_exact_text_first(retriever):
    results = retriever.retrieve("Plagiarism is a major offense under discipline rules.")
    assert results[0].section_number == "5"
    assert results[0].similarity == pytest.approx(1.0, abs=1e-4)


def test_retrieve_respects_k(retriever):
    assert len(retriever.retrieve("anything", k=2)) == 2


def test_meets_floor_true_for_strong_match(retriever):
    results = retriever.retrieve("Undergraduate grading and credit rules and GPA.")
    assert retriever.meets_floor(results) is True


def test_meets_floor_false_for_weak_match(retriever):
    # An unrelated query yields low similarity -> below floor -> refuse.
    results = retriever.retrieve("zzz qqq unrelated gibberish tokens")
    assert retriever.meets_floor(results) is False


def test_meets_floor_false_on_empty():
    r = Retriever(FakeEmbedder(), store=None, top_k=3, similarity_floor=0.5)
    assert r.meets_floor([]) is False


# --- Hybrid retrieval ------------------------------------------------------

@pytest.fixture()
def hybrid(tmp_path):
    chunks = _chunks()
    store = VectorStore(tmp_path / "db", "doc")
    store.rebuild(chunks, np.stack([_vec(c.text) for c in chunks]), "fake")
    return Retriever(FakeEmbedder(), store, top_k=3, similarity_floor=0.5,
                     bm25_index=BM25Index.from_store(store), keyword_weight=1.0)


def test_hybrid_finds_chunk_the_embedder_cannot(hybrid, retriever):
    """The motivating case: exact term present, sentence meaning absent.

    The hash embedder scores this phrasing near zero against every chunk, so
    only the keyword side can surface the plagiarism chunk.
    """
    question = "plagiarism"
    assert best_similarity(retriever.retrieve(question)) < 0.5
    assert "d_1" in [c.chunk_id for c in hybrid.retrieve(question)]


def test_hybrid_ranks_agreement_first(hybrid):
    # Present in both rankings (exact text match semantically, exact terms
    # lexically), so it must outrank chunks found by one side alone.
    results = hybrid.retrieve("Plagiarism is a major offense under discipline rules.")
    assert results[0].chunk_id == "d_1"


def test_hybrid_keyword_hit_carries_true_cosine(hybrid):
    results = hybrid.retrieve("Plagiarism is a major offense under discipline rules.")
    # Exact text match -> cosine 1.0, whichever ranking surfaced it.
    assert best_similarity(results) == pytest.approx(1.0, abs=1e-4)


def test_hybrid_respects_k(hybrid):
    assert len(hybrid.retrieve("plagiarism grading examination", k=2)) == 2


def test_hybrid_still_refuses_gibberish(hybrid):
    # Fusion must not manufacture relevance: no shared terms, no meaning.
    results = hybrid.retrieve("zzz qqq unrelated gibberish tokens")
    assert hybrid.meets_floor(results) is False


def test_hybrid_preserves_the_floor_decision(hybrid, retriever):
    """Hybrid changes which excerpts are shown, never whether we answer."""
    for question in ["Undergraduate grading and credit rules and GPA.",
                     "zzz qqq unrelated gibberish tokens",
                     "plagiarism"]:
        assert hybrid.meets_floor(hybrid.retrieve(question)) is \
            retriever.meets_floor(retriever.retrieve(question))


def test_no_index_matches_semantic_only(hybrid, retriever):
    question = "Undergraduate grading and credit rules and GPA."
    plain = Retriever(FakeEmbedder(), hybrid.store, top_k=3, similarity_floor=0.5)
    assert [c.chunk_id for c in plain.retrieve(question)] == \
        [c.chunk_id for c in retriever.retrieve(question)]


def test_best_similarity_of_empty_is_zero():
    assert best_similarity([]) == 0.0
