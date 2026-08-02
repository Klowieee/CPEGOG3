"""
test_bm25_index.py — Tests for src/retrieval/bm25_index.py.

Covers tokenization, keyword ranking, the zero-score cutoff, and the cosine
rescoring that lets a keyword-only hit carry a score comparable to a semantic
one (the property meets_floor depends on).

Dependencies:
    pytest, numpy, chromadb, src.retrieval.*, src.chunking.chunker.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.chunker import Chunk                         # noqa: E402
from src.retrieval.bm25_index import BM25Index, tokenize       # noqa: E402
from src.retrieval.vector_store import RetrievedChunk, VectorStore  # noqa: E402


def _retrieved(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, part="doc",
                          section_number="5", section_title="STANDARDS",
                          provisions=["5.3"], pages=[60],
                          citation=f"cite-{chunk_id}", similarity=0.0)


def _unit(*values) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def index():
    chunks = [
        _retrieved("c1", "Plagiarism is a major offense."),
        _retrieved("c2", "Grading and credit rules for undergraduates."),
        _retrieved("c3", "Comprehensive examination requirements."),
    ]
    embeddings = np.stack([_unit(1, 0), _unit(0, 1), _unit(1, 1)])
    return BM25Index(chunks, embeddings)


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Plagiarism, a MAJOR offense!") == [
        "plagiarism", "a", "major", "offense"]


def test_tokenize_keeps_digits():
    assert tokenize("provision 5.3.1") == ["provision", "5", "3", "1"]


def test_search_ranks_exact_term_first(index):
    assert index.search("plagiarism", 3)[0] == "c1"


def test_search_excludes_zero_score_chunks(index):
    # Only c1 contains "plagiarism"; the others must not be padded in.
    assert index.search("plagiarism", 3) == ["c1"]


def test_search_returns_nothing_for_unmatched_query(index):
    assert index.search("zzz qqq", 3) == []


def test_search_handles_empty_query(index):
    assert index.search("", 3) == []


def test_lookup_scores_with_true_cosine(index):
    # Query vector identical to c1's embedding -> cosine 1.0.
    assert index.lookup("c1", _unit(1, 0)).similarity == pytest.approx(1.0)
    # Orthogonal to c1 -> cosine 0.0.
    assert index.lookup("c1", _unit(0, 1)).similarity == pytest.approx(0.0)


def test_lookup_does_not_mutate_the_cached_chunk(index):
    index.lookup("c1", _unit(1, 0))
    assert index.lookup("c1", _unit(0, 1)).similarity == pytest.approx(0.0)


def test_lookup_preserves_metadata(index):
    chunk = index.lookup("c2", _unit(0, 1))
    assert chunk.chunk_id == "c2"
    assert chunk.citation == "cite-c2"
    assert chunk.provisions == ["5.3"]


def test_from_store_round_trip(tmp_path):
    # Three chunks, not two: BM25 inverse document frequency is zero for a
    # term carried by half the corpus, so a two-document fixture would score
    # every query at zero regardless of the code under test.
    chunks = [
        Chunk("d_1", "Plagiarism is a major offense under discipline rules.",
              "doc", "General Provisions", "5", "STANDARDS", ["5.3"], [60], 200),
        Chunk("d_2", "Undergraduate grading and credit rules.",
              "doc", "Undergraduate", "10", "GRADING", ["10.1"], [101], 200),
        Chunk("d_3", "Graduate comprehensive examination requirements.",
              "doc", "Graduate", "16", "COMPS", ["16.1"], [118], 200),
    ]
    vectors = np.stack([_unit(1, 0), _unit(0, 1), _unit(1, 1)])
    store = VectorStore(tmp_path / "db", "doc")
    store.rebuild(chunks, vectors, "fake")

    index = BM25Index.from_store(store)
    assert len(index) == 3
    assert index.search("plagiarism", 3) == ["d_1"]
    assert index.lookup("d_1", _unit(1, 0)).similarity == pytest.approx(1.0, abs=1e-5)
