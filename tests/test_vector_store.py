"""
test_vector_store.py — Tests for src/retrieval/vector_store.py (Phase 6).

These run against a REAL ChromaDB instance persisted in a pytest tmp_path —
Chroma is embedded and light enough that faking it would test nothing. What
IS faked are the vectors (deterministic hash vectors from the Phase 5 test
pattern), so no embedding model is needed. Covered: rebuild round-trip with
metadata fidelity, idempotent rebuild, similarity ordering, distance→
similarity conversion, list-metadata JSON round-trip, and all three
validate() failure modes.

Dependencies:
    pytest, numpy, chromadb, src.retrieval.vector_store, src.chunking.chunker.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.chunker import Chunk                          # noqa: E402
from src.retrieval.vector_store import StoreError, VectorStore  # noqa: E402

DIM = 32
MODEL = "fake-model-v1"


def vec(text: str) -> np.ndarray:
    """Deterministic unit vector for a text (same scheme as embedder tests)."""
    h = hashlib.sha256(text.encode()).digest()
    v = np.array([b - 128 for b in h[:DIM]], dtype=np.float32)
    return v / np.linalg.norm(v)


def make_chunks() -> list[Chunk]:
    return [
        Chunk("doc_0001", "Undergraduate › Section 10\nGrading rules text.",
              "doc", "Undergraduate", "10", "GRADING", ["10.1", "10.2"],
              [101, 102], 300),
        Chunk("doc_0002", "General Provisions › Section 5\nPlagiarism rules.",
              "doc", "General Provisions", "5", "STANDARDS", ["5.3.1.1.6"],
              [60], 250),
        Chunk("doc_0003", "Appendices\nRoom directory content.",
              "doc", "Appendices", None, None, [], [260], 200),
    ]


@pytest.fixture()
def store(tmp_path) -> VectorStore:
    chunks = make_chunks()
    vectors = np.stack([vec(c.text) for c in chunks])
    s = VectorStore(tmp_path / "db", "doc")
    s.rebuild(chunks, vectors, MODEL)
    return s


def test_rebuild_and_count(store):
    assert store.count() == 3


def test_rebuild_is_idempotent(tmp_path):
    chunks = make_chunks()
    vectors = np.stack([vec(c.text) for c in chunks])
    s = VectorStore(tmp_path / "db", "doc")
    s.rebuild(chunks, vectors, MODEL)
    s.rebuild(chunks, vectors, MODEL)      # second run must not duplicate
    assert s.count() == 3


def test_query_returns_exact_match_first_with_metadata(store):
    target = make_chunks()[1]              # the plagiarism chunk
    results = store.query(vec(target.text), k=3)
    top = results[0]
    assert top.chunk_id == "doc_0002"
    assert top.similarity == pytest.approx(1.0, abs=1e-4)  # identical vector
    # Metadata round-trip, including JSON-encoded lists and None section.
    assert top.part == "General Provisions"
    assert top.provisions == ["5.3.1.1.6"]
    assert top.pages == [60]
    assert top.citation.startswith("General Provisions, Section 5")
    appendix = next(r for r in results if r.chunk_id == "doc_0003")
    assert appendix.section_number is None and appendix.provisions == []


def test_results_ordered_by_descending_similarity(store):
    results = store.query(vec("Grading rules text query-ish"), k=3)
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)
    assert len(results) == 3


def test_persistence_across_instances(tmp_path):
    chunks = make_chunks()
    vectors = np.stack([vec(c.text) for c in chunks])
    VectorStore(tmp_path / "db", "doc").rebuild(chunks, vectors, MODEL)
    reopened = VectorStore(tmp_path / "db", "doc")   # new client, same dir
    assert reopened.count() == 3
    reopened.validate(MODEL)                          # metadata persisted too


def test_validate_passes_on_matching_model(store):
    store.validate(MODEL)                             # no exception


def test_validate_rejects_model_mismatch(store):
    with pytest.raises(StoreError, match="Re-run scripts/run_ingestion.py"):
        store.validate("some-other-model")


def test_validate_rejects_missing_collection(tmp_path):
    s = VectorStore(tmp_path / "db", "never-built")
    with pytest.raises(StoreError, match="not found"):
        s.validate(MODEL)


def test_rebuild_rejects_misaligned_inputs(tmp_path):
    chunks = make_chunks()
    wrong = np.stack([vec("only"), vec("two")])
    with pytest.raises(StoreError, match="one-to-one"):
        VectorStore(tmp_path / "db", "doc").rebuild(chunks, wrong, MODEL)


def test_load_all_returns_every_chunk_with_metadata(store):
    chunks, vectors = store.load_all()
    assert len(chunks) == 3
    assert vectors.shape == (3, DIM)

    by_id = {c.chunk_id: c for c in chunks}
    plagiarism = by_id["doc_0002"]
    assert plagiarism.section_number == "5"
    assert plagiarism.provisions == ["5.3.1.1.6"]     # JSON round-trip
    assert plagiarism.citation


def test_load_all_vectors_align_with_chunks(store):
    """Row order must match, or BM25 lookups would score the wrong chunk."""
    chunks, vectors = store.load_all()
    for chunk, vector in zip(chunks, vectors):
        assert np.dot(vector, vec(chunk.text)) == pytest.approx(1.0, abs=1e-5)
