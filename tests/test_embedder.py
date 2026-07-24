"""
test_embedder.py — Tests for src/embedding/embedder.py (Phase 5).

The wrapper's real responsibilities — applying the query prefix to queries
only, L2-normalizing every vector, preserving batch order, and staying
import-safe without ML libraries — are all tested with a deterministic fake
encoder (no model download, runs anywhere). One integration test loads the
actual bge-small model and is skipped automatically when it cannot be
downloaded (e.g. offline CI); it runs on a normal developer machine.

Dependencies:
    pytest, numpy, src.embedding.embedder.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedding.embedder import Embedder, _l2_normalize  # noqa: E402

DIM = 16


class FakeEncoder:
    """Deterministic hashing encoder: same text -> same vector, different
    text -> (almost surely) different vector. Records what it was asked to
    encode so tests can assert on prefixing."""

    def __init__(self):
        self.seen: list[str] = []

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            for j in range(DIM):
                out[i, j] = h[j] - 128  # spread around zero
        return out


def make_embedder():
    return Embedder(model_name="fake", query_prefix="QUERY: ", encoder=FakeEncoder())


def test_embed_texts_shape_and_order():
    emb = make_embedder()
    vecs = emb.embed_texts(["alpha", "beta", "gamma"])
    assert vecs.shape == (3, DIM)
    # Order preserved: re-embedding one text matches its row.
    single = emb.embed_texts(["beta"])[0]
    assert np.allclose(single, vecs[1])


def test_vectors_are_unit_norm():
    emb = make_embedder()
    vecs = emb.embed_texts(["some passage text", "another one"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    q = emb.embed_query("a question")
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-5)


def test_query_gets_prefix_passages_do_not():
    encoder = FakeEncoder()
    emb = Embedder("fake", "QUERY: ", encoder=encoder)
    emb.embed_texts(["plain passage"])
    emb.embed_query("plain passage")
    assert "plain passage" in encoder.seen            # passage: no prefix
    assert "QUERY: plain passage" in encoder.seen     # query: prefixed


def test_query_and_passage_differ_due_to_prefix():
    emb = make_embedder()
    passage_vec = emb.embed_texts(["leave of absence policy"])[0]
    query_vec = emb.embed_query("leave of absence policy")
    # Same underlying text, but the prefix makes the query embedding differ.
    assert not np.allclose(passage_vec, query_vec)


def test_empty_input_returns_empty():
    emb = make_embedder()
    out = emb.embed_texts([])
    assert out.shape[0] == 0


def test_l2_normalize_handles_zero_vector():
    zeros = np.zeros((1, 4), dtype=np.float32)
    out = _l2_normalize(zeros)
    assert np.all(out == 0)  # no divide-by-zero, stays zero


def test_lazy_load_not_triggered_when_encoder_injected():
    # Constructing and using an injected encoder must never import the model.
    emb = make_embedder()
    emb.embed_query("x")  # would raise on import if it tried to load the model


# --- Real model integration (skipped when the model can't be downloaded) -------

@pytest.mark.integration
def test_real_bge_model_if_available():
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception:
        pytest.skip("sentence-transformers not installed")

    try:
        emb = Embedder("BAAI/bge-small-en-v1.5",
                       "Represent this sentence for searching relevant passages: ")
        vecs = emb.embed_texts(["Plagiarism is a major offense."])
    except Exception as exc:  # no network to download weights, etc.
        pytest.skip(f"bge-small unavailable: {type(exc).__name__}")

    assert vecs.shape == (1, 384)                       # bge-small dimension
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0, atol=1e-4)
    # Relevance sanity: a related query should score higher than an unrelated one.
    related = emb.embed_query("What is the penalty for plagiarism?")
    unrelated = emb.embed_query("What time does the library open?")
    assert float(vecs[0] @ related) > float(vecs[0] @ unrelated)
