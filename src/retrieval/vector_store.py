"""
vector_store.py — Persistent ChromaDB adapter for chunk storage and search.

Purpose:
    The single gateway between this project and ChromaDB. It (a) rebuilds
    the collection from chunks + vectors during ingestion, (b) answers
    similarity queries at chat time, and (c) refuses to serve queries if
    the collection was built with a different embedding model or is empty —
    turning two silent-failure modes into loud startup errors.

Inputs:
    Chunks (src.chunking.chunker.Chunk) with matching vectors for rebuild;
    a query vector (from Embedder.embed_query) with top-k for search.

Outputs:
    list[RetrievedChunk] — chunk data plus cosine similarity, best first.

Dependencies:
    chromadb (pinned >=0.5,<0.6 — its API has churned between minors),
    numpy, src.chunking.chunker.

Why this file exists:
    docs/vector_database.md selects ChromaDB for its combined storage of
    vectors + text + filterable metadata. Isolating all Chroma calls here
    (Architectural Decision: one adapter file) means an API change or a
    future store migration touches exactly one module. Two Chroma facts are
    encoded here rather than left for callers to know:
      * Chroma metadata values must be scalars, so list fields (provisions,
        pages) are stored as JSON strings and decoded on the way out.
      * With cosine space, Chroma returns DISTANCE (1 - similarity); this
        adapter converts back so the rest of the system speaks similarity,
        matching the similarity_floor semantics in settings.yaml.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Must be set BEFORE chromadb is imported: in 0.5.x the telemetry client can
# initialize at import time, ignoring later Settings(anonymized_telemetry=False).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
import numpy as np

# Chroma 0.5.x + newer posthog versions log a harmless "Failed to send
# telemetry" error on every operation even with telemetry disabled (known
# upstream incompatibility). Silence that specific logger; real Chroma
# errors surface through exceptions, not this logger.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from src.chunking.chunker import Chunk

log = logging.getLogger(__name__)

# Collection-level metadata keys recording how the index was built.
META_EMBEDDING_MODEL = "embedding_model"
META_DOCUMENT_ID = "document_id"

ADD_BATCH_SIZE = 128


@dataclass
class RetrievedChunk:
    """A chunk returned by similarity search, with its score and citation."""

    chunk_id: str
    text: str
    part: str
    section_number: str | None
    section_title: str | None
    provisions: list[str]
    pages: list[int]
    citation: str
    similarity: float             # cosine similarity in [-1, 1]; higher = closer


class StoreError(Exception):
    """Raised for store misuse that must halt startup (wrong model, empty)."""


class VectorStore:
    """Persistent, single-collection vector store backed by ChromaDB."""

    def __init__(self, persist_dir: Path | str, collection_name: str):
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            # A local, offline-first tool should not phone home; disabling
            # telemetry also silences noisy send-failure messages.
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )

    # --- Ingestion side ---------------------------------------------------

    def rebuild(
        self, chunks: list[Chunk], vectors: np.ndarray, embedding_model: str
    ) -> int:
        """Drop and recreate the collection from chunks and their vectors.

        Rebuilding (rather than upserting) makes ingestion idempotent: running
        it twice yields the same collection, never duplicates.

        Args:
            chunks: Chunk objects in the same order as `vectors` rows.
            vectors: (n, dim) array from Embedder.embed_texts(chunk texts).
            embedding_model: Model name recorded in collection metadata; the
                query side validates against it (see validate()).

        Returns:
            Number of records stored.

        Raises:
            StoreError: If chunks and vectors disagree in length.
        """
        if len(chunks) != vectors.shape[0]:
            raise StoreError(
                f"chunks ({len(chunks)}) and vectors ({vectors.shape[0]}) "
                "must align one-to-one"
            )

        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass  # first build: nothing to delete

        collection = self._client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                META_EMBEDDING_MODEL: embedding_model,
                META_DOCUMENT_ID: chunks[0].document if chunks else "",
            },
        )

        for start in range(0, len(chunks), ADD_BATCH_SIZE):
            batch = chunks[start:start + ADD_BATCH_SIZE]
            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors[start:start + len(batch)].tolist(),
                documents=[c.text for c in batch],
                metadatas=[_chunk_metadata(c) for c in batch],
            )
        log.info("Stored %d chunks in collection '%s'",
                 len(chunks), self.collection_name)
        return len(chunks)

    # --- Query side --------------------------------------------------------

    def validate(self, expected_model: str) -> None:
        """Fail loudly if the collection is missing, empty, or model-mismatched.

        Called once at chat startup. Querying an index built with a different
        embedding model returns garbage similarities with no error — this
        guard converts that silent failure into an instructive one.
        """
        try:
            collection = self._client.get_collection(self.collection_name)
        except Exception as exc:
            raise StoreError(
                f"Collection '{self.collection_name}' not found. "
                "Run scripts/run_ingestion.py first."
            ) from exc

        if collection.count() == 0:
            raise StoreError(
                f"Collection '{self.collection_name}' is empty. "
                "Run scripts/run_ingestion.py first."
            )

        built_with = (collection.metadata or {}).get(META_EMBEDDING_MODEL)
        if built_with != expected_model:
            raise StoreError(
                f"Collection was built with embedding model '{built_with}' "
                f"but settings.yaml specifies '{expected_model}'. "
                "Re-run scripts/run_ingestion.py to rebuild the index."
            )

    def query(self, vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        """Return the k most similar chunks, best first.

        Args:
            vector: (dim,) normalized query vector from Embedder.embed_query.
            k: Number of results (capped at the collection size by Chroma).

        Returns:
            RetrievedChunk list ordered by descending cosine similarity.
        """
        collection = self._client.get_collection(self.collection_name)
        result = collection.query(
            query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        return [
            _to_chunk(chunk_id, text, meta, 1.0 - float(distance))
            for chunk_id, text, meta, distance in zip(
                result["ids"][0], result["documents"][0],
                result["metadatas"][0], result["distances"][0],
            )
        ]

    def load_all(self) -> tuple[list[RetrievedChunk], np.ndarray]:
        """Every stored chunk plus its embedding, row-aligned.

        Read once at startup to build the BM25 keyword index (see
        src/retrieval/bm25_index.py). The embeddings come along so a chunk
        found by keyword alone can still be scored with its true cosine
        similarity against the query — without that, keyword hits would have
        no comparable score and the similarity floor would be meaningless.

        Returns:
            (chunks with similarity=0.0 as a placeholder, (n, dim) float32
            embedding matrix in the same order).
        """
        collection = self._client.get_collection(self.collection_name)
        result = collection.get(include=["documents", "metadatas", "embeddings"])

        chunks = [
            _to_chunk(chunk_id, text, meta, 0.0)
            for chunk_id, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]
        vectors = np.asarray(result["embeddings"], dtype=np.float32)
        return chunks, vectors

    def count(self) -> int:
        """Number of stored chunks (0 if the collection does not exist)."""
        try:
            return self._client.get_collection(self.collection_name).count()
        except Exception:
            return 0


def _to_chunk(chunk_id: str, text: str, meta: dict,
              similarity: float) -> RetrievedChunk:
    """Rebuild a RetrievedChunk from one Chroma record, decoding JSON fields."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        part=meta["part"],
        section_number=meta.get("section_number") or None,
        section_title=meta.get("section_title") or None,
        provisions=json.loads(meta["provisions_json"]),
        pages=json.loads(meta["pages_json"]),
        citation=meta["citation"],
        similarity=similarity,
    )


def _chunk_metadata(chunk: Chunk) -> dict:
    """Flatten a Chunk into Chroma-legal scalar metadata.

    Chroma metadata values must be str/int/float/bool — no lists — so
    provisions and pages are JSON-encoded; query() decodes them back. The
    pre-rendered citation string is stored so the chat layer never has to
    reconstruct citations from parts.
    """
    return {
        "part": chunk.part,
        "section_number": chunk.section_number or "",
        "section_title": chunk.section_title or "",
        "provisions_json": json.dumps(chunk.provisions),
        "pages_json": json.dumps(chunk.pages),
        "citation": chunk.citation(),
        "token_count": chunk.token_count,
    }
