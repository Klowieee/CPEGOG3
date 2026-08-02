# Vector Database Selection
## DLSU Student Handbook RAG Chatbot

**Prepared by:** Vector Database Specialist — v1.0, 20 July 2026

---

## 1. Requirements

- Fully local, embedded (no server process to manage) — this is a student laptop project.
- Persistent on disk (`data/vector_db/`) so ingestion runs once.
- **Metadata storage and filtering** — mandatory for the part/section citation scheme (AD-4) and the multi-document future (AC-3).
- Cosine similarity; scale target is only ~400 vectors now, ~2,000 later.
- Understandable and defensible.

## 2. Comparison

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **ChromaDB** | Embedded, persistent by default; stores text + metadata + vectors together; native metadata filtering (`where={"part": "Undergraduate"}`); minimal API (collection.add/query); very common in RAG literature → easy to defend | Heavier install than FAISS; young project, occasional API churn (pin the version) | **Selected** |
| FAISS | Fastest pure vector search; battle-tested | It is a vector *index*, not a database: no metadata, no persistence of documents — we would hand-build a metadata sidecar store, i.e., reimplement Chroma badly | Rejected |
| Qdrant | Excellent filtering, production-grade | Runs as a server (Docker) in its standard mode; operational overhead contradicts simplicity-first for a terminal app | Rejected |
| LanceDB | Embedded, nice columnar format | Smaller community and fewer references for a panel audience; no capability we need that Chroma lacks | Rejected |
| SQLite + manual cosine (numpy) | Zero new dependencies; genuinely sufficient for 400 vectors | Reinventing storage/query plumbing distracts from the thesis contribution; no filtering ergonomics | Rejected (but an honest footnote for the paper: at this scale it would work) |

## 3. Decision

**ChromaDB, persistent client, single collection per document, cosine distance.**

The deciding factor is not search speed — at ~400 vectors every option is instantaneous — but that Chroma is the only embedded option that stores **vectors, chunk text, and structured metadata together with filtering**, which the citation design and multi-document roadmap require. FAISS optimizes a problem this project does not have (vector count) while lacking the feature it does need (metadata).

## 4. Design Details

- Collection name: the `document` id (e.g., `student-handbook-2021-2025`); adding future documents = adding collections (or one collection filtered by `document` — decided at implementation of the second document; v1 keeps one collection).
- Collection metadata records the embedding model name and chunking parameters used to build it; the retriever validates this at startup (guards against querying with a mismatched model).
- Rebuild policy: `run_ingestion.py` drops and recreates the collection — idempotent by design.
- Access is isolated behind `src/retrieval/VectorStore`, so if Chroma's API churns or a migration is ever wanted, one file changes.

## 5. Resource Footprint

Persisted store for v1: a few MB on disk; query latency well under 100 ms on CPU. Not a bottleneck at any point on the expansion roadmap.

## 6. Implementation Notes (Phase 6)

Implemented in src/retrieval/vector_store.py as the single Chroma-touching
module. Verified behaviors: rebuild is idempotent (drop-and-recreate);
list-valued metadata (provisions, pages) is JSON-encoded to satisfy Chroma's
scalar-only metadata and decoded on query; cosine DISTANCE from Chroma is
converted to SIMILARITY so the retrieval floor in settings.yaml keeps its
documented meaning; validate() fails loudly at startup on a missing/empty
collection or an embedding-model mismatch (guarding the classic silent
garbage-results failure). Chroma 0.5.x telemetry is disabled and its known
posthog logging incompatibility silenced. Scale check with the real 374-chunk
corpus: full rebuild ~0.7 s; single query ~6 ms — confirming §5's claim that
the store is nowhere near a bottleneck.
