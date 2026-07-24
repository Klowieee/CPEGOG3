# Embedding Strategy
## DLSU Student Handbook RAG Chatbot

**Prepared by:** Embedding Specialist — v1.0, 20 July 2026

---

## 1. Requirements on the Embedding Model

- Runs locally on CPU within the hardware envelope (16 GB RAM, frequent CPU-only) — embeddings stay local by design (AD-2), so this is non-negotiable.
- English-only (NFR-7) — multilingual capability is unnecessary weight.
- Strong *retrieval* performance (asymmetric question→passage matching), not just sentence similarity.
- Input window ≥ our max chunk size (500 tokens + breadcrumb).
- Simple to use via `sentence-transformers`.

## 2. Candidates Compared

| Model | Dims | Size | Input limit | Assessment |
|---|---|---|---|---|
| **BAAI/bge-small-en-v1.5** | 384 | ~130 MB | 512 tok | Purpose-built for retrieval; consistently near the top of its size class on retrieval benchmarks (MTEB); supports a query instruction prefix that improves question→passage matching |
| all-MiniLM-L6-v2 | 384 | ~90 MB | 256 tok | The classic tutorial default; **256-token input limit is disqualifying** for 350–500-token chunks (silent truncation would corrupt the index); general similarity training, weaker at retrieval |
| bge-base-en-v1.5 | 768 | ~440 MB | 512 tok | Marginal quality gain over small on a 400-chunk corpus; 2× storage and slower CPU inference — unjustified here |
| nomic-embed-text-v1.5 | 768 | ~550 MB | 8192 tok | Long context we don't need; heavier; remote-code trust flag in some setups |
| API embeddings (OpenAI etc.) | — | — | — | Violates AD-2 (retrieval fully local), adds cost and a second external dependency; rejected on principle, not capability |

## 3. Recommendation

**BAAI/bge-small-en-v1.5.**

Reasoning, in order of importance:
1. **Fits the corpus and chunking design exactly** — 512-token input covers the 500-token max chunk plus breadcrumb; MiniLM's 256 limit does not.
2. **Retrieval-specialized** — trained for question→passage matching, which is precisely the RAG workload; measurably stronger than MiniLM in its class.
3. **CPU-practical** — ~130 MB, embeds the entire ~400-chunk corpus in minutes and a single query in tens of milliseconds on the target CPU.
4. **Defensible** — well-documented, widely cited, easy to justify to a panel.

**Usage details:**
- Passages embedded as-is; queries embedded with the model's recommended instruction prefix ("Represent this sentence for searching relevant passages: …") — implemented inside the `Embedder` wrapper so callers cannot get it wrong.
- Vectors L2-normalized; cosine similarity in ChromaDB.
- Model name is config-driven; changing it requires re-running ingestion (the system will store the model name in the collection metadata and refuse to query a collection built with a different model — a cheap guard against a classic silent-failure bug).

## 4. Resource Estimates

| Quantity | Estimate |
|---|---|
| Model memory (loaded) | ~300–400 MB RAM |
| Index size (~400 chunks × 384-d float32) | ~0.6 MB vectors; a few MB total with text + metadata |
| Corpus embedding time (CPU) | Single-digit minutes, one-time |
| Query embedding latency (CPU) | ~10–50 ms |

## 5. Scalability

Adding every document listed under Future Expansion (calendar, registrar policies, scholarship handbook, thesis manual…) would plausibly multiply the corpus ~5×, to ~2,000 chunks — still trivially within this model's and ChromaDB's comfortable range on the target hardware. The embedding choice therefore does not constrain the expansion roadmap.

## 6. Implementation Note (Phase 5)

The `Embedder` (src/embedding/embedder.py) enforces the two rules above in a
single place: the query instruction prefix is applied to queries only, and
every vector is L2-normalized (so the vector store's cosine similarity is a
plain dot product). The sentence-transformers model is loaded lazily and the
encoder is injectable, so the wrapper's logic is unit-tested with a
deterministic fake encoder — no 2 GB download needed for CI — while a marked
integration test exercises the real bge-small model on a developer machine
and confirms the 384-dim output and that a related query out-scores an
unrelated one.
