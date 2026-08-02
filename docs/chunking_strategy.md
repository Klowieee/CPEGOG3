# Chunking Strategy
## DLSU Student Handbook RAG Chatbot

**Prepared by:** Chunking Specialist — v1.0, 20 July 2026

Per project rules, no chunk sizes were assumed; the strategy below follows from measured properties of the actual document.

---

## 1. Document Analysis Findings

| Property | Measured value |
|---|---|
| Pages | 339 |
| Total corpus | ~84,400 words ≈ ~110–120K tokens |
| Structure | Three top-level parts (General Provisions, Undergraduate, Graduate) + appendices; 94 "Section N: Title" headings; hierarchical provisions numbered up to 7 levels deep (e.g., 5.3.1.1.6) |
| Verified structure | Parts are marked by chapter dividers: General Provisions (§1-6), Undergraduate (§7-13), Graduate (§14-20), Student Activities (§21), Appendices. Section numbers are **globally unique 1-21** in the main body — they do NOT restart per part (an early assumption, corrected by parsing the actual PDF in Phase 2). |
| Provision lengths (linearized-text estimate) | median ≈ 32 words; mean ≈ 385; p90 ≈ 365; long tail into thousands of words; ~44% of detected provisions under 30 words |
| Parsing caveat | Linearized text extraction detects only ~200 numbered headings; many heading numbers land mid-line after reflow → heading detection must use font metadata (pdfplumber), not text patterns alone |

## 2. Strategy Selected: Section-Aware Hierarchical Chunking with Merge/Split Normalization

**Rejected alternatives:**
- *Fixed-size chunking (e.g., 500 tokens, 50 overlap, structure-blind):* simple, but splits provisions mid-rule, destroys the provision→chunk mapping that citations depend on, and mixes text across part boundaries. Two real hazards make this unacceptable here: section *titles* repeat across parts (Section 10 "Credit, Grading and Retention" for Undergraduate vs Section 17 of the same title for Graduate; "Graduation" is §12 and §19), and the Appendices restart provision numbering (labels like 5.x reappear), so part metadata is required to keep citations unambiguous. FR-3 requires precise citations.
- *Pure heading-based chunking (one chunk per numbered provision):* respects structure but the measured length distribution breaks it — 44% of provisions are under 30 words (retrieval-useless fragments lacking context) while the largest sections run thousands of words (exceed any sensible embedding input).
- *Semantic chunking (embedding-similarity breakpoints):* adds model-driven nondeterminism and is hard to explain in a defense; the document already provides better boundaries than a similarity heuristic would find.

**Selected approach — three passes:**

1. **Structural segmentation.** Using font-size/weight from pdfplumber, build the document tree: part → section → numbered provision. Every leaf inherits the full ancestor path as metadata.
2. **Merge pass.** Provisions under **80 tokens** are merged with their siblings under the nearest common parent heading until the group reaches the target size, keeping the parent heading text inside the chunk (so a chunk about "5.3.1.1.6 Plagiarism" also contains the framing "5.3.1.1 Major offenses…" — the fragment alone is meaningless without it). The provision list is preserved in metadata.
3. **Split pass.** Segments over **500 tokens** are split at paragraph boundaries targeting **~350 tokens**, with **50-token overlap** between consecutive splits so a rule straddling a boundary remains retrievable. Split chunks share the same section metadata and carry a `part_index`.

**Heading context injection:** every chunk's text is prefixed with its breadcrumb (`Undergraduate › Section 10: Credit, Grading, and Retention`). This measurably helps dense retrieval because the embedding then encodes topical context that short provisions lack, and it costs ~10 tokens.

## 3. Parameter Summary and Justification

| Parameter | Value | Why |
|---|---|---|
| Target chunk size | ~350 tokens | Large enough to hold a complete rule with context; small enough that 5 retrieved chunks ≈ 1,750 tokens keeps the prompt focused; well within bge-small's 512-token input limit |
| Max chunk size | 500 tokens | Hard cap ≈ embedding model input limit with margin for the breadcrumb prefix |
| Min chunk size (merge threshold) | 80 tokens | Below this, chunks are context-free fragments (median provision is only ~32 words) |
| Overlap (split pass only) | 50 tokens | Protects boundary-straddling rules; overlap between *unrelated sections* is deliberately zero — overlapping across a section boundary would attach wrong citations |

## 4. Estimates

| Quantity | Estimate | Basis |
|---|---|---|
| Number of chunks | ~350–450 | ~115K tokens ÷ ~300 effective tokens/chunk after merge/split |
| Embedding storage | < 1 MB | ~400 × 384 dims × 4 bytes ≈ 0.6 MB (+ text/metadata ≈ a few MB in ChromaDB) |
| Ingestion time | Minutes on CPU | bge-small embeds hundreds of short texts per minute on CPU |
| Retrieval latency | < 100 ms | Brute-force cosine over ~400 vectors is trivial |

Consequence worth stating in the thesis: at this corpus size the vector search itself is effectively free; end-to-end latency is dominated by the generation API call.

## 5. Validation Plan

- `chunks.jsonl` is human-inspectable; a spot-check script prints N random chunks with metadata.
- Automated checks: no chunk exceeds max tokens; every chunk has part + section metadata; chunk count within the estimated band; every handbook section is represented by ≥1 chunk.
- Retrieval sanity: the golden question set (testing.md) verifies that known provisions surface for their canonical questions.

## 6. Measured Results (Phase 4 implementation)

Verified on the actual handbook after implementation: **374 chunks**
(within the estimated 350–450 band); token sizes min 42 / median 340 /
max 462 (cap respected); all 21 sections represented; every cleaned
content page covered by at least one chunk. Two implementation findings:
(1) one section heading wraps across lines ("SECTION 14: FEES,
SCHOLARSHIPS, AND / PAYMENTS") and required a heading-continuation merge
pass in the parser; (2) the appendix room-directory tables contain almost
no sentence punctuation, so the split pass operates at line granularity —
which also preserves true per-line page numbers for citations across long
multi-page passages.
