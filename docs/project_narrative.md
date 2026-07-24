# Project Narrative — DLSU Student Handbook RAG Chatbot

A single, paper-shaped account of the project: what problem it solves, how it
is built, which design assumptions survived contact with the real document,
and what the evaluation measured. It is written to be self-contained — a
research paper could be drafted from this file alone. Where a claim is a
*measured* result it is labelled as such; where it is a *design guarantee* or
an unmeasured target, that is stated explicitly, so nothing here should be
reported as a result that was not actually observed.

Companion design docs (deeper rationale per component): `architecture.md`,
`chunking_strategy.md`, `embedding_strategy.md`, `vector_database.md`,
`prompting.md`, `rag_pipeline.md`, `system_design.md`, `testing.md`,
`limitations.md`, `future_work.md`.

---

## 1. Abstract

We present a local, terminal-based Retrieval-Augmented Generation (RAG)
assistant that answers natural-language questions about the 339-page **DLSU
Student Handbook (AY 2021–2025)** and cites the exact handbook section each
answer is drawn from, refusing when the handbook does not cover a question.
All retrieval machinery (parsing, chunking, embedding, vector search) runs
locally; only answer generation calls a hosted small language model (Llama
3.1 8B via the Groq API), behind a swappable backend interface. Beyond the
baseline system, the project contributes three empirically-grounded findings
specific to a **single-domain policy corpus**: (i) a global cosine-similarity
floor cannot separate covered from not-covered questions, because on a
single-domain corpus even off-topic questions score well above any usable
threshold; (ii) vague, casually-worded questions fail not at retrieval but at
generation, and can be rescued by rewriting the question into the corpus's
own vocabulary *when the model itself signals non-coverage*; and (iii) a
BM25+dense hybrid, the textbook fix for exact-term retrieval, did not improve
retrieval on this corpus and degraded vague-question recall at full weight,
so it ships implemented but disabled — a measured negative result.

---

## 2. Motivation and the Pivot

Students cannot realistically read a 339-page handbook to find one rule, and a
general chatbot will confidently invent university policy. The requirement is
answers that are **grounded, quotable, and citable** to a specific section,
and that **refuse** rather than guess.

The project began as a **GPT-2 + LoRA fine-tuning** effort — teach a small
language model the handbook's content directly. It was redirected to RAG after
adviser guidance, for reasons that are themselves a defensible research point:

- Handbook content is **facts that must be quoted precisely**, not a style to
  be imitated. Fine-tuning blends knowledge into weights, where it cannot be
  quoted verbatim or attributed to a section.
- The handbook **changes per edition**. RAG re-indexes a new PDF in seconds;
  fine-tuning requires retraining.
- **Citations and refusal** — the two hard requirements — are natural in RAG
  (cite the retrieved chunk; refuse when nothing is retrieved) and awkward in a
  pure generative model.

The pivot is thus not an incidental detour but the paper's framing: for a
citable-rules domain, retrieval-grounded generation dominates
fine-tuned generation on precisely the axes that matter (attribution,
updatability, refusal).

---

## 3. System Design

Two independent pipelines share one vector store (`architecture.md` AD-1):

**Offline ingestion (run once per edition).** Parse the PDF with `pdfplumber`
(font size and column position recover headings and provision labels that
plain text extraction loses) → clean out non-content → chunk into ~350-token
**section-aware** units, each tagged with `document`, `part`, `section`, and
`provision` metadata → embed locally with `bge-small-en-v1.5` (384-dim,
retrieval-specialized, CPU-friendly, 512-token input covers the max chunk) →
persist in ChromaDB (the one embedded store holding vectors + text +
filterable metadata together).

**Online query (interactive).** Embed the question → retrieve the top-*k*
chunks by cosine similarity → assemble a grounded prompt → call the LLM behind
an `LLMBackend` interface → map the model's `[n]` markers back to real section
citations → render in the terminal, or refuse.

Load-bearing design decisions:

- **Part-level metadata is mandatory (AD-4).** Section *numbers* are globally
  unique 1–21, but section *titles* repeat across parts ("Credit, Grading and
  Retention" is §10 Undergraduate and §17 Graduate; "Graduation" is §12 and
  §19), and the Appendices restart provision numbering. Without `part`,
  citations for duplicated titles are ambiguous.
- **The model never writes section numbers.** It cites excerpt numbers `[n]`;
  the application maps them to structured citations. This eliminates the most
  common citation-hallucination mode by construction, not by prompting.
- **The generator is swappable (AD-3).** `LLMBackend` isolates the provider;
  a local Ollama backend is a drop-in with no other changes.

Corpus statistics (measured): 339 pages → 10,978 cleaned content lines →
**374 chunks** (median 340 tokens, none over the 500-token cap), covering all
21 sections and every content page. Full index rebuild ≈ 0.7 s; single query
≈ 6 ms — retrieval is not the latency bottleneck; the API call dominates.

---

## 4. Design Met Reality (Methodology)

The strongest methodological point is that the design was treated as a
hypothesis and tested against the actual document. Three assumptions were
wrong; the test suite caught them; they were corrected:

1. **Section numbering.** Assumed to restart per part; parsing proved it is
   globally unique 1–21. What actually repeats is section *titles*. The design
   kept part-level metadata for the real reason (title collisions), not the
   assumed one.
2. **The grading scale is not provisions.** "4.0 Excellent … 9.9 Deferred"
   sits in the left column and mimics provision labels, producing the absurd
   citation "prov. 10.1–9.9". Fix: inside Section N a real label starts with
   "N."; table values are retained as text but never as citations.
3. **Wrapped headings and page spans.** Section 14's title wraps two lines
   (nearly dropped); an early splitter collapsed page numbers across
   multi-page passages (would mis-cite). Both caught by tests and fixed.

The lesson to state plainly: *we tested our own design against the source,
three assumptions failed, and the suite caught them.* That is the methodology,
not a weakness.

---

## 5. Grounding and the Two-Layer Refusal

Hallucination is prevented by two layers with one shared, deterministic
user-facing refusal message (`prompting.md`):

1. **Retrieval floor.** If the best similarity is below a threshold (0.35),
   refuse *before* calling the model — cost-free and hallucination-proof.
2. **Generation guard.** The system prompt instructs the model to answer only
   from the numbered excerpts and to emit the sentinel `NOT_COVERED` when they
   do not answer the question. A reply that resolves *no* valid citation
   marker is treated as a grounding failure and (after one corrective retry)
   flagged rather than trusted. The refusal text lives in application code, so
   the behavior is deterministic and testable on a single string.

The current system prompt deliberately allows **partial coverage** — if the
excerpts answer the question even partially or the answer is reasonably
inferable, the model answers with what it has, and reserves `NOT_COVERED` for
excerpts that are genuinely unrelated. This matters for §6.2 below: it is the
line between "the handbook does not cover this" and "the retrieved excerpts,
this time, did not contain it."

---

## 6. Contributions: Findings on a Single-Domain Corpus

The three findings below are what elevate the project from "a working RAG
demo" to something with a claim. All numbers are measured on the current
374-chunk index with `bge-small-en-v1.5`, `top_k = 8`, floor `0.35`.

### 6.1 A global similarity floor cannot detect off-topic questions

The two-layer design assumes Layer 1 (the floor) catches not-covered
questions cheaply. On a single-domain corpus it does not. Measured best
similarity for five deliberately out-of-scope questions:

| Not-covered question | Best cosine similarity |
|---|---|
| How much is the tuition per unit in pesos? | 0.61 |
| What is the wifi password for the campus network? | 0.60 |
| Who is the best professor for calculus? | 0.52 |
| What are the dormitory room rental prices? | 0.57 |
| What time does the library close on Saturdays? | 0.64 |

Every one clears the 0.35 floor: on a corpus that is entirely "about
university rules," even off-topic questions are lexically and semantically
adjacent to *something*. Raising the floor to catch them (it would need to
exceed ~0.65) refuses genuinely answerable questions first — real questions
score 0.62–0.81. **Conclusion:** on a single-domain corpus the floor's only
achievable job is avoiding *false refusals*; deciding that retrieved excerpts
do not answer the question must fall to Layer 2 (the model). This directly
qualifies the standard "reject below a similarity threshold" recipe, which
implicitly assumes a corpus broad enough for off-topic queries to score low.

### 6.2 Vague questions fail at generation, not retrieval — rescue on refusal

A casually-worded question ("what happens if I copy someone's homework") is
covered by the handbook but phrased nothing like it ("academic dishonesty,"
"major offense," "sanction"). The intuitive hypothesis is that such questions
fail at retrieval. Measured, they do **not**: 6/6 vague phrasings in the
golden set retrieve their correct section within the top 8. What fails is that
the retrieved excerpts land *beside* the rule rather than *on* it — e.g. the
homework question surfaces mostly Appendix boilerplate plus one true Section 5
chunk — and the model, correctly, answers `NOT_COVERED`.

Because retrieval never dips below the floor (see §6.1), a floor-triggered
rewrite would never fire. The mechanism instead triggers on the **model's own
`NOT_COVERED`**: that refusal is the honest signal that retrieval missed. On
that signal the system spends one small LLM call rewriting the question into
handbook vocabulary (1–3 formal search queries), retrieves again, merges with
the original chunks (keeping the best cosine per chunk), and asks the model
once more. If it refuses again, the refusal stands.

Properties that make this safe and cheap:

- **Cost is incurred only on refusal.** Questions answered on the first pass
  never touch the rewriter — the common path is unchanged.
- **Fail-closed and non-recursive.** The rewriter catches all LLM errors and
  returns no queries on failure; a failed rewrite degrades to the original
  refusal rather than crashing. One rewrite, one retry, maximum.
- **No wasted calls.** If the rewrite retrieves nothing new, the second
  answer call is skipped — re-prompting with identical excerpts cannot change
  the outcome.

This reframes query rewriting from a blind pre-retrieval expansion into a
**targeted, model-signalled rescue**, which is both cheaper (it fires rarely)
and better-motivated (it fires exactly when the system has evidence it needs
to).

### 6.3 Hybrid BM25+dense retrieval — a measured negative result

Hybrid retrieval (a keyword channel fused with dense retrieval) is the
textbook fix for exact-term queries, and was the top-listed future-work item.
It was implemented (`src/retrieval/bm25_index.py`, reciprocal rank fusion,
keyword-only hits rescored with true cosine so the floor stays meaningful) and
measured. Two things emerged:

1. The motivating case had expired. Semantic search **already** retrieves the
   plagiarism provision (5.3.1.1.6) in the top 8; the reported miss no longer
   reproduced.
2. Sweeping the fusion weight over a 10-question answerable set and a
   10-question vague probe set showed hybrid does not pay for itself:

| keyword_weight | answerable hit@8 | answerable hit@1 | vague hit@8 |
|---|---|---|---|
| 0.0 (dense only) | 10/10 | 8/10 | 8/10 |
| 0.25 – 0.75 | 10/10 | 9/10 | 8/10 |
| 1.0 (standard RRF) | 10/10 | 9/10 | **7/10** |

A marginal answerable hit@1 gain (8→9) is offset by a vague-recall loss at
full weight: with only `top_k` slots, keyword hits evict good semantic
neighbours. The feature ships **disabled**, configurable, with the
reproduction in `scripts/eval_retrieval.py`. A negative result, honestly
measured and retained, is a stronger contribution than shipping the feature
would have been — it says the standard recipe does not transfer to a small
single-domain corpus where dense retrieval already saturates recall.

---

## 7. Evaluation

**Methodology.** Retrieval quality is tested **without** the LLM: if retrieval
fails, generation cannot succeed, and LLM-free tests are free, fast, and
deterministic (`testing.md`). A golden question set (`tests/golden_set.yaml`)
is graded in four classes: *answerable* (question + expected section),
*vague_answerable* (the same substance in a student's words), *not_covered*
(must refuse), and *adversarial* (empty input, injection, non-English).

**Measured results (current index, dense-only, top_k = 8).**

| Metric | Result |
|---|---|
| Answerable retrieval hit@8 | 10/10 (100%) |
| Answerable retrieval hit@1 | 8/10 (80%) |
| Vague retrieval hit@8 (reaches correct section) | 6/6 (100%) |
| Not-covered questions below the 0.35 floor | 0/5 (all clear it — see §6.1) |
| Automated tests passing | 130 (2 live-API integration tests skipped without a key) |

**Design-guaranteed, not statistically measured.** Citation accuracy: the
model never emits section numbers (§3), so a rendered citation always
corresponds to a real retrieved chunk. False-answer-from-fabrication is
structurally prevented; whether a *correct-form* citation truly supports each
sentence is a manual-rubric item (`testing.md` §4) not yet scored at scale.

**Not yet measured (targets only).** End-to-end answer correctness (rubric
target ≥ 85%), latency p50/p95, and false-refusal rate under the rewrite
rescue require a live API key and a larger graded set; `testing.md` §4 defines
them and `scripts/eval_retrieval.py` produces the retrieval half offline.

---

## 8. Limitations

- **Dense-only by default; keyword channel disabled** (§6.3). Exact-term
  queries rely on dense retrieval already saturating recall on this corpus —
  true here, not guaranteed on a larger one.
- **The floor cannot gate off-topic questions** (§6.1); off-topic refusal
  depends entirely on the model reading the excerpts.
- **Vague questions cost more.** A refused-then-rescued question spends a
  rewrite call plus a second answer call (~9–10k tokens); Groq's free tier
  caps this model at 6000 tokens/minute, so a burst of vague questions can
  rate-limit. `top_k` is the lever to shrink the prompt if needed.
- **External generation.** Retrieval is fully local; generation needs a
  network and a key. A local SLM backend is designed for but not implemented.
- **Small model, single document, English-only, no conversation memory** —
  each documented honestly in `limitations.md` rather than over-engineered.

---

## 9. Future Work

In value-to-effort order (`future_work.md`), updated for what this iteration
settled:

- **Local SLM backend (Ollama).** The natural headline experiment: local vs
  API answer quality and latency on the target hardware, with zero changes
  outside `src/llm`. Now the highest-value open item (hybrid retrieval, the
  former #1, has been tried and shelved with evidence).
- **Cross-encoder reranker.** Re-score top-20 → top-8 for ordering; the
  retriever centralizes ranking, so it inserts cleanly.
- **Multi-document expansion.** The metadata schema already carries
  `document`; adding sources is additive.
- **GUI/web front end and conversation memory.** `answer_question` is
  UI-agnostic; the prompt builder already takes a message list.
- **Expanded graded golden set + regression dashboard** — turns the
  before/after discipline into the results chapter directly.

---

## 10. Reproducibility

```bash
uv sync                                   # environment (incl. rank-bm25, pytest)
export GROQ_API_KEY="..."                 # generation only; retrieval needs no key
uv run python scripts/run_ingestion.py    # build the 374-chunk index (once)
uv run python scripts/eval_retrieval.py   # retrieval metrics + hybrid/vague report (no LLM)
uv run pytest                             # 130 tests; live-API ones skip without a key
uv run python scripts/debug_question.py "what happens if I copy someone's homework"
```

`scripts/debug_question.py` traces a single question through both refusal
layers and the rewrite rescue, printing per-chunk similarities and the
rewritten queries — the tool behind the §6 findings.

---

## 11. Suggested Framing for the Paper

Load-bearing claims, each backed by a measurement or a design argument above:

1. **For a citable-rules domain, RAG dominates SLM fine-tuning** on
   attribution, updatability, and refusal (§2).
2. **A global similarity floor is not a coverage detector on a single-domain
   corpus** — off-topic questions score 0.52–0.64 against real questions'
   0.62–0.81 (§6.1). This qualifies a widely-used RAG refusal recipe.
3. **Vague-question failure is a generation failure, not a retrieval
   failure**, and is best rescued by model-signalled query rewriting rather
   than blind pre-retrieval expansion (§6.2).
4. **Textbook hybrid retrieval does not transfer to a small single-domain
   corpus where dense recall already saturates** — a measured negative result
   (§6.3).

Ready-made figures: the floor separation plot (real vs off-topic similarity
distributions, §6.1); the fusion-weight ablation table (§6.3); the
refuse→rewrite→retry sequence diagram (`system_design.md` §3); and the
design-met-reality corrections table (§4) as a methodology exhibit.
