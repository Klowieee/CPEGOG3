# Future Work

Concrete next steps, roughly in order of value-to-effort. Each is designed to
fit the existing architecture without a rewrite.

## Retrieval quality

- **Hybrid retrieval (BM25 + dense).** Add a keyword channel and fuse scores
  so exact terms (provision labels, acronyms like "LOA") retrieve reliably.
  Highest-value improvement; touches only `src/retrieval`.
- **Cross-encoder reranker.** Re-score the top-k (say top-20 → top-5) with a
  small reranker for better ordering. The retriever already centralizes
  ranking; insert between retrieve and prompt assembly.
- **Similarity-floor auto-tuning.** Sweep the floor over the golden set and
  pick the point that best separates answerable from not-covered; ship the
  resulting value. `scripts/eval_retrieval.py` already produces the data.

## Generation

- **Local SLM backend.** Implement `LocalBackend(LLMBackend)` wrapping Ollama
  (e.g. a quantized 3–8B model) so the whole system runs offline. This is the
  natural headline experiment: compare local vs API answers and latency on
  the GTX 1650 / CPU target. The `LLMBackend` seam means no other module
  changes.
- **Answer streaming.** Stream tokens to the terminal for snappier UX.

## Multi-document expansion

The metadata schema already carries `document`, so adding sources is additive:

- Academic calendar, registrar policies, scholarship handbook, thesis manual,
  college/department guidelines.
- Ingest each with its own `document` id; optionally filter retrieval by
  document, or let the user pick a scope. The vector store's metadata
  filtering supports this today.

## Course planning

The planner (Phase 15) is deliberately conservative about what it claims. The
next steps are all about claiming *more*, safely:

- **Course-offering awareness.** The single biggest gap: the planner can put a
  course in "next term" that the college is not offering. §10.20/§10.21 exist
  precisely because that happens. Ingesting a term offerings list (or scraping
  the enrolment system) would turn the plan from *permitted* into *possible*.
- **Parse the official program flowchart, not just the checklist.** The
  flowchart PDF states the prerequisite graph explicitly, where the checklist
  only lists it per row (or not at all). The obstacle is that its edges are
  drawn arrows, not text; box positions plus arrow endpoints would need geometric
  reconstruction. Worth attempting for programs whose checklist has no
  prerequisite column at all — the case that currently degrades to term ordering.
- **AND-of-ORs prerequisites (`schema_version: 2`).** Model *"GEMATMW or
  CSMATH1"* as a real disjunction instead of recording both as required. Today
  such a cell over-constrains (safe, but it delays courses unnecessarily) and is
  flagged in `extraction.warnings`. Deferred until a real checklist proves it
  common enough to justify the schema change.
- **Retention and standing (§10.17).** The planner reads grades only to decide
  what counts as passed. It could also warn about accumulated-failure limits and
  the 30%-units-remaining retention rule — with citations, as it already does
  for load limits.
- **Multi-program support.** One curriculum YAML per program is already the
  shape; a shifting student needs two, plus credit-transfer rules.
- **Optional LLM assist for a checklist with no prerequisite column.** The one
  place generation could earn its keep: send ≤25 rows per call and ask for
  `CODE: PREREQ1, PREREQ2` one per line (the plain-line format
  `src/chat/rewriter.py` proved, never JSON). It must stay opt-in, confined to
  `scripts/inspect_checklist.py`, and land in the YAML as
  `prereq_confidence: llm_guess` so the user sees every guess in a file they are
  already reviewing — the human is the validator, not the parser. The planner
  core stays LLM-free (AD-7).

## Interface

- **GUI / web front end.** `ChatEngine.answer_question` and
  `ChatEngine.plan_courses` are both UI-agnostic; a Streamlit or Flask layer can
  call them directly (AC-4). Reuses the entire core unchanged — the planner in
  particular needs no key and no network, so it would work in a purely local
  deployment.
- **Conversation memory.** Add history to the prompt and optionally use it to
  reformulate the retrieval query, enabling follow-ups. The prompt builder
  already takes a message list.

## Evaluation

- **Expand the golden set** to 60–100 questions with graded relevance, and
  track answer-correctness and citation-accuracy over changes.
- **Regression dashboard.** Record hit@k, false-answer, and false-refusal
  rates per change (chunking params, prompt, provider) — this becomes the
  results chapter of the paper.
