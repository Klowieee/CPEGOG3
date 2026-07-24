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

## Interface

- **GUI / web front end.** `ChatEngine.answer_question` is UI-agnostic; a
  Streamlit or Flask layer can call it directly (AC-4). Reuses the entire
  core unchanged.
- **Conversation memory.** Add history to the prompt and optionally use it to
  reformulate the retrieval query, enabling follow-ups. The prompt builder
  already takes a message list.

## Evaluation

- **Expand the golden set** to 60–100 questions with graded relevance, and
  track answer-correctness and citation-accuracy over changes.
- **Regression dashboard.** Record hit@k, false-answer, and false-refusal
  rates per change (chunking params, prompt, provider) — this becomes the
  results chapter of the paper.
