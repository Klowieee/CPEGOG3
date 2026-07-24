# Limitations

An honest account of what version 1 does not do well. Stating these plainly
is part of the project's methodology (and pre-empts the obvious defense
questions).

## Retrieval

- **Dense-only retrieval by default.** We match on meaning. A BM25 keyword
  channel is implemented (`src/retrieval/bm25_index.py`) and can be switched
  on in settings, but it ships **disabled**: measured on this corpus it did
  not improve retrieval and at full weight it made vague questions worse.
  See system_design.md §6 for the numbers.
- **No reranking.** The top-k are used as-is. A cross-encoder reranker would
  improve ordering, at some latency cost. The architecture leaves an
  insertion point for it.
- **The similarity floor cannot detect off-topic questions.** On a
  single-domain corpus every question is "about university rules" to a
  degree: measured, deliberately not-covered questions score 0.50–0.64
  against a 0.35 floor, so none of them are stopped there. The floor's real
  job is avoiding *false* refusals; deciding that the excerpts do not answer
  the question is Layer 2's (the model's). Raising the floor far enough to
  catch off-topic questions would refuse real ones first.
- **Vague questions cost more.** A casually-worded question that the model
  first answers `NOT_COVERED` triggers a rewrite call plus a second answer
  call. On Groq's free tier (6000 tokens/minute for this model) a burst of
  such questions can approach the rate limit — see the note in
  config/settings.yaml.

## Generation

- **Depends on an external API.** Answer generation calls a hosted model
  (Groq/Llama 3.1 8B by default). The retrieval stack is fully local, but
  generation needs a network and a key. A local SLM backend is designed for
  but not implemented (future_work.md).
- **Small model.** An 8B model is used deliberately (cost, speed, the "SLM"
  framing). It is generally faithful when handed the right excerpts, but is
  weaker than frontier models at nuanced multi-part questions.
- **Data leaves the machine at generation time.** The question and the
  retrieved handbook excerpts are sent to the API provider. The handbook is
  public, so exposure is minimal, but it is not zero.

## Document scope

- **One edition, English only.** The AY 2021–2025 Student Handbook, in
  English. Non-English questions are out of scope (the embedding model and
  corpus are English); they may be answered poorly rather than refused.
- **Parsing is tuned to this template.** Heading and provision detection use
  font sizes and column positions measured from this InDesign export. A
  differently-formatted future edition may need the thresholds in
  `src/ingestion/parser.py` re-checked (they are named constants for exactly
  this reason).
- **Tables and directories are approximate.** The appendix office/room
  directories are small-type tabular content; they are kept and chunked, but
  their line-by-line structure is flattened, so answers about them are less
  precise than answers about prose policy.

## Behavior

- **No conversation memory.** Follow-ups that rely on a previous question
  ("what about for graduate students?") are not understood in context; ask
  full questions.
- **No guarantee of completeness.** The bot answers from the top-k retrieved
  excerpts; if a rule is spread across many sections, it may cite only the
  most similar ones. It is an assistant for finding and explaining handbook
  content, not an authority — the handbook itself governs.
