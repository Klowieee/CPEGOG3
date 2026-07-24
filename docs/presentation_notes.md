# Presentation Notes

A guide for presenting this project during a thesis/capstone defense. Focus on
the *decisions* and *evidence*, not the code.

## One-sentence summary

A local, terminal-based Retrieval-Augmented Generation chatbot that answers
questions about the 339-page DLSU Student Handbook and cites the exact section
each answer comes from, refusing when the handbook does not cover a question.

## The pipeline in one breath

Parse the PDF → clean out non-content → chunk into ~350-token section-aware
units with citation metadata → embed locally with bge-small → store in
ChromaDB. At question time: embed the question → retrieve the 5 most similar
chunks → if none are similar enough, refuse; otherwise ask a small LLM to
answer using only those chunks, citing them by number, which we map back to
real section citations.

## Why each major technology (be ready to defend)

- **RAG over fine-tuning:** the handbook is facts that must be quoted
  precisely and can change per edition; retrieval keeps answers grounded and
  updatable without retraining. (This project began as a GPT-2 fine-tune and
  was redirected to RAG for exactly this reason.)
- **pdfplumber:** headings are encoded by font size and provisions by column
  position; we need font/coordinate access, which plain text extraction lacks.
- **Section-aware chunking:** a policy document must not have rules split
  mid-sentence or chunks that mix sections, or citations break.
- **bge-small-en-v1.5:** retrieval-specialized, CPU-friendly, and its
  512-token input covers our max chunk (MiniLM's 256 would truncate).
- **ChromaDB:** the only embedded option that stores vectors + text +
  filterable metadata together, which the citations and multi-document
  roadmap need.
- **Groq / Llama 3.1 8B via API:** free, fast, and genuinely a *small*
  language model — the "SLM" framing stays honest; the backend is swappable.

## The strongest part of the story: design met reality

We validated assumptions against the actual document and corrected them —
show these:

1. **Section numbering.** We assumed section numbers restart per part. Parsing
   proved they are globally unique 1–21; what actually repeats is section
   *titles* (Section 10 and 17 are both "Credit, Grading and Retention"). We
   corrected the design and kept part-level metadata for the real reason.
2. **The grading scale isn't provisions.** "4.0 Excellent … 9.9 Deferred"
   sits in the left column and looks exactly like provision labels, producing
   the absurd citation "prov. 10.1–9.9". Fix: inside Section N a real label
   starts with "N."; table values are kept as text but never as citations.
3. **Wrapped heading & page metadata.** Section 14's title wraps across two
   lines (would have been dropped); an early splitter collapsed page numbers
   across multi-page passages (would have mis-cited). Both caught by tests and
   fixed.

The lesson to state out loud: *"We treated our own design as a hypothesis and
tested it against the source. Three assumptions were wrong; our test suite
caught them; we corrected them."* That is the methodology, not a weakness.

## Numbers to quote

- 339 pages → 10,978 cleaned content lines → **374 chunks** (median 340
  tokens, none over the 500 cap), covering all 21 sections and every content
  page.
- Vector store: full rebuild ~0.7 s; single query ~6 ms — retrieval is not the
  bottleneck; the API call dominates latency.
- **69 automated tests** pass; logic is tested with lightweight fakes so the
  suite runs without the 2 GB model or an API key, while marked integration
  tests exercise the real model/API on a developer machine.

## Two-layer refusal (a favorite question: "how do you prevent hallucination?")

1. **Retrieval floor** — if nothing clears the similarity threshold, refuse
   *before* calling the model. Free and hallucination-proof.
2. **Generation guard** — the model must answer only from the given excerpts
   and cite them; a reply that cites nothing resolvable is converted to a
   refusal (fail-closed). Section numbers are never written by the model — the
   app maps excerpt numbers to citations — so citations can't be invented.

## Live demo script (safe order)

1. An answerable discipline question ("Is plagiarism a major offense?") →
   show the answer and the section citation.
2. A duplicated-title question ("undergraduate grading rules") → show it cites
   Section 10, not the graduate Section 17.
3. A not-covered question ("campus wifi password") → show the polite refusal.
4. `scripts/eval_retrieval.py` → show hit@k on the golden set with no LLM
   involved.

## If asked "what would you do next?"

Hybrid retrieval (for exact terms), a local SLM backend (fully offline, and a
clean local-vs-API experiment), and multi-document expansion — the metadata
schema already supports it. See docs/future_work.md.
