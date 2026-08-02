# Terminal Chatbot

How the interactive chatbot behaves and how to run it.

## Running

```bash
uv run python scripts/run_ingestion.py   # once, to build the index
export GROQ_API_KEY=your-key             # a free key from the provider console
uv run python scripts/run_chat.py
```

The chatbot opens with a banner showing the handbook title and edition
(AY 2021–2025), then loops: you type a question, it prints an answer with
numbered sources, and repeats. Type `exit`, `quit`, or press Ctrl-C to leave.

## What a session looks like

```
┌──────────────────────────────────────────────┐
│ DLSU Student Handbook Assistant                │
│ Edition: AY 2021-2025                          │
│ Ask a question about the handbook. Type 'exit'.│
└──────────────────────────────────────────────┘

You: Is plagiarism a major offense?

Plagiarism is listed among the major offenses under the Lasallian Community
Standards, and is treated as a form of academic dishonesty [1].

Sources:
  [1] General Provisions, Section 5: LASALLIAN COMMUNITY STANDARDS, prov. 5.3.1.1.6, p. 60
```

## Behavior you can rely on

- **Grounded answers only.** The model is given the retrieved excerpts and
  instructed to answer solely from them. It cites each claim with an excerpt
  number; the application maps that number to the real section/provision/page,
  so citations cannot be hallucinated section numbers.
- **Polite refusal for anything not in the handbook.** Two independent layers:
  (1) if nothing retrieved is similar enough (below `retrieval.similarity_floor`),
  the app refuses *without* calling the API; (2) if excerpts were retrieved but
  don't actually answer, the model returns a refusal sentinel and the app shows
  the same polite message. Either way you get: *"That doesn't seem to be covered
  in the Student Handbook (AY 2021-2025). You may want to contact the relevant
  DLSU office directly."*
- **Graceful on failure.** A missing API key produces a clear instruction at
  startup, not a traceback. A network/API error during a question produces a
  short apology and keeps the loop alive.
- **Independent questions.** Version 1 does not use conversation history; each
  question is answered on its own (see docs/future_work.md).

## Tuning

Everything adjustable lives in `config/settings.yaml`: the retrieval
`top_k` and `similarity_floor`, the LLM `model`/`temperature`, and the
`refusal_message`. Use `scripts/eval_retrieval.py` to see how floor changes
affect answerable vs not-covered questions before editing it.
