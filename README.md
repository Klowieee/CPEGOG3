# DLSU Student Handbook RAG Chatbot

A local, terminal-based chatbot that answers natural-language questions about
the **DLSU Student Handbook (AY 2021–2025)** using Retrieval-Augmented
Generation. Retrieval (embeddings + vector search) runs fully on your machine;
answer generation uses a hosted small language model via API.

Undergraduate academic project. Design rationale for every component is in
[`docs/`](docs/) — start with `architecture.md`.

## How It Works (one paragraph)

An offline ingestion pipeline parses the handbook PDF, cleans it, splits it
into ~350-token section-aware chunks (each tagged with its part, section, and
provision numbers), embeds them locally with `bge-small-en-v1.5`, and stores
them in ChromaDB. At chat time, your question is embedded, the 8 most similar
chunks are retrieved, and a small LLM (Llama 3.1 8B via the Groq API) answers
**using only those excerpts**, citing them by number; the app maps those
numbers back to real handbook sections. If the excerpts don't answer the
question, it refuses politely instead of guessing — and if the question was
simply phrased unlike the handbook, it rewrites the question into handbook
wording and tries once more before refusing.

The design rationale and the empirical findings behind these choices are
consolidated in [`docs/project_narrative.md`](docs/project_narrative.md).

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). `uv` reads the
dependencies from `pyproject.toml` and manages the virtual environment for you.

```bash
uv sync                  # creates .venv and installs everything (incl. pytest)
```

> The first sync downloads PyTorch (via sentence-transformers, ~2 GB) — run it
> on good wifi. `uv` caches packages globally, so later syncs are fast.

Place the handbook PDF at `data/handbooks/student-handbook.pdf`.

Set your API key (obtain a free key from the provider's console):

```bash
export GROQ_API_KEY="..."        # Windows (PowerShell): $env:GROQ_API_KEY="..."
```

The key is read from the environment only — never write it into any file in
this repository. See `.env.example`.

## Usage

`uv run` executes a command inside the project environment without needing to
activate it manually:

```bash
uv run python scripts/run_ingestion.py   # one-time: build the vector index
uv run python scripts/run_chat.py        # start the chatbot
```

(Prefer activating the venv? `source .venv/bin/activate` — Windows:
`.venv\Scripts\activate` — then run `python scripts/...` directly.)

Inspect the parser output at any time (Phase 2):

```bash
uv run python scripts/inspect_parse.py                 # role counts for the whole PDF
uv run python scripts/inspect_parse.py --pages 101 102 # role-tagged view of pages 101-102
```

Evaluate retrieval quality against the golden set (no LLM needed, Phase 7/11):

```bash
uv run python scripts/eval_retrieval.py
```

## Run Tests

```bash
uv run pytest
```

## Note for pip users

A `requirements.txt` is also provided if you are not using uv:
`python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
The dependency lists in `pyproject.toml` and `requirements.txt` are kept in sync.

## Project Structure

See `docs/architecture.md` §3. In short: `src/` holds one package per
pipeline stage; `scripts/` holds the two entry points; `config/settings.yaml`
holds every tunable parameter; `data/` holds inputs, intermediate artifacts,
and the vector store; `docs/` holds all design documentation.

## Implementation Status

| Phase | Component | Status |
|---|---|---|
| 1 | Project setup (structure, config, utils) | ✅ Done |
| 2 | Document ingestion (PDF parsing) | ✅ Done |
| 3 | Cleaning | ✅ Done |
| 4 | Chunking | ✅ Done |
| 5 | Embedding generation | ✅ Done |
| 6 | Vector database | ✅ Done |
| 7 | Retriever | ✅ Done |
| 8 | Prompt assembly | ✅ Done |
| 9 | LLM integration | ✅ Done |
| 10 | Terminal chatbot | ✅ Done |
| 11 | Testing (golden set) | ✅ Done |
| 12 | Final documentation | ✅ Done |
| 13 | Vague-question query rewriting (rescue on model refusal) | ✅ Done |
| 14 | Hybrid BM25 retrieval (built, measured, disabled by default) | ✅ Done |
