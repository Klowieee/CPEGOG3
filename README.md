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

> **Just want to run it?** [`RUNNING.md`](RUNNING.md) is a complete
> step-by-step guide for both features, from a fresh clone, with troubleshooting.

## Setup from a fresh clone

The repository holds **code only**. Every PDF, the vector index, and the
extracted curriculum are gitignored — the handbook is copyrighted and a
curriculum file contains a student's grades — so a clone needs the steps below
before it will answer anything.

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

**1. Install dependencies.** `uv` reads `pyproject.toml` and creates the venv:

```bash
git clone https://github.com/Klowieee/CPEGOG3.git
cd CPEGOG3
uv sync                  # creates .venv and installs everything (incl. pytest)
```

> The first sync downloads PyTorch (via sentence-transformers, ~2 GB) — run it
> on good wifi. `uv` caches packages globally, so later syncs are fast.

**2. Add the handbook PDF** at `data/handbooks/student-handbook.pdf`.

**3. Build the vector index** — one time, a few minutes; it also downloads the
embedding model (~130 MB) on first run:

```bash
uv run python scripts/run_ingestion.py
```

You should end with roughly 369 chunks stored. Everything so far is local.

**4. Set your API key** for answer generation (free key from
[console.groq.com](https://console.groq.com)):

```bash
export GROQ_API_KEY="..."        # Windows (PowerShell): $env:GROQ_API_KEY="..."
```

The key is read from the environment only — never write it into any file in
this repository. See `.env.example`.

**5. For the course planner only**, add your program checklist PDF to
`data/checklists/` and extract it once:

```bash
uv run python scripts/inspect_checklist.py data/checklists/YOUR-CHECKLIST.pdf
```

Steps 2-4 are for handbook questions; steps 2-3 are skippable if you only want
`/plan`, which needs neither the handbook index nor an API key.

### Checking it worked

```bash
uv run pytest                            # ~450 tests, no key or network needed
uv run python scripts/eval_retrieval.py  # retrieval quality, no LLM involved
```

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

## Course Planner (Phase 15)

Work out what to enrol in next, and in what order, from your program checklist.
Ordering is computed from the prerequisite graph in plain code — **no LLM call
and no API tokens are spent** — while the unit limits it applies are shown with
real handbook citations retrieved from the local index.

Put your checklist PDF in `data/checklists/`, then inspect what the parser made
of it before trusting any plan:

```bash
uv run python scripts/inspect_checklist.py data/checklists/your-checklist.pdf
```

Its first three lines tell you what it managed to read:

```
PREREQUISITE SOURCE: column   (45 of 103 course(s) have stated prerequisites)
YEAR/TERM GROUPING:  found (103 of 103 placed)
ALREADY TAKEN:       0 course(s) (0 units); 103 remaining (209 units)
```

It writes `data/checklists/<program>.curriculum.yaml` — a hand-editable file
listing every course, its units, its prerequisites and corequisites, and whether
you have passed it. **The planner reads that file, never the PDF**
(`docs/architecture.md` AD-8), so anything the parser got wrong you fix once, by
hand, in one place. Re-running the script will not overwrite your corrections
(pass `--force` if you want it to).

> Most DLSU checklists have no grade column — there is nowhere on the sheet to
> record what you passed. `/plan` asks you to type those courses, or you can set
> `taken: true` on them in the curriculum file.

Then, inside the chatbot, type `/plan`. It finds the checklist itself, prints the
whole program a term at a time, and asks what you have finished:

```
Which terms have you completed? (1-12; e.g. "1-6" or "1-5,7")
Terms completed: 1-5
  ✓ 45 course(s) marked from term(s) 1-5

Anything else you've passed? (codes, or Enter to skip)
Anything in there you HAVEN'T passed? (e.g. a failed course — codes, or Enter) LOGDSGN
  ↻ LOGDSGN will be planned again, and anything needing them stays blocked
```

Whole terms first because that is how progress is actually described; the last
question is what makes *"I finished terms 1-5 but failed one of them"*
expressible. A removed course goes back into the plan as a retake, and anything
depending on it waits.

You then get a term-by-term plan through to graduation — each term capped at the
load **your checklist** prescribes — plus a printable HTML page at
`data/plans/<program>-plan.html`. Open it in any browser.

> `data/checklists/` and `data/plans/` are gitignored: the curriculum file
> contains your grades.

Design rationale, and an honest account of what the planner cannot do, are in
[`docs/course_planner.md`](docs/course_planner.md).

## Run Tests

```bash
uv run pytest
```

## Note for pip users

A `requirements.txt` is also provided if you are not using uv:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dependency lists in `pyproject.toml` and `requirements.txt` are kept in
sync. Note that `rank-bm25` is not optional despite hybrid retrieval shipping
disabled: `src/retrieval/retriever.py` imports it at module level, so the app
will not start without it.

## Project Structure

See `docs/architecture.md` §3. In short: `src/` holds one package per
pipeline stage; `scripts/` holds the entry points and inspection tools;
`config/settings.yaml` holds every tunable parameter; `data/` holds inputs,
intermediate artifacts, and the vector store; `docs/` holds all design
documentation.

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
| 15 | Course planner (checklist → prerequisite flowchart) | ✅ Done |
