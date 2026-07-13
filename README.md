# RAGML — Student Handbook FAQ Chatbot

A domain-specific chatbot that answers student handbook / enrollment / policy
questions using a small pre-trained Hugging Face model (GPT-2 or DialoGPT),
adapted via prompt engineering + light fine-tuning, with a **fallback guard**
that stops the bot from answering questions outside the handbook's scope.

This repo is organized to match your 5-week plan 1:1.

## Project structure

```
ragml/
├── README.md                     ← this file (plan + how to run everything)
├── requirements.txt
├── data/
│   ├── handbook_faq_sample.json  ← original placeholder schema example (safe to delete)
│   ├── handbook_faq_real.json    ← Week 1: 24 real Q&A pairs pulled from your uploaded DLSU handbook
│   ├── handbook_full_text.txt    ← full plain-text extraction of the 339-page handbook (for pulling more Q&A pairs)
│   └── prepare_dataset.py        ← Week 1: cleans raw text/CSV into train-ready JSONL
├── model/
│   ├── model_selection_notes.md  ← Week 2: base model comparison + decision
│   ├── prompt_template.py        ← Week 3: the system prompt / prompt-engineering layer
│   ├── finetune_gpt2.py          ← Week 3: LoRA fine-tuning script
│   └── domain_guard.py           ← the "fallback if off-topic" logic
└── app/
    └── cli.py                    ← Week 4: terminal chat interface
```

## Weekly plan → what to actually do

### Week 1 — Data Collection & Environment Setup
1. **Source information extraction**: done — your uploaded DLSU handbook
   (AY 2021-2025 edition, 339 pages) was extracted to clean text at
   `data/handbook_full_text.txt` using `pdftotext -layout` (the PDF has
   embedded fonts, so extraction is clean, no OCR needed).
2. **Dataset**: started — `data/handbook_faq_real.json` has 24 real Q&A
   pairs covering Attendance, Examinations, Grading, Honors, Enrollment,
   Fees and Scholarships, Discipline, Grievance, and Student Wellbeing,
   written from the actual policy text (not placeholder content). This is
   enough to prove the pipeline works, but too small to fine-tune well --
   see "Growing the dataset" below before Week 3.
3. **Workspace Initialization**: `pip install -r requirements.txt`, confirm
   you can load a model (`python model/finetune_gpt2.py --dry-run`).

#### Growing the dataset past 24 pairs
`data/handbook_full_text.txt` has the full handbook. Sections not yet
covered but worth mining: Section 1 (General Directives), Section 2
(Student Classification beyond shifting), Section 4 (Social Norms),
the rest of Section 5 (specific discipline offenses and sanctions),
Appendix C (Student Services), Appendix F (student organizations directory),
Appendix L (attire policy), and Appendix S (Student Media). Aim for
100+ pairs total -- grep the file for section/appendix headers to
navigate it quickly.

### Week 2 — Model Selection
Compare 2–3 small models you can realistically fine-tune on free-tier
Colab/local CPU-GPU:
- `distilgpt2` — smallest, fastest, weakest generation quality
- `gpt2` (124M) — good default, well documented
- `microsoft/DialoGPT-small` — pre-tuned for conversational turns

Write your comparison in `model/model_selection_notes.md` (I've drafted a
starting table — fill in your own benchmark numbers once you test).

### Week 3 — Training of Model
- **Design System Prompts**: `model/prompt_template.py` — this is the
  prompt-engineering layer that goes in front of every generation call,
  even if you also fine-tune.
- **Fine-tuning**: `model/finetune_gpt2.py` uses LoRA (via `peft`) so it's
  light enough to run on a single GPU or Colab T4. Full fine-tuning of even
  a 124M model on CPU is painfully slow — LoRA is strongly recommended here.

### Week 4 — CLI Interface Development
`app/cli.py` ties it together: loads the fine-tuned model, applies the
system prompt, and — critically — runs every query through
`model/domain_guard.py` **before** generating a response, so off-topic
questions get a polite fallback instead of a hallucinated answer.

### Week 5 onwards — Testing & Debugging
- **Automatic**: BLEU score against held-out FAQ answers (script stub in
  `data/prepare_dataset.py` comments — happy to build this out once you have
  real fine-tuned outputs to score).
- **Human evaluation rubric** (suggested, 1–5 scale each):
  - *Relevance*: does it answer what was asked?
  - *Fluency*: is it grammatically coherent?
  - *Faithfulness*: does it avoid inventing policies not in the handbook?
  - *Fallback accuracy*: does it correctly decline off-topic questions
    instead of guessing?

## Why the fallback matters
A small fine-tuned GPT-2 will still generate *something* for any prompt,
even nonsense unrelated to your handbook — it doesn't "know" the boundaries
of its training data. `domain_guard.py` uses a lightweight similarity check
(TF-IDF or sentence-embedding cosine similarity against your known FAQ
topics) to catch out-of-scope questions **before** they reach the generator,
and returns a fixed, honest fallback message instead.
