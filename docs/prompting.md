# Prompt Engineering Strategy
## DLSU Student Handbook RAG Chatbot

**Prepared by:** Prompt Engineering Specialist — v1.0, 20 July 2026

---

## 1. Design Goals

1. Answers grounded exclusively in retrieved handbook text (FR-2).
2. Verifiable citations mapped back to structured metadata (FR-3).
3. Reliable refusal when the context does not answer the question (FR-4) — second layer behind the retrieval similarity floor.
4. Plain student-friendly wording, quoting the handbook where wording is precise/disciplinary (FR-5).
5. Low variance: temperature 0.1, tightly specified output shape.

## 2. System Prompt (v1 draft — to be refined during testing)

```
You are the DLSU Student Handbook Assistant. You answer questions from
students using ONLY the handbook excerpts provided in each message.

Rules:
1. Use only the numbered excerpts provided. Do not use outside knowledge
   about DLSU or universities in general.
2. If the excerpts do not contain the answer, reply exactly:
   NOT_COVERED
   and nothing else.
3. Cite every claim with the excerpt number(s) in square brackets, e.g. [1]
   or [1][3], placed at the end of the sentence the claim appears in.
4. Explain rules in plain, clear English a student can act on. When a
   provision defines an offense, penalty, requirement, or deadline, quote
   the handbook's exact wording for that part in quotation marks.
5. Be concise. Answer the question asked; do not summarize unrelated
   excerpt content.
6. Do not give advice beyond what the handbook states. If a student asks
   what they should do, describe what the handbook provides for their
   situation.
```

Design notes:
- **`NOT_COVERED` sentinel:** the application detects this token and renders the user-facing polite refusal itself ("That doesn't seem to be covered in the Student Handbook (AY 2021–2025). You may want to contact the relevant DLSU office directly."). Keeping the refusal text in application code — not model output — makes the behavior deterministic and testable.
- **Excerpt-number citations:** the model cites `[2]`; the application maps `[2]` to the chunk's structured metadata and renders the real citation. The model never generates section numbers itself — this eliminates the most common citation-hallucination mode.
- **Quote-vs-plain rule (rule 4):** operationalizes FR-5 with a concrete trigger list (offense/penalty/requirement/deadline) instead of asking the model to judge "legal significance."

## 3. User Message Template

```
HANDBOOK EXCERPTS:

[1] {part} › Section {n}: {title} (prov. {provision}, p. {page})
{chunk text}

[2] ...

QUESTION: {user question}
```

Excerpts appear in retrieval-score order. The citation header doubles as context for the model (it can see which part a rule belongs to, e.g., Undergraduate vs Graduate — important given duplicated section numbers).

## 4. Refusal: Two-Layer Design (recap)

| Layer | Trigger | Behavior |
|---|---|---|
| Retrieval | best similarity < floor (init 0.35) | Refuse locally; **no API call** (cost-free, hallucination-proof) |
| Generation | excerpts retrieved but unresponsive | Model outputs `NOT_COVERED` → app renders refusal |

Both paths produce the identical user-facing message, so testing can assert on one string.

## 5. Post-Processing Validation

After generation the application checks:
- Response is `NOT_COVERED` → refusal path.
- Otherwise, response must contain ≥1 citation marker `[n]` with n in the provided range; markers out of range are stripped and logged; a response with *zero* valid markers is treated as a grounding failure → refusal path with a logged warning (fail-closed, per "accuracy over speed").

## 6. Conversation Memory & Follow-ups

Out of scope for v1 (FR-6): each question is embedded and answered independently, and the terminal makes no claim otherwise. The prompt templates take a message list, so adding history later is a prompt-assembly change only.

## 7. Prompt-Injection Consideration

Handbook text is trusted (we ingest it ourselves), but user questions are not. The system prompt's rule 1 plus the fixed template (question always last, clearly labeled) is adequate for a v1 student project; testing.md includes adversarial questions ("ignore your instructions and…") to verify the model stays in role. This limitation is documented honestly in limitations.md rather than over-engineered now.
