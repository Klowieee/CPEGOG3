"""
builder.py — Build grounded prompts and parse the model's cited answers.

Purpose:
    Two tightly-coupled jobs (docs/prompting.md): (1) assemble a chat prompt
    that presents retrieved chunks as numbered, citation-headed excerpts and
    instructs the model to answer ONLY from them; (2) parse the model's reply
    — detect the NOT_COVERED refusal sentinel, and map the model's excerpt
    numbers [1][2] back to the real structured citations of the chunks it
    used. The model never writes section numbers itself, which removes the
    main citation-hallucination path.

Inputs:
    A question and the retrieved chunks (build_prompt); the model's raw reply
    and the same chunks (parse_response).

Outputs:
    build_prompt -> list[{"role","content"}] messages.
    build_citation_retry -> the same messages plus a corrective turn, used when
        the first reply cited nothing resolvable.
    parse_response -> ParsedAnswer(text, citations, refused, unverified).

Dependencies:
    src.retrieval.vector_store (RetrievedChunk); re/dataclasses (stdlib).

Why this file exists:
    Prompting and response-parsing are one contract: the template defines the
    [n] markers, so the parser that resolves them lives beside it. Keeping the
    refusal sentinel and citation-mapping here makes them unit-testable
    without the LLM.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.retrieval.vector_store import RetrievedChunk

log = logging.getLogger(__name__)

REFUSAL_SENTINEL = "NOT_COVERED"

SYSTEM_PROMPT = (
    "You are the DLSU Student Handbook Assistant. You answer questions from "
    "students using ONLY the handbook excerpts provided in each message.\n\n"
    "Rules:\n"
    "1. Use only the numbered excerpts provided. Do not use outside knowledge "
    "about DLSU or universities in general.\n"
    "2. Base your answer only on the excerpts. If the answer can be found in "
    "or reasonably inferred from them, answer it. If the excerpts answer the "
    "question even partially, answer with what they do contain. Only if the "
    "excerpts are truly unrelated to the question, reply with exactly "
    f"{REFUSAL_SENTINEL} and nothing else.\n"
    "3. Cite every claim with the excerpt number(s) in square brackets, at the "
    "end of the sentence the claim appears in. Write each number in its own "
    "brackets: [1] or [1][3]. Never write [1, 3], and never write section or "
    "page numbers inside the brackets.\n"
    "4. Explain rules in plain, clear English a student can act on. When a "
    "provision defines an offense, penalty, requirement, or deadline, quote "
    "the handbook's exact wording for that part in quotation marks.\n"
    "5. Be concise. Answer the question asked; do not summarize unrelated "
    "excerpt content.\n"
    "6. Do not give advice beyond what the handbook states."
)

# Matches citation markers like [1], [12], and grouped forms like [1, 3] or
# [2; 4]. The grouped form is NOT what the system prompt asks for, but models
# emit it constantly — matching only "[1]" made every grouped citation
# unresolvable, which the parser then read as "the model cited nothing" and
# turned into a false out-of-scope refusal.
_MARKER_RE = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")

# Wrapping characters a model may put around a bare sentinel ("**NOT_COVERED**").
_SENTINEL_TRIM = "*`_\"' .\n\t"


@dataclass
class Citation:
    """A resolved citation the UI can render, tied to an excerpt the model used."""

    marker: int                  # the [n] the model wrote
    citation: str                # e.g. "Undergraduate, Section 10: ..., p. 101"
    chunk_id: str


@dataclass
class ParsedAnswer:
    """Outcome of parsing a model reply."""

    text: str
    citations: list[Citation]
    refused: bool
    unverified: bool = False     # answered, but no marker could be resolved


def _split_group(group: str) -> list[int]:
    """Turn a marker group's inner text ("1" or "1, 3") into numbers."""
    return [int(part) for part in re.split(r"[,;]", group) if part.strip()]


def _extract_markers(text: str) -> list[int]:
    """All excerpt numbers cited in the text, in order of appearance."""
    numbers: list[int] = []
    for match in _MARKER_RE.finditer(text):
        numbers.extend(_split_group(match.group(1)))
    return numbers


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """Assemble the system+user messages for the generation call.

    Each chunk becomes a numbered excerpt headed by its citation string, so
    the model can see which part/section it belongs to (crucial where section
    titles repeat across Undergraduate/Graduate). Excerpts are presented in
    retrieval-rank order.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        # chunk.text already begins with a breadcrumb; prepend the citation
        # so the model can reference part/section explicitly.
        blocks.append(f"[{i}] {chunk.citation}\n{chunk.text}")
    excerpts = "\n\n".join(blocks)

    user_content = (
        f"HANDBOOK EXCERPTS:\n\n{excerpts}\n\n"
        f"QUESTION: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


CITATION_RETRY_INSTRUCTION = (
    "Your answer did not cite any excerpt. Rewrite the same answer, adding the "
    "excerpt number in square brackets at the end of every sentence that makes "
    "a claim — each number in its own brackets, e.g. [1] or [1][3]. Change "
    "nothing else. If the excerpts genuinely do not answer the question, reply "
    f"with exactly {REFUSAL_SENTINEL} instead."
)


def build_citation_retry(messages: list[dict], reply: str) -> list[dict]:
    """Extend a prompt with the model's uncited reply and a corrective turn.

    Used once by src.chat.core when parse_response reports unverified=True:
    asking the model to re-cite is cheaper and far more accurate than guessing
    which excerpt each sentence came from.
    """
    return [
        *messages,
        {"role": "assistant", "content": reply},
        {"role": "user", "content": CITATION_RETRY_INSTRUCTION},
    ]


def parse_response(raw: str, chunks: list[RetrievedChunk]) -> ParsedAnswer:
    """Interpret the model's reply against the chunks it was given.

    Detects the NOT_COVERED sentinel; otherwise maps each [n] marker to the
    corresponding chunk's real citation. Out-of-range markers are stripped
    from the text and ignored (a mild model error, not a failure).

    A reply that has real text but no resolvable marker is NOT a refusal — it
    is returned with unverified=True so the caller can retry or present it
    with a caveat. Collapsing that case into a refusal is what made the app
    answer "not covered in the handbook" for questions the handbook covers.

    Args:
        raw: The model's raw text reply.
        chunks: The excerpts passed to the model, in [1..n] order.

    Returns:
        ParsedAnswer.
    """
    text = (raw or "").strip()

    if not text:
        # Nothing to show. The backend normally raises before we get here.
        return ParsedAnswer(text="", citations=[], refused=True)

    markers = _extract_markers(text)
    valid = [m for m in markers if 1 <= m <= len(chunks)]

    # A bare sentinel is a refusal. So is a reply that opens with the sentinel
    # and cites nothing; but "NOT_COVERED does not apply here ... [1]" is a
    # real, cited answer and must not be swallowed by a startswith() check.
    bare = text.strip(_SENTINEL_TRIM).upper()
    if bare == REFUSAL_SENTINEL or (text.upper().startswith(REFUSAL_SENTINEL)
                                    and not valid):
        return ParsedAnswer(text="", citations=[], refused=True)

    if not valid:
        # Answered, but grounding cannot be verified. Hand it back flagged;
        # src.chat.core decides whether to retry or show it with a caveat.
        log.warning("Model reply resolved no citation marker: %.200r", text)
        return ParsedAnswer(text=text, citations=[], refused=False,
                            unverified=True)

    # Rewrite each marker group, dropping out-of-range numbers and normalizing
    # "[1, 3]" to "[1][3]" so the visible text matches the rendered sources.
    def _strip(match: re.Match) -> str:
        kept = [n for n in _split_group(match.group(1)) if 1 <= n <= len(chunks)]
        return "".join(f"[{n}]" for n in kept)

    cleaned = _MARKER_RE.sub(_strip, text).strip()

    seen: set[int] = set()
    citations: list[Citation] = []
    for m in valid:
        if m in seen:
            continue
        seen.add(m)
        chunk = chunks[m - 1]
        citations.append(Citation(marker=m, citation=chunk.citation,
                                   chunk_id=chunk.chunk_id))

    return ParsedAnswer(text=cleaned, citations=citations, refused=False)
