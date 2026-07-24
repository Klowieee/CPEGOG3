"""
rewriter.py — Turn a vague question into handbook wording, on demand.

Purpose:
    Rescue questions that retrieval cannot match. A student asks "what
    happens if I copy someone's homework"; the handbook says "academic
    dishonesty" and "major offense". Dense retrieval keys on wording, so the
    student's phrasing scores below the similarity floor and the system
    refuses a question the handbook plainly answers. This module spends one
    small LLM call rewriting the question into the handbook's vocabulary so
    retrieval gets a second, better-aimed attempt.

Inputs:
    The user's question; an LLMBackend (typically a second APIBackend with a
    small token budget).

Outputs:
    Up to max_queries handbook-style search queries. Never an exception.

Dependencies:
    src.llm.backend.

Why this file exists:
    It lives in the chat package, not retrieval, on purpose: retrieval stays
    LLM-free so scripts/eval_retrieval.py can measure it offline and for
    free. Rewriting is an orchestration concern, and ChatEngine treats it as
    optional — with no rewriter wired in, behavior is exactly as before.
"""

from __future__ import annotations

import logging
import re

from src.llm.backend import LLMBackend, LLMError

log = logging.getLogger(__name__)

# Topic list matters more than it looks: it is what steers an 8b model from
# generic paraphrase ("what are the consequences of copying homework") toward
# the handbook's actual register ("academic dishonesty major offense sanction").
REWRITE_SYSTEM = (
    "You turn a student's casual question into search queries for the DLSU "
    "Student Handbook.\n\n"
    "The handbook uses formal language about topics such as: major and minor "
    "offenses, discipline and sanctions, academic dishonesty (cheating, "
    "plagiarism), grading and credit, attendance and absences, enrollment and "
    "registration, tuition, scholarships and financial aid, honors and awards, "
    "graduation requirements, grievance and complaint procedures, student "
    "organizations, and student services.\n\n"
    "Rewrite the question as 1 to 3 alternative search queries that use the "
    "handbook's formal wording. Each query should be a short phrase, not a "
    "question. Output one query per line. No numbering, no bullets, no "
    "quotation marks, no explanation."
)

# Leading list markers an instruction-following model adds despite being told
# not to: "1. ", "2) ", "- ", "* ", "• ".
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def build_rewrite_prompt(question: str) -> list[dict]:
    """Chat messages for one rewrite call."""
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"Question: {question}"},
    ]


def parse_rewrites(raw: str, original: str, max_queries: int = 3) -> list[str]:
    """Extract clean search queries from the model's line-per-query reply.

    Plain lines rather than JSON because small models mangle JSON often
    enough that the parser would become the unreliable part. Anything that
    does not look like a query — blanks, an echo of the original question,
    a duplicate — is dropped rather than fixed.
    """
    queries: list[str] = []
    seen = {original.strip().lower()}

    for line in (raw or "").splitlines():
        cleaned = _BULLET_RE.sub("", line).strip().strip('"\'')
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(cleaned)
        if len(queries) == max_queries:
            break

    return queries


class QueryRewriter:
    """Rewrites a question into handbook phrasing via one LLM call."""

    def __init__(self, backend: LLMBackend, max_queries: int = 3):
        self.backend = backend
        self.max_queries = max_queries

    def rewrite(self, question: str) -> list[str]:
        """Return handbook-style queries, or [] if the rewrite did not work.

        This never raises. The rewrite is a best-effort rescue on a path that
        was already heading for a refusal, so a failed rewrite must degrade
        to the original behavior rather than turn a refusal into a crash.
        """
        try:
            raw = self.backend.generate(build_rewrite_prompt(question))
        except LLMError as exc:
            log.warning("Query rewrite failed (%s); keeping original results.", exc)
            return []

        queries = parse_rewrites(raw, question, self.max_queries)
        if queries:
            log.info("Rewrote %r as %s", question, queries)
        else:
            log.warning("Rewrite returned nothing usable for %r", question)
        return queries
