"""
core.py — Compose retrieval, prompting, and generation into one answer.

Purpose:
    Provide answer_question(question) -> Answer, the single function that ties
    the whole RAG pipeline together and encodes the two-layer refusal policy.
    A future GUI/web front end reuses this unchanged (AC-4); the terminal is
    just one caller.

Flow (docs/system_design.md §3):
    retrieve -> if retrieval is hopeless and a rewriter is wired in: rewrite
                the question into handbook wording and retrieve again
             -> if below similarity floor: refuse (no API call)
             -> build prompt -> generate -> parse
             -> if model said NOT_COVERED and a rewriter is wired in: rewrite
                the question, retrieve better excerpts, and try once more —
                a vague question retrieves on meaning and can land next to
                the rule without ever hitting it
             -> if it still said NOT_COVERED: refuse
             -> if it answered but cited nothing: ask once for citations;
                if still uncited, answer with the retrieved sources flagged
                as unverified (an uncited answer is a formatting failure, not
                an out-of-scope question — conflating the two made the bot
                refuse questions the handbook covers)
             -> else: answer + resolved citations.

Also provides plan_courses(curriculum, taken) -> CoursePlan, the Phase 15
counterpart: same UI-agnostic contract, but it makes NO LLM call at all
(Architectural Decision AD-7 — see docs/course_planner.md §5).

Inputs:
    A question string, or a Curriculum plus the set of passed course codes; a
    ChatEngine holding the retriever, backend, refusal message, and the
    course-planning limits.

Outputs:
    Answer(text, citations, refused), or CoursePlan(plan, policy, curriculum).

Dependencies:
    src.retrieval.retriever, src.llm.backend, src.prompts.builder,
    src.curriculum.planner, src.curriculum.policy.

Why this file exists:
    Keeping orchestration in one place (separate from the terminal I/O) is
    what makes the core testable end-to-end with fakes and reusable by other
    interfaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.chat.rewriter import QueryRewriter
from src.curriculum.model import Curriculum, StudyPlan
from src.curriculum.planner import build_plan, resolve_taken
from src.curriculum.policy import PolicyRule, load_policy
from src.llm.backend import LLMBackend, LLMError
from src.prompts.builder import (
    Citation,
    build_citation_retry,
    build_prompt,
    parse_response,
)
from src.retrieval.retriever import Retriever, best_similarity
from src.retrieval.vector_store import RetrievedChunk
from src.utils.config import PlannerSettings

log = logging.getLogger(__name__)


@dataclass
class Answer:
    """A user-facing answer or refusal."""

    text: str
    citations: list[Citation]
    refused: bool
    error: str | None = None       # set when generation failed technically
    # True when the model answered but cited nothing resolvable even after a
    # corrective retry: the citations below are the retrieved sources, not the
    # model's own attribution, and the UI must say so.
    unverified: bool = False


@dataclass
class CoursePlan:
    """A study plan plus the handbook rules that shaped it.

    The planning counterpart to Answer: one object carrying everything a UI
    needs to render, including the caveats. `error` is set when the curriculum
    could not be planned at all — the same convention as Answer.error.
    """

    plan: StudyPlan
    policy: list[PolicyRule]
    curriculum: Curriculum
    # The codes actually treated as passed — the artifact's own flags plus
    # whatever the caller supplied. Carried so the UI reports the same figure
    # the planner used; deriving it separately is how a plan and its own header
    # end up disagreeing.
    taken: frozenset[str] = frozenset()
    error: str | None = None


class ChatEngine:
    """Holds the wired components and answers questions."""

    def __init__(self, retriever: Retriever, backend: LLMBackend,
                 refusal_message: str,
                 rewriter: QueryRewriter | None = None,
                 rescue_margin: float = 0.05,
                 planner: PlannerSettings = PlannerSettings()):
        """
        Args:
            rewriter: Optional. When set, a weak retrieval gets one rewrite
                attempt before the floor decides. None restores the original
                retrieve-or-refuse behavior exactly.
            rescue_margin: How far above the floor still counts as weak.
            planner: Course-planning limits. A frozen dataclass, so the default
                instance is safe to share (same pattern as RewriteSettings).
        """
        self.retriever = retriever
        self.backend = backend
        self.refusal_message = refusal_message
        self.rewriter = rewriter
        self.rescue_margin = rescue_margin
        self.planner = planner

    def answer_question(self, question: str) -> Answer:
        """Answer a single question, or refuse, per the two-layer policy."""
        question = (question or "").strip()
        if not question:
            return Answer(self.refusal_message, [], refused=True)

        results = self.retriever.retrieve(question)
        rescued = False

        # Retrieval this weak rarely produces a usable answer, so rewrite
        # before spending the answer call rather than after.
        if self.rewriter is not None and self.needs_rescue(results):
            results, rescued = self._rescue(question, results), True

        # Layer 1 — retrieval floor: refuse without spending an API call.
        if not self.retriever.meets_floor(results):
            log.debug("Below similarity floor; refusing without generation.")
            return Answer(self.refusal_message, [], refused=True)

        try:
            parsed = self._generate(question, results)

            # Layer 2 said "not covered". For a vaguely-worded question that
            # usually means the excerpts are topically close but never state
            # the rule — the question was retrieved on meaning, not on the
            # handbook's vocabulary. Rewrite it into that vocabulary and give
            # the model one better-aimed set of excerpts before refusing.
            if parsed.refused and self.rewriter is not None and not rescued:
                retried = self._rescue(question, results)
                if _chunk_ids(retried) != _chunk_ids(results):
                    reparsed = self._generate(question, retried)
                    if not reparsed.refused:
                        parsed, results = reparsed, retried
        except LLMError as exc:
            return Answer(
                "Sorry, I couldn't reach the language model to answer that "
                "just now. Please check your connection or API key and try "
                "again.",
                [], refused=False, error=str(exc),
            )

        if parsed.refused:
            return Answer(self.refusal_message, [], refused=True)

        if parsed.unverified:
            # Still uncited. The excerpts DID clear the similarity floor, so
            # the honest outcome is the answer plus the sections it was drawn
            # from, flagged — never the "not covered" message.
            log.warning("Answer remains uncited after retry; flagging.")
            return Answer(parsed.text, _retrieved_citations(results),
                          refused=False, unverified=True)

        return Answer(parsed.text, parsed.citations, refused=False)

    def plan_courses(self, curriculum: Curriculum,
                     taken: set[str] | None = None,
                     attempted: set[str] | None = None) -> CoursePlan:
        """Plan the remaining courses in a checklist. Makes NO LLM call.

        The second UI-agnostic entry point alongside answer_question, so a
        future GUI reuses this unchanged (AC-4). Everything here is local: four
        retrievals to fetch the handbook provisions that justify each constraint,
        then pure computation. Architectural Decision AD-7 keeps the model out of
        it — a schedule must be reproducible, and a confidently wrong schedule is
        worse than no schedule.

        Never raises for a data reason: build_plan reports cycles and unplannable
        courses inside the returned plan, and a policy lookup against a stale
        index degrades to "citation unavailable" rather than failing.
        """
        policy = load_policy(self.retriever, self.planner,
                             term_caps=curriculum.term_caps)
        plan = build_plan(
            curriculum, taken or set(),
            max_units=self.planner.max_units,
            min_units=self.planner.min_units,
            max_terms=self.planner.max_terms,
            pair_labs=self.planner.pair_labs,
            attempted=attempted,
        )
        log.info("Planned %d term(s); %d blocked, %d unreachable, %d cycle(s).",
                 len(plan.terms), len(plan.blocked), len(plan.unreachable),
                 len(plan.cycles))
        return CoursePlan(plan=plan, policy=policy, curriculum=curriculum,
                          taken=frozenset(resolve_taken(curriculum, taken)))

    def _generate(self, question: str, results: list[RetrievedChunk]):
        """Prompt the model with these excerpts and parse the reply.

        Includes the one corrective retry for an answer that cites nothing:
        that is a formatting failure, not an out-of-scope question, so it is
        worth asking again for the same answer with markers.
        """
        messages = build_prompt(question, results)
        raw = self.backend.generate(messages)
        parsed = parse_response(raw, results)

        if parsed.unverified:
            log.warning("Uncited reply; requesting citations once more.")
            retry_raw = self.backend.generate(build_citation_retry(messages, raw))
            parsed = parse_response(retry_raw, results)
        return parsed

    def needs_rescue(self, results: list[RetrievedChunk]) -> bool:
        """True when retrieval is too weak to be worth an answer call.

        Note this fires rarely in practice: on a single-domain corpus even an
        off-topic question scores around 0.5-0.6, well clear of the floor. The
        rescue that does most of the work is the Layer-2 one in
        answer_question — this is the cheap pre-check for the genuinely
        hopeless case.
        """
        return best_similarity(results) < (
            self.retriever.similarity_floor + self.rescue_margin)

    def _rescue(self, question: str,
                original: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Rewrite the question and retrieve again, keeping the best chunks."""
        queries = self.rewriter.rewrite(question)
        if not queries:
            return original
        return self.retrieve_merged(queries, original)

    def retrieve_merged(self, queries: list[str],
                        original: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Retrieve for each query and merge with the original results.

        The original results stay in the pool, so a rewrite that retrieves
        worse than the question itself cannot make the answer worse — the
        merged set is ranked by similarity and can only gain candidates.
        """
        merged: dict[str, RetrievedChunk] = {c.chunk_id: c for c in original}
        for query in queries:
            for chunk in self.retriever.retrieve(query):
                existing = merged.get(chunk.chunk_id)
                # Same chunk, different query: keep the better score, since
                # that is the one the floor should be judged on.
                if existing is None or chunk.similarity > existing.similarity:
                    merged[chunk.chunk_id] = chunk

        results = sorted(merged.values(), key=lambda c: c.similarity,
                         reverse=True)[:self.retriever.top_k]
        log.info("Rescue retrieval: best similarity %.3f -> %.3f",
                 best_similarity(original), best_similarity(results))
        return results


def _chunk_ids(results: list[RetrievedChunk]) -> list[str]:
    """Identity of a result set, for spotting a rescue that changed nothing."""
    return [c.chunk_id for c in results]


def _retrieved_citations(results: list[RetrievedChunk]) -> list[Citation]:
    """Citations for the excerpts the model was given, in retrieval order.

    Used only on the unverified path, where the model did not attribute its
    own claims and the best we can honestly offer is "these are the sections
    the answer was drawn from".
    """
    return [Citation(marker=i, citation=chunk.citation, chunk_id=chunk.chunk_id)
            for i, chunk in enumerate(results, start=1)]
