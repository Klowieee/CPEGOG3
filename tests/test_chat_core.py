"""
test_chat_core.py — Tests for src/chat/core.py (Phase 10). No model/API.

Exercises the full orchestration and both refusal layers with fakes:
  * below-floor retrieval -> refuse, NO backend call (layer 1)
  * good retrieval + grounded reply -> answer with mapped citations
  * good retrieval + model NOT_COVERED -> refuse (layer 2)
  * backend failure -> graceful error answer
  * empty question -> refuse
  * layer-2 refusal + a rewriter -> rewrite, retrieve again, answer

Phase 15 adds the planning seam, whose headline property is what it does NOT
do: plan_courses must never touch the backend (AD-7).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat.core import ChatEngine                        # noqa: E402
from src.curriculum.model import (                          # noqa: E402
    Course,
    Curriculum,
    PrereqConfidence,
    PrereqSource,
)
from src.curriculum.policy import POLICY_QUERIES            # noqa: E402
from src.llm.backend import LLMBackend, LLMError            # noqa: E402
from src.prompts.builder import REFUSAL_SENTINEL            # noqa: E402
from src.retrieval.vector_store import RetrievedChunk       # noqa: E402
from src.utils.config import PlannerSettings                # noqa: E402

REFUSAL = "Not covered by the handbook."


def make_chunk(sim, chunk_id="d_1"):
    return RetrievedChunk(chunk_id, "Undergraduate › Section 10\nGrading rules.",
                          "Undergraduate", "10", "GRADING", ["10.1"], [101],
                          "Undergraduate, Section 10, p. 101", sim)


class FakeRetriever:
    def __init__(self, results, floor=0.35, results_by_question=None):
        self._results = results
        # Lets a rewritten query retrieve something different from the
        # original question, which is the whole point of the rescue path.
        self._by_question = results_by_question or {}
        self.similarity_floor = floor
        self.top_k = 5
        self.queries = []

    def retrieve(self, question, k=None):
        self.queries.append(question)
        return self._by_question.get(question, self._results)

    def meets_floor(self, results):
        return bool(results) and max(
            (r.similarity for r in results), default=0.0) >= self.similarity_floor


class FakeRewriter:
    """Stands in for QueryRewriter; records whether it was consulted."""

    def __init__(self, queries):
        self._queries = queries
        self.calls = 0

    def rewrite(self, question):
        self.calls += 1
        return self._queries


class FakeBackend(LLMBackend):
    def __init__(self, reply=None, raise_error=False, replies=None):
        self.reply = reply
        self.replies = list(replies) if replies else None   # one per call
        self.raise_error = raise_error
        self.called = False
        self.calls = 0
        self.last_messages = None

    def generate(self, messages):
        self.called = True
        self.calls += 1
        self.last_messages = messages
        if self.raise_error:
            raise LLMError("boom")
        if self.replies:
            return self.replies[min(self.calls - 1, len(self.replies) - 1)]
        return self.reply


def test_below_floor_refuses_without_calling_backend():
    backend = FakeBackend(reply="should not be used")
    engine = ChatEngine(FakeRetriever([make_chunk(0.10)]), backend, REFUSAL)
    ans = engine.answer_question("something obscure")
    assert ans.refused and ans.text == REFUSAL
    assert backend.called is False           # layer-1 saved the API call


def test_grounded_answer_returns_citations():
    backend = FakeBackend(reply="Grades use a scale [1].")
    engine = ChatEngine(FakeRetriever([make_chunk(0.7)]), backend, REFUSAL)
    ans = engine.answer_question("How does grading work?")
    assert not ans.refused
    assert ans.citations and ans.citations[0].citation.startswith("Undergraduate")


def test_model_not_covered_refuses():
    backend = FakeBackend(reply=REFUSAL_SENTINEL)
    engine = ChatEngine(FakeRetriever([make_chunk(0.7)]), backend, REFUSAL)
    ans = engine.answer_question("Something the excerpts don't answer")
    assert ans.refused and ans.text == REFUSAL


def test_backend_failure_is_graceful():
    backend = FakeBackend(raise_error=True)
    engine = ChatEngine(FakeRetriever([make_chunk(0.7)]), backend, REFUSAL)
    ans = engine.answer_question("How does grading work?")
    assert not ans.refused           # not a content refusal
    assert ans.error == "boom" and "couldn't reach" in ans.text


def test_empty_question_refuses():
    backend = FakeBackend(reply="unused")
    engine = ChatEngine(FakeRetriever([make_chunk(0.7)]), backend, REFUSAL)
    ans = engine.answer_question("   ")
    assert ans.refused and backend.called is False


def test_uncited_reply_is_retried_and_then_accepted():
    # First reply forgets the markers; the corrective retry supplies them.
    backend = FakeBackend(replies=["Grades use a 4.0 scale.",
                                   "Grades use a 4.0 scale [1]."])
    engine = ChatEngine(FakeRetriever([make_chunk(0.7)]), backend, REFUSAL)
    ans = engine.answer_question("How does grading work?")
    assert backend.calls == 2
    assert not ans.refused and not ans.unverified
    assert [c.marker for c in ans.citations] == [1]
    # The retry carries the first reply plus a corrective user turn.
    assert backend.last_messages[-2]["role"] == "assistant"
    assert backend.last_messages[-1]["role"] == "user"


def test_still_uncited_answer_is_shown_flagged_not_refused():
    # Regression for the "always says not covered" bug: retrieval cleared the
    # floor, so an uncited answer must NOT become the out-of-scope message.
    backend = FakeBackend(reply="Grades use a 4.0 scale.")
    engine = ChatEngine(FakeRetriever([make_chunk(0.7)]), backend, REFUSAL)
    ans = engine.answer_question("How does grading work?")
    assert backend.calls == 2                    # one retry, then accept
    assert not ans.refused
    assert ans.unverified
    assert ans.text != REFUSAL
    assert [c.citation for c in ans.citations] == \
        ["Undergraduate, Section 10, p. 101"]    # the retrieved sources


def test_grouped_markers_are_accepted():
    backend = FakeBackend(reply="Grades use a 4.0 scale [1, 1].")
    engine = ChatEngine(FakeRetriever([make_chunk(0.7)]), backend, REFUSAL)
    ans = engine.answer_question("How does grading work?")
    assert backend.calls == 1 and not ans.refused and not ans.unverified


# --- Vague-question rescue -------------------------------------------------

def test_answered_question_never_consults_the_rewriter():
    """The common path must cost nothing extra."""
    rewriter = FakeRewriter(["handbook wording"])
    engine = ChatEngine(FakeRetriever([make_chunk(0.7)]),
                        FakeBackend(reply="Grades use a scale [1]."),
                        REFUSAL, rewriter=rewriter)
    ans = engine.answer_question("How does grading work?")
    assert not ans.refused
    assert rewriter.calls == 0


def test_layer2_refusal_triggers_rewrite_and_answers():
    # Retrieval clears the floor but the excerpts miss; the rewritten query
    # retrieves a different chunk and the second attempt answers.
    better = make_chunk(0.8, chunk_id="d_2")
    retriever = FakeRetriever(
        [make_chunk(0.6)],
        results_by_question={"academic dishonesty sanction": [better]},
    )
    rewriter = FakeRewriter(["academic dishonesty sanction"])
    backend = FakeBackend(replies=[REFUSAL_SENTINEL, "Expulsion is possible [1]."])
    engine = ChatEngine(retriever, backend, REFUSAL, rewriter=rewriter)

    ans = engine.answer_question("what happens if i copy homework")
    assert not ans.refused
    assert ans.text == "Expulsion is possible [1]."
    assert rewriter.calls == 1
    assert backend.calls == 2
    assert "academic dishonesty sanction" in retriever.queries


def test_layer2_refusal_still_refuses_when_rewrite_yields_nothing():
    rewriter = FakeRewriter([])          # simulates a failed rewrite call
    backend = FakeBackend(reply=REFUSAL_SENTINEL)
    engine = ChatEngine(FakeRetriever([make_chunk(0.6)]), backend, REFUSAL,
                        rewriter=rewriter)
    ans = engine.answer_question("what is the wifi password")
    assert ans.refused and ans.text == REFUSAL
    assert rewriter.calls == 1
    assert backend.calls == 1            # no second answer call wasted


def test_rescue_that_retrieves_nothing_new_skips_the_second_call():
    # The rewrite returned queries, but they retrieve the same chunk, so
    # re-prompting the model with identical excerpts would just burn a call.
    rewriter = FakeRewriter(["same thing again"])
    backend = FakeBackend(reply=REFUSAL_SENTINEL)
    engine = ChatEngine(FakeRetriever([make_chunk(0.6)]), backend, REFUSAL,
                        rewriter=rewriter)
    ans = engine.answer_question("something not in the handbook")
    assert ans.refused
    assert backend.calls == 1


def test_second_attempt_refusal_is_still_a_refusal():
    retriever = FakeRetriever(
        [make_chunk(0.6)],
        results_by_question={"formal wording": [make_chunk(0.8, chunk_id="d_2")]},
    )
    backend = FakeBackend(reply=REFUSAL_SENTINEL)   # refuses both times
    engine = ChatEngine(retriever, backend, REFUSAL,
                        rewriter=FakeRewriter(["formal wording"]))
    ans = engine.answer_question("who is the best professor")
    assert ans.refused and ans.text == REFUSAL
    assert backend.calls == 2


def test_below_floor_rescue_happens_before_generating():
    # Hopeless retrieval: rewrite first, and if that recovers a strong chunk
    # the answer call still happens.
    strong = make_chunk(0.8, chunk_id="d_2")
    retriever = FakeRetriever(
        [make_chunk(0.10)],
        results_by_question={"formal wording": [strong]},
    )
    backend = FakeBackend(reply="Answered [1].")
    engine = ChatEngine(retriever, backend, REFUSAL,
                        rewriter=FakeRewriter(["formal wording"]))
    ans = engine.answer_question("vague thing")
    assert not ans.refused
    assert backend.calls == 1


def test_below_floor_without_rescue_still_refuses_without_backend():
    backend = FakeBackend(reply="should not be used")
    engine = ChatEngine(FakeRetriever([make_chunk(0.10)]), backend, REFUSAL,
                        rewriter=FakeRewriter([]))
    ans = engine.answer_question("something obscure")
    assert ans.refused
    assert backend.called is False


# --- Course planning (Phase 15) --------------------------------------------

def _curriculum():
    passed = Course("GEMATMW", "Mathematics in the Modern World", 3, 1, 1,
                    (), (), PrereqConfidence.STATED, True, "3.5")
    next_up = Course("CSMATH2", "Discrete Structures", 3, 1, 2,
                     ("GEMATMW",), (), PrereqConfidence.STATED)
    return Curriculum("bscs-st", "BS Computer Science", 3,
                      {c.code: c for c in (passed, next_up)},
                      PrereqSource.COLUMN)


def test_plan_courses_makes_no_backend_call():
    """Architectural Decision AD-7, in its enforceable form.

    Course ordering is deterministic code; the LLM is not in the loop. A plan
    must be reproducible and must cost nothing, so this sits deliberately
    beside test_below_floor_refuses_without_calling_backend above.
    """
    backend = FakeBackend(reply="should not be used")
    engine = ChatEngine(FakeRetriever([make_chunk(0.8)]), backend, REFUSAL)

    engine.plan_courses(_curriculum())

    assert backend.called is False


def test_plan_courses_returns_plan_and_policy():
    retriever = FakeRetriever([make_chunk(0.8)])
    engine = ChatEngine(retriever, FakeBackend(), REFUSAL)

    result = engine.plan_courses(_curriculum())

    assert [c.code for c in result.plan.terms[0].courses] == ["CSMATH2"]
    assert result.error is None
    # One retrieval per policy rule, and every rule carries the number applied.
    assert len(result.policy) == len(POLICY_QUERIES)
    assert any(r.key == "max_units" and r.value == 15.0 for r in result.policy)


def test_plan_courses_honors_the_configured_limits():
    """The engine must pass PlannerSettings through, not hard-code 15."""
    courses = [Course(f"AAA{i:03d}", f"Course {i}", 3) for i in range(4)]
    curriculum = Curriculum("x", "X", 3, {c.code: c for c in courses},
                            PrereqSource.COLUMN)
    engine = ChatEngine(FakeRetriever([make_chunk(0.8)]), FakeBackend(), REFUSAL,
                        planner=PlannerSettings(max_units=6.0, min_units=0.0))

    result = engine.plan_courses(curriculum)

    assert result.plan.terms[0].units == 6.0


def test_plan_courses_accepts_extra_taken_codes():
    engine = ChatEngine(FakeRetriever([make_chunk(0.8)]), FakeBackend(), REFUSAL)

    result = engine.plan_courses(_curriculum(), {"CSMATH2"})

    assert result.plan.terms == []


def test_plan_courses_on_empty_curriculum_returns_an_empty_plan():
    empty = Curriculum("x", "X", 3, {}, PrereqSource.NONE)
    engine = ChatEngine(FakeRetriever([make_chunk(0.8)]), FakeBackend(), REFUSAL)

    result = engine.plan_courses(empty)

    assert result.plan.terms == []
    assert result.error is None
