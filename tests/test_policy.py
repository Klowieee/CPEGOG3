"""
test_policy.py — Tests for src/curriculum/policy.py (Phase 15). No model/API.

Exercises the whole grounding contract with a hand-rolled fake retriever:
  * the applied numbers come from PlannerSettings, never from the chunk text
  * all four rules come back, in POLICY_QUERIES order, one retrieval each
  * the citation is taken from the highest-similarity chunk, not results[0]
  * below the similarity floor -> no citation, but the number still applies
  * a retriever that raises -> rules still returned, uncited
  * excerpts are whitespace-collapsed and trimmed with an ellipsis

The integration test at the bottom runs the real retriever against the built
index (skipped without one) and needs NO API key — load_policy makes zero LLM
calls by design (AD-7).

Dependencies:
    pytest, src.curriculum.policy, src.utils.config, src.retrieval.vector_store.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.curriculum.policy import (                       # noqa: E402
    ELLIPSIS,
    EXCERPT_CHARS,
    POLICY_QUERIES,
    load_policy,
)
from src.retrieval.vector_store import RetrievedChunk     # noqa: E402
from src.utils.config import PlannerSettings, load_settings  # noqa: E402

CITATION = "Undergraduate, Section 10: CREDIT, GRADING AND RETENTION, prov. 10.2, p. 101"
PROVISION = ("Undergraduate › Section 10: CREDIT, GRADING AND RETENTION\n"
             "10.2 The maximum academic load is 15 units.")


def make_chunk(sim, text=PROVISION, citation=CITATION, chunk_id="d_1"):
    """A RetrievedChunk built positionally, as tests/test_retriever.py does."""
    return RetrievedChunk(chunk_id, text, "Undergraduate", "10",
                          "CREDIT, GRADING AND RETENTION", ["10.2"], [101],
                          citation, sim)


class FakeRetriever:
    """Canned retrieval, in the shape tests/test_chat_core.py uses."""

    def __init__(self, results, floor=0.35, results_by_query=None):
        self._results = results
        self._by_query = results_by_query or {}
        self.similarity_floor = floor
        self.top_k = 5
        self.calls = []

    def retrieve(self, question, k=None):
        self.calls.append(question)
        return self._by_query.get(question, self._results)


class ExplodingRetriever:
    """Stands in for a stale or half-built index: every query raises."""

    def __init__(self, floor=0.35):
        self.similarity_floor = floor
        self.top_k = 5
        self.calls = []

    def retrieve(self, question, k=None):
        self.calls.append(question)
        raise RuntimeError("Collection 'student_handbook' not found")


def _rules(retriever=None, planner=None):
    return load_policy(retriever or FakeRetriever([make_chunk(0.72)]),
                       planner or PlannerSettings())


def _by_key(rules):
    return {rule.key: rule for rule in rules}


# --- The numbers ---------------------------------------------------------------

def test_policy_rules_use_configured_numbers():
    rules = _by_key(_rules())
    assert rules["max_units"].value == 15.0
    assert rules["min_units"].value == 12.0
    # The statement must quote the number the planner will actually apply, not
    # a number lifted out of the retrieved text.
    assert "15" in rules["max_units"].statement
    assert "12" in rules["min_units"].statement


def test_non_numeric_rules_have_no_value():
    rules = _by_key(_rules())
    assert rules["lab_coreq"].value is None
    assert rules["nstp"].value is None
    # Having no number does not mean having no evidence: both are still cited.
    assert rules["lab_coreq"].citation == CITATION
    assert rules["nstp"].citation == CITATION


def test_configured_override_flows_into_the_statement():
    # A program whose checklist states its own cap (§10.2's "or the number of
    # units indicated on the program checklist") must be quoted with ITS number.
    rules = _by_key(_rules(planner=PlannerSettings(max_units=18.0)))
    assert rules["max_units"].value == 18.0
    assert "18 units" in rules["max_units"].statement


# --- Order and cost ------------------------------------------------------------

def test_all_four_rules_are_returned_in_stable_order():
    keys = [rule.key for rule in _rules()]
    assert keys == list(POLICY_QUERIES)
    assert len(keys) == 4
    # Same input, same order: a plan's constraint list must not reshuffle
    # between runs, since it is rendered into the Markdown export.
    assert [rule.key for rule in _rules()] == keys


def test_policy_queries_the_retriever_once_per_rule():
    retriever = FakeRetriever([make_chunk(0.72)])
    load_policy(retriever, PlannerSettings())
    # Four local retrievals, no more — and zero LLM calls, which is why this
    # module has no backend to fake at all (AD-7).
    assert len(retriever.calls) == len(POLICY_QUERIES)
    assert retriever.calls == list(POLICY_QUERIES.values())


# --- Choosing the evidence -----------------------------------------------------

def test_citation_comes_from_the_best_chunk():
    # Deliberately NOT sorted by similarity: with hybrid retrieval the list
    # arrives in fused-rank order, so results[0] is not necessarily the closest
    # chunk. The citation must follow the similarity, like best_similarity().
    weak = make_chunk(0.41, citation="Graduate, Section 17, p. 118", chunk_id="d_2")
    best = make_chunk(0.83, citation=CITATION, chunk_id="d_3")
    rules = _by_key(_rules(FakeRetriever([weak, best])))
    assert rules["max_units"].citation == CITATION
    assert rules["max_units"].similarity == 0.83


def test_weak_retrieval_yields_no_citation_but_still_applies_the_number():
    # WHY: the number is configuration, not something read out of the chunk, so
    # a bad retrieval cannot invalidate it — the plan is still capped at 15.
    # What IS withdrawn is the claim that the handbook was shown to say so.
    rules = _by_key(_rules(FakeRetriever([make_chunk(0.10)])))
    assert rules["max_units"].citation is None
    assert rules["max_units"].excerpt == ""
    assert rules["max_units"].value == 15.0
    # The score is still reported, so a caller can say how close it got.
    assert rules["max_units"].similarity == 0.10


def test_empty_retrieval_yields_no_citation():
    rules = _by_key(_rules(FakeRetriever([])))
    assert rules["nstp"].citation is None
    assert rules["nstp"].similarity == 0.0


def test_retrieval_failure_does_not_raise():
    # A stale or missing index must not break planning: the rules come back
    # uncited rather than as an exception out of load_policy.
    retriever = ExplodingRetriever()
    rules = load_policy(retriever, PlannerSettings())
    assert [rule.key for rule in rules] == list(POLICY_QUERIES)
    assert all(rule.citation is None and rule.similarity == 0.0
               for rule in rules)
    assert all(rule.excerpt == "" for rule in rules)
    assert _by_key(rules)["max_units"].value == 15.0
    # Every query was still attempted; one failure does not abort the loop.
    assert len(retriever.calls) == len(POLICY_QUERIES)


# --- Excerpts ------------------------------------------------------------------

def test_excerpt_is_trimmed_and_whitespace_collapsed():
    messy = ("Undergraduate › Section 10\n10.2   The maximum   academic load\n"
             "\tis 15 units, or the number of units indicated on the program "
             "checklist. " + "Overload requires the approval of the Dean. " * 6)
    excerpt = _by_key(_rules(FakeRetriever([make_chunk(0.72, text=messy)])))[
        "max_units"].excerpt

    assert "\n" not in excerpt and "\t" not in excerpt and "  " not in excerpt
    assert len(excerpt) <= EXCERPT_CHARS + len(ELLIPSIS)
    assert excerpt.endswith(ELLIPSIS)
    assert excerpt.startswith("Undergraduate › Section 10 10.2 The maximum "
                              "academic load is 15 units,")
    # The whole point of showing the handbook's own words: §10.2's caveat, which
    # the planner cannot resolve, must reach the student inside the excerpt.
    assert "or the number of units indicated on the program checklist." in excerpt


def test_short_excerpt_is_not_marked_as_trimmed():
    excerpt = _by_key(_rules())["max_units"].excerpt
    assert ELLIPSIS not in excerpt
    assert excerpt == ("Undergraduate › Section 10: CREDIT, GRADING AND RETENTION "
                       "10.2 The maximum academic load is 15 units.")


# --- Integration against the built index ---------------------------------------

def _index_ready():
    try:
        from src.retrieval.vector_store import VectorStore
        s = load_settings()
        store = VectorStore(s.paths.vector_db_dir, s.document.id)
        store.validate(s.embedding.model)
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _index_ready(),
                    reason="vector index not built (run scripts/run_ingestion.py)")
def test_unit_load_provisions_are_retrievable():
    # Verified against the live index: Undergraduate section 10 chunks carry
    # "maximum academic load"/"at least 12 academic units" and the co-requisite
    # rule. Asserted on part/keyword rather than on an exact provision list, so
    # re-chunking (which moves provision boundaries) cannot make this brittle.
    # Needs no API key: load_policy retrieves only.
    from src.retrieval.retriever import build_retriever

    settings = load_settings()
    retriever = build_retriever(settings)
    rules = _by_key(load_policy(retriever, settings.planner))

    max_rule = rules["max_units"]
    assert max_rule.citation is not None, (
        "the unit-cap provision fell below the similarity floor; "
        f"best was {max_rule.similarity:.3f}"
    )
    assert "Undergraduate" in max_rule.citation
    assert max_rule.excerpt and max_rule.value == settings.planner.max_units

    lab = retriever.retrieve(POLICY_QUERIES["lab_coreq"], k=3)
    assert any("co-requisite" in chunk.text.lower() for chunk in lab), \
        "no top-3 chunk mentions the co-requisite rule"
