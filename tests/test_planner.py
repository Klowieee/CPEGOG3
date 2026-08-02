"""
test_planner.py — Tests for src/curriculum/planner.py (Phase 15).

The planner is pure computation, so unlike generation it can be pinned down
exactly. These tests assert the invariants docs/testing.md §2.1 promises, and
the ones that would actually hurt a student if broken:

* the unit cap is never exceeded, and a corequisite pair is never split;
* a plan is identical no matter what order the courses arrive in;
* a malformed graph (cycle, over-cap bundle, 100-course chain) is reported and
  survived rather than raising or hanging;
* nothing is invented when the checklist states no prerequisites.

Synthetic fixtures throughout — no PDF, no index, no network, no model.

Dependencies:
    pytest, src.curriculum.model, src.curriculum.planner.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.curriculum.model import (            # noqa: E402
    Course,
    Curriculum,
    PrereqConfidence,
    PrereqSource,
)
from src.curriculum.planner import (          # noqa: E402
    break_cycles,
    build_plan,
    coreq_bundles,
    downstream_counts,
    find_cycles,
    levels,
)

LIMITS = {"max_units": 15.0, "min_units": 12.0, "max_terms": 8}


def C(code: str, units: float = 3, *, prereqs=(), coreqs=(), year=None, term=None,
      taken=False, placeholder=False, title=None, grade=None,
      confidence=PrereqConfidence.STATED) -> Course:
    """Compact Course builder; only `code` is usually interesting."""
    return Course(code=code, title=title or f"Course {code}", units=units,
                  year=year, term=term, prereqs=tuple(prereqs),
                  coreqs=tuple(coreqs), confidence=confidence, taken=taken,
                  grade=grade, placeholder=placeholder)


def cur(*courses: Course, source=PrereqSource.COLUMN, terms_per_year: int = 3,
        max_units=None) -> Curriculum:
    return Curriculum(
        program_id="test", program_name="Test Program",
        terms_per_year=terms_per_year,
        courses={c.code: c for c in courses},
        prereq_source=source, max_units_override=max_units)


def plan(*courses: Course, taken=None, source=PrereqSource.COLUMN, **overrides):
    limits = {**LIMITS, **overrides}
    max_units = limits.pop("max_units_override", None)
    return build_plan(cur(*courses, source=source, max_units=max_units),
                      taken or set(), **limits)


def codes_by_term(result) -> list[list[str]]:
    return [[c.code for c in term.courses] for term in result.terms]


# --- Graph primitives ----------------------------------------------------------

def test_levels_counts_prerequisite_depth():
    assert levels({"A": set(), "B": {"A"}, "C": {"B"}}) == {"A": 0, "B": 1, "C": 2}


def test_levels_takes_the_deepest_prerequisite():
    got = levels({"A": set(), "B": {"A"}, "C": {"A", "B"}})
    assert got["C"] == 2


def test_downstream_counts_are_transitive():
    got = downstream_counts({"A": set(), "B": {"A"}, "C": {"B"}})
    assert got["A"] == 2      # unlocks B, and C through it
    assert got["B"] == 1
    assert got["C"] == 0


def test_find_cycles_returns_nothing_for_a_dag():
    assert find_cycles({"A": set(), "B": {"A"}}) == []


def test_find_cycles_canonicalizes_members():
    """Sorted members and a sorted outer list, so output cannot vary by run."""
    assert find_cycles({"B": {"A"}, "A": {"B"}}) == [["A", "B"]]


def test_find_cycles_finds_a_self_loop():
    assert find_cycles({"A": {"A"}}) == [["A"]]


def test_find_cycles_reports_every_cycle():
    """Two disjoint 2-cycles plus a 3-cycle — Tarjan gets all three at once."""
    edges = {
        "A": {"B"}, "B": {"A"},
        "C": {"D"}, "D": {"C"},
        "E": {"G"}, "F": {"E"}, "G": {"F"},
        "H": set(),
    }
    assert find_cycles(edges) == [["A", "B"], ["C", "D"], ["E", "F", "G"]]


def test_find_cycles_ignores_edges_to_unknown_nodes():
    assert find_cycles({"A": {"GONE"}}) == []


def test_break_cycles_makes_the_graph_acyclic():
    edges = {"A": {"B"}, "B": {"A"}}
    removed = break_cycles(edges, find_cycles(edges), lambda c: (c,))

    assert removed
    assert find_cycles(edges) == []
    levels(edges)      # must not raise: proves acyclicity


def test_break_cycles_keeps_the_order_key_direction():
    """With A ordered first, B may depend on A but not the reverse."""
    edges = {"A": {"B"}, "B": {"A"}}
    break_cycles(edges, find_cycles(edges), lambda c: (c,))

    assert edges["B"] == {"A"}
    assert edges["A"] == set()


def test_break_cycles_removes_a_self_loop():
    edges = {"A": {"A"}}
    break_cycles(edges, find_cycles(edges), lambda c: (c,))
    assert edges["A"] == set()


# --- Corequisite bundling ------------------------------------------------------

def test_stated_corequisites_form_one_bundle():
    courses = {c.code: c for c in [C("LEC001", 3, coreqs=["LAB001"]),
                                   C("LAB001", 1)]}
    bundles, _ = coreq_bundles(courses, {"LEC001": 0, "LAB001": 0}, False)

    assert len(bundles) == 1
    assert bundles[0].codes == ("LAB001", "LEC001")
    assert bundles[0].units == 4
    assert bundles[0].inferred is False


def test_coreq_bundle_uses_the_max_prereq_level():
    courses = {c.code: c for c in [C("LEC001", 3, coreqs=["LAB001"]),
                                   C("LAB001", 1)]}
    bundles, _ = coreq_bundles(courses, {"LEC001": 2, "LAB001": 0}, False)
    assert bundles[0].level == 2


def test_ordinary_courses_become_one_element_bundles():
    courses = {c.code: c for c in [C("AAA111"), C("BBB222")]}
    bundles, _ = coreq_bundles(courses, {"AAA111": 0, "BBB222": 0}, False)

    assert [b.codes for b in bundles] == [("AAA111",), ("BBB222",)]


def test_inferred_lab_pairing_is_flagged():
    """§10.10.1 pairing is a heuristic, so it must announce itself."""
    courses = {c.code: c for c in [
        C("CCPROG1", 3, year=1, term=1, title="Logic Formulation"),
        C("LBYCPA1", 1, year=1, term=1, title="Logic Formulation Laboratory"),
    ]}
    bundles, notes = coreq_bundles(courses, {"CCPROG1": 0, "LBYCPA1": 0}, True)

    assert len(bundles) == 1
    assert bundles[0].codes == ("CCPROG1", "LBYCPA1")
    assert bundles[0].inferred is True
    assert any("LBYCPA1" in n and "10.10.1" in n for n in notes)


def test_pair_labs_false_disables_inference():
    courses = {c.code: c for c in [
        C("CCPROG1", 3, year=1, term=1, title="Logic Formulation"),
        C("LBYCPA1", 1, year=1, term=1, title="Logic Formulation Laboratory"),
    ]}
    bundles, notes = coreq_bundles(courses, {"CCPROG1": 0, "LBYCPA1": 0}, False)

    assert len(bundles) == 2
    assert notes == []


def test_lab_pairing_needs_a_unique_match():
    """Two identically-titled lectures: guessing would be worse than not."""
    courses = {c.code: c for c in [
        C("CCPROG1", 3, year=1, term=1, title="Logic Formulation"),
        C("CCPROG9", 3, year=1, term=1, title="Logic Formulation"),
        C("LBYCPA1", 1, year=1, term=1, title="Logic Formulation Laboratory"),
    ]}
    bundles, notes = coreq_bundles(courses, {c: 0 for c in courses}, True)

    assert len(bundles) == 3
    assert any("not paired" in n for n in notes)


def test_lab_pairing_requires_the_same_year_and_term():
    courses = {c.code: c for c in [
        C("CCPROG1", 3, year=1, term=1, title="Logic Formulation"),
        C("LBYCPA1", 1, year=2, term=1, title="Logic Formulation Laboratory"),
    ]}
    bundles, _ = coreq_bundles(courses, {c: 0 for c in courses}, True)
    assert len(bundles) == 2


# --- Packing -------------------------------------------------------------------

def test_no_prereqs_all_eligible_first_term_up_to_cap():
    result = plan(*[C(f"AAA{i:03d}", 3) for i in range(6)])

    assert len(result.terms[0].courses) == 5      # 5 x 3u = 15u cap
    assert result.terms[0].units == 15
    assert len(result.deferred) == 1
    assert len(result.available_now) == 6         # all six were takeable


def test_prereq_chain_levels_across_terms():
    result = plan(C("AAA111"), C("BBB222", prereqs=["AAA111"]),
                  C("CCC333", prereqs=["BBB222"]))

    assert codes_by_term(result) == [["AAA111"], ["BBB222"], ["CCC333"]]


def test_taken_courses_unlock_their_dependents():
    result = plan(C("AAA111", taken=True), C("BBB222", prereqs=["AAA111"]))

    assert codes_by_term(result) == [["BBB222"]]
    assert "AAA111" not in result.blocked


def test_taken_passed_in_by_the_caller_also_unlocks():
    result = plan(C("AAA111"), C("BBB222", prereqs=["AAA111"]),
                  taken={"aaa111"})              # lowercase on purpose

    assert codes_by_term(result) == [["BBB222"]]


def test_unknown_typed_code_is_reported_not_silently_dropped():
    result = plan(C("AAA111"), taken={"MATH101"})
    assert any("MATH101" in n for n in result.notes)


def test_unit_cap_is_never_exceeded():
    result = plan(*[C(f"AAA{i:03d}", 4) for i in range(9)])

    for term in result.terms:
        assert term.units <= 15, f"{term.label} exceeded the cap"


def test_checklist_max_units_override_wins():
    """§10.2 defers to the number indicated on the program checklist."""
    courses = [C(f"AAA{i:03d}", 3) for i in range(6)]
    result = plan(*courses, max_units_override=18.0)

    assert result.terms[0].units == 18


def test_corequisites_land_in_the_same_term():
    """A 3u lecture + 1u lab against a 3u cap must be bumped TOGETHER.

    Splitting them would violate §10.10.1; scheduling only the lab would be
    worse than scheduling neither.
    """
    result = plan(C("LEC001", 3, coreqs=["LAB001"]), C("LAB001", 1),
                  C("AAA111", 3), max_units=3.0)

    for term in result.terms:
        codes = {c.code for c in term.courses}
        assert ("LEC001" in codes) == ("LAB001" in codes)


def test_coreq_bundle_larger_than_cap_is_unreachable_not_dropped():
    """Reported and explained, never silently vanished, and it must terminate."""
    result = plan(C("LEC001", 12, coreqs=["LAB001"]), C("LAB001", 8),
                  C("AAA111", 3))

    assert set(result.unreachable) == {"LAB001", "LEC001"}
    assert any("exceeds" in n for n in result.notes)
    assert codes_by_term(result) == [["AAA111"]]


def test_below_floor_term_gets_a_note_but_is_not_overfilled():
    result = plan(C("AAA111", 3))

    assert result.terms[0].units == 3
    assert any("10.1" in n for n in result.notes)


def test_floor_note_is_absent_when_the_cap_is_what_limited_the_term():
    """A full 15-unit term is not "below the floor" — no §10.1 note belongs.

    The trailing 9-unit term here legitimately does get one (nothing else was
    eligible), so the invariant is about which term is flagged, not whether any
    note exists: the cap must never be reported as a shortfall.
    """
    result = plan(*[C(f"AAA{i:03d}", 3) for i in range(8)])

    assert result.terms[0].units == 15
    floor_notes = [n for n in result.notes if "10.1" in n]
    assert not any("next term" in n.lower() for n in floor_notes)


def test_available_now_includes_cap_deferred_courses():
    result = plan(*[C(f"AAA{i:03d}", 3) for i in range(7)])

    assert len(result.available_now) == 7
    assert len(result.deferred) == 2
    assert set(result.deferred) <= set(result.available_now)


def test_max_terms_stops_the_loop():
    """A 100-course chain must not run away; it stops and says so."""
    chain = [C("AAA000")]
    for i in range(1, 100):
        chain.append(C(f"AAA{i:03d}", prereqs=[f"AAA{i - 1:03d}"]))

    result = plan(*chain, max_terms=3)

    assert len(result.terms) == 3
    assert len(result.blocked) == 97
    assert any("horizon" in n for n in result.notes)


def test_blocked_courses_say_what_blocks_them():
    result = plan(C("LEC001", 12, coreqs=["LAB001"]), C("LAB001", 8),
                  C("BBB222", prereqs=["LEC001"]))

    assert "BBB222" in result.blocked
    assert any("BBB222" in n and "LEC001" in n for n in result.notes)


def test_placeholders_sort_after_real_courses_of_the_same_level():
    result = plan(C("PLACEHOLDER_GE_1", 3, placeholder=True), C("AAA111", 3),
                  max_units=3.0)

    assert codes_by_term(result)[0] == ["AAA111"]


def test_a_course_that_unlocks_more_is_preferred_among_equals():
    """Among same-level choices, take what shortens the plan."""
    result = plan(C("AAA111", 3), C("BBB222", 3),
                  C("CCC333", 3, prereqs=["BBB222"]),
                  C("DDD444", 3, prereqs=["CCC333"]),
                  max_units=3.0)

    assert codes_by_term(result)[0] == ["BBB222"]


# --- The checklist's own per-term unit limits ----------------------------------

def with_caps(*courses, caps=None, source=PrereqSource.COLUMN, **overrides):
    """Plan a curriculum that states its own per-term loads."""
    limits = {**LIMITS, **overrides}
    curriculum = cur(*courses, source=source)
    curriculum.term_caps = dict(caps or {})
    return build_plan(curriculum, set(), **limits)


def test_checklist_term_limit_beats_the_general_fifteen():
    """Undergraduate §10.2: 15 units "or the number indicated on the checklist".

    An engineering checklist prescribing 19 is the governing number, so the
    planner must not hold the term down to the general maximum.
    """
    courses = [C(f"AAA{i:03d}", 3, year=1, term=1) for i in range(8)]
    result = with_caps(*courses, caps={1: 19.0})

    assert result.terms[0].units == 18       # six 3-unit courses; 21 would exceed
    assert result.terms[0].cap == 19.0


def test_each_term_uses_its_own_prescribed_limit():
    courses = ([C(f"AAA{i:03d}", 3, year=1, term=1) for i in range(4)] +
               [C(f"BBB{i:03d}", 3, year=1, term=2,
                  prereqs=[f"AAA{i:03d}"]) for i in range(4)])
    result = with_caps(*courses, caps={1: 6.0, 2: 12.0})

    assert [t.cap for t in result.terms[:2]] == [6.0, 12.0]
    assert result.terms[0].units == 6.0


def test_planning_resumes_at_the_first_unfinished_term():
    """Someone who cleared terms 1-2 enters term 3, so term 3's limit applies."""
    done = [C("AAA111", 3, year=1, term=1, taken=True),
            C("BBB222", 3, year=1, term=2, taken=True)]
    todo = [C(f"CCC{i:03d}", 3, year=1, term=3) for i in range(4)]
    result = with_caps(*done, *todo, caps={1: 3.0, 2: 3.0, 3: 9.0})

    assert result.terms[0].cap == 9.0
    assert result.terms[0].checklist_term == 3


def test_non_credit_courses_do_not_consume_the_unit_limit():
    """NSTP and the Lasallian series must be taken but sit outside the load.

    This is the checklist's own arithmetic: "18 (3)" is 18 credited units plus
    a 3-unit non-credit course, and both are scheduled.
    """
    courses = [C("AAA111", 3, year=1, term=1), C("BBB222", 3, year=1, term=1),
               Course("NSTP101", "NSTP", 3.0, 1, 1, credited=False)]
    result = with_caps(*courses, caps={1: 6.0})

    scheduled = {c.code for c in result.terms[0].courses}
    assert scheduled == {"AAA111", "BBB222", "NSTP101"}
    assert result.terms[0].units == 6.0      # the 3 non-credit units excluded


def test_program_max_units_override_beats_the_per_term_limits():
    courses = [C(f"AAA{i:03d}", 3, year=1, term=1) for i in range(6)]
    curriculum = cur(*courses)
    curriculum.term_caps = {1: 19.0}
    curriculum.max_units_override = 9.0

    result = build_plan(curriculum, set(), **LIMITS)

    assert result.terms[0].units == 9.0


def test_a_heavier_checklist_load_is_explained_with_the_provision():
    result = with_caps(*[C(f"AAA{i:03d}", 3, year=1, term=1) for i in range(8)],
                       caps={1: 19.0})
    assert any("10.2" in n and "19" in n for n in result.notes)


def test_absent_term_caps_fall_back_to_the_configured_maximum():
    """A curriculum that says nothing about terms keeps the old behavior."""
    result = plan(*[C(f"AAA{i:03d}", 3) for i in range(8)])

    assert result.terms[0].units == 15
    assert result.terms[0].cap == 15.0


def test_bundle_over_every_term_limit_is_unreachable():
    courses = [C("LEC001", 12, year=1, term=1, coreqs=["LAB001"]),
               C("LAB001", 8, year=1, term=1)]
    result = with_caps(*courses, caps={1: 15.0})

    assert set(result.unreachable) == {"LAB001", "LEC001"}


# --- Retaking a failed course --------------------------------------------------

def test_a_removed_course_is_planned_again_as_a_retake():
    """The "I finished terms 1-5 but failed one" case.

    Marking a whole term done and then removing one course must put that course
    back into the plan, not leave it silently skipped.
    """
    result = plan(C("AAA111", 3, year=1, term=1, taken=True),
                  C("BBB222", 3, year=1, term=2, taken=True),
                  C("CCC333", 3, year=1, term=2, taken=True),
                  taken=set())          # nothing extra; the flags say it all

    assert result.terms == []           # everything already passed

    # Now the student says BBB222 was actually failed: it comes back.
    curriculum = cur(C("AAA111", 3, year=1, term=1, taken=True),
                     C("BBB222", 3, year=1, term=2),
                     C("DDD444", 3, year=1, term=3, prereqs=["BBB222"]))
    retake = build_plan(curriculum, set(), **LIMITS)

    assert codes_by_term(retake) == [["BBB222"], ["DDD444"]]


def test_dependents_of_a_failed_course_wait_for_the_retake():
    curriculum = cur(C("BBB222", 3, year=1, term=1),
                     C("CCC333", 3, year=1, term=2, prereqs=["BBB222"]),
                     C("DDD444", 3, year=1, term=2, prereqs=["CCC333"]))
    result = build_plan(curriculum, set(), **LIMITS)

    assert codes_by_term(result) == [["BBB222"], ["CCC333"], ["DDD444"]]


def test_a_failed_course_takes_its_corequisite_lab_with_it():
    """§10.10.1: retaking a lecture means retaking its laboratory alongside."""
    curriculum = cur(C("LOGDSGN", 3, year=1, term=1, coreqs=["LBYCPG4"]),
                     C("LBYCPG4", 1, year=1, term=1))
    result = build_plan(curriculum, set(), **LIMITS)

    assert set(codes_by_term(result)[0]) == {"LOGDSGN", "LBYCPG4"}


def test_passing_a_course_without_its_prerequisite_is_flagged():
    """Marking a whole term done, then failing one course inside it, can leave
    a dependent marked as passed. That cannot happen in reality, so say so."""
    result = plan(C("AAA111", 3, year=1, term=1),               # not passed
                  C("BBB222", 3, year=1, term=2, prereqs=["AAA111"], taken=True))

    assert any("BBB222" in n and "AAA111" in n and "passed" in n
               for n in result.notes)


def test_a_consistent_history_is_not_flagged():
    result = plan(C("AAA111", 3, year=1, term=1, taken=True),
                  C("BBB222", 3, year=1, term=2, prereqs=["AAA111"], taken=True),
                  C("CCC333", 3, year=1, term=3, prereqs=["BBB222"]))

    assert not any("marked as passed" in n for n in result.notes)


def test_many_contradictions_are_summarized_not_listed_forever():
    courses = [C("AAA111", 3, year=1, term=1)]
    courses += [C(f"BBB{i:03d}", 3, year=1, term=2, prereqs=["AAA111"],
                  taken=True) for i in range(9)]
    result = plan(*courses)

    flagged = [n for n in result.notes if "marked as passed" in n]
    assert len(flagged) == 5
    assert any("4 more" in n for n in result.notes)


# --- Soft prerequisites: sat, not necessarily passed ---------------------------

def soft(code: str, units: float = 3, *, soft_prereqs=(), **kw) -> Course:
    return Course(code=code, title=f"Course {code}", units=units,
                  confidence=PrereqConfidence.STATED,
                  soft_prereqs=tuple(soft_prereqs), **kw)


def test_soft_prerequisite_is_cleared_by_having_failed_it():
    """The rule the checklist's "S" marker means.

    ECNOMIC lists CALENG1 as a SOFT requisite: having sat Differential Calculus
    clears you for Engineering Economics even if you failed it. Treating that as
    a hard prerequisite would delay the course for no reason.
    """
    curriculum = cur(C("CALENG1", 3, year=1, term=1),
                     soft("ECNOMIC", 3, soft_prereqs=["CALENG1"], year=1, term=2))

    result = build_plan(curriculum, set(), attempted={"CALENG1"}, **LIMITS)

    first = codes_by_term(result)[0]
    assert "ECNOMIC" in first, "a failed soft prerequisite should still clear it"
    assert "CALENG1" in first, "the failed course itself is still retaken"


def test_hard_prerequisite_is_not_cleared_by_having_failed_it():
    """The contrast that makes the distinction worth having."""
    curriculum = cur(C("CALENG1", 3, year=1, term=1),
                     C("CALENG2", 3, year=1, term=2, prereqs=["CALENG1"]))

    result = build_plan(curriculum, set(), attempted={"CALENG1"}, **LIMITS)

    assert codes_by_term(result) == [["CALENG1"], ["CALENG2"]]


def test_soft_prerequisite_never_sat_still_orders_the_courses():
    """Soft does not mean optional: you must still have been through it."""
    curriculum = cur(C("CALENG1", 3, year=1, term=1),
                     soft("ECNOMIC", 3, soft_prereqs=["CALENG1"], year=1, term=2))

    result = build_plan(curriculum, set(), max_units=3.0, min_units=0.0,
                        max_terms=8)

    assert codes_by_term(result) == [["CALENG1"], ["ECNOMIC"]]


def test_passing_a_course_implies_having_attempted_it():
    curriculum = cur(C("CALENG1", 3, year=1, term=1, taken=True),
                     soft("ECNOMIC", 3, soft_prereqs=["CALENG1"], year=1, term=2))

    result = build_plan(curriculum, set(), **LIMITS)

    assert codes_by_term(result) == [["ECNOMIC"]]


def test_a_failing_grade_in_the_artifact_counts_as_attempted():
    """A recorded 0.0 means the course was sat, so it clears a soft requisite."""
    failed = Course("CALENG1", "Differential Calculus", 3, 1, 1,
                    confidence=PrereqConfidence.STATED, taken=False, grade="0.0")
    curriculum = cur(failed,
                     soft("ECNOMIC", 3, soft_prereqs=["CALENG1"], year=1, term=2))

    result = build_plan(curriculum, set(), **LIMITS)

    assert "ECNOMIC" in codes_by_term(result)[0]


def test_soft_prerequisite_off_the_checklist_is_noted_not_blocking():
    curriculum = cur(soft("ECNOMIC", 3, soft_prereqs=["GONE999"]))
    result = build_plan(curriculum, set(), **LIMITS)

    assert codes_by_term(result) == [["ECNOMIC"]]
    assert any("GONE999" in n for n in result.notes)


def test_failing_a_soft_prerequisite_is_not_an_impossible_history():
    """Passing ECNOMIC while having failed CALENG1 is legitimate, not a warning."""
    curriculum = cur(C("CALENG1", 3, year=1, term=1, grade="0.0"),
                     soft("ECNOMIC", 3, soft_prereqs=["CALENG1"], year=1, term=2,
                          taken=True))

    result = build_plan(curriculum, set(), **LIMITS)

    assert not any("marked as passed" in n for n in result.notes)


def test_never_having_sat_a_soft_prerequisite_is_an_impossible_history():
    curriculum = cur(C("CALENG1", 3, year=1, term=1),
                     soft("ECNOMIC", 3, soft_prereqs=["CALENG1"], year=1, term=2,
                          taken=True))

    result = build_plan(curriculum, set(), **LIMITS)

    assert any("ECNOMIC" in n and "CALENG1" in n for n in result.notes)


# --- Determinism ---------------------------------------------------------------

def test_tie_break_is_stable_under_input_reordering():
    """THE determinism guarantee (docs/course_planner.md §8).

    A study plan that changed between runs would be unusable, so the packing
    tie-break key is total by construction. Feeding the same courses in
    different insertion orders must produce an identical plan.
    """
    courses = [
        C("GEMATMW", 3, year=1, term=1),
        C("CSMATH2", 3, year=1, term=2, prereqs=["GEMATMW"]),
        C("CCPROG1", 3, year=1, term=1),
        C("CCPROG2", 3, year=1, term=2, prereqs=["CCPROG1"]),
        C("LBYCPA1", 1, year=1, term=1, coreqs=["CCPROG1"]),
        C("CSADPRG", 3, year=2, term=1, prereqs=["CCPROG2"]),
        C("PLACEHOLDER_GE_1", 3, year=2, term=1, placeholder=True),
        C("GERIZAL", 3, year=2, term=2),
    ]
    orderings = [
        courses,
        list(reversed(courses)),
        courses[3:] + courses[:3],
        sorted(courses, key=lambda c: c.code),
        sorted(courses, key=lambda c: -c.units),
    ]

    baseline = codes_by_term(plan(*orderings[0]))
    for shuffled in orderings[1:]:
        assert codes_by_term(plan(*shuffled)) == baseline


def test_repeated_runs_are_identical_even_with_a_cycle():
    courses = [C("AAA111", prereqs=["BBB222"]), C("BBB222", prereqs=["AAA111"]),
               C("CCC333")]
    first, second = plan(*courses), plan(*courses)

    assert first.cycles == second.cycles == [["AAA111", "BBB222"]]
    assert codes_by_term(first) == codes_by_term(second)


# --- Cycles end to end ---------------------------------------------------------

def test_cycle_is_reported_and_broken_deterministically():
    """A mutual prerequisite is an extraction error, not a curriculum.

    Refusing to plan would turn one bad cell into total failure, so both
    courses are still scheduled and the user is told which field to fix.
    """
    result = plan(C("AAA111", 3, year=1, term=1, prereqs=["BBB222"]),
                  C("BBB222", 3, year=1, term=2, prereqs=["AAA111"]))

    assert result.cycles == [["AAA111", "BBB222"]]
    scheduled = {c.code for term in result.terms for c in term.courses}
    assert scheduled == {"AAA111", "BBB222"}
    assert any("AAA111" in n and "BBB222" in n for n in result.notes)
    assert any("prereqs" in n for n in result.notes)


def test_self_loop_is_survived():
    result = plan(C("AAA111", 3, prereqs=["AAA111"]))

    assert result.cycles == [["AAA111"]]
    assert codes_by_term(result) == [["AAA111"]]
    assert any("itself" in n for n in result.notes)


def test_prereq_pointing_off_checklist_is_noted_not_blocking():
    result = plan(C("AAA111", 3, prereqs=["GONE999"]))

    assert codes_by_term(result) == [["AAA111"]]
    assert "AAA111" not in result.blocked
    assert any("GONE999" in n for n in result.notes)


# --- The three prereq_source cases ---------------------------------------------

def test_year_term_mode_orders_by_checklist_without_inventing_edges():
    """Case (b): ordering from the sheet's layout, and NO fabricated edges.

    Synthesizing "every Y1T1 course precedes every Y1T2 course" would draw a
    dense, confident, wrong graph — so levels come from the term index instead.
    """
    result = plan(C("BBB222", 3, year=1, term=2), C("AAA111", 3, year=1, term=1),
                  C("CCC333", 3, year=2, term=1),
                  source=PrereqSource.YEAR_TERM, max_units=3.0)

    assert codes_by_term(result) == [["AAA111"], ["BBB222"], ["CCC333"]]
    # Nothing was invented: no course gained a prerequisite.
    assert all(not c.prereqs for term in result.terms for c in term.courses)
    assert any("year/term" in n for n in result.notes)


def test_year_term_mode_puts_unplaced_courses_last():
    result = plan(C("AAA111", 3, year=1, term=1), C("ZZZ999", 3),
                  source=PrereqSource.YEAR_TERM, max_units=3.0)

    assert codes_by_term(result) == [["AAA111"], ["ZZZ999"]]


def test_none_mode_claims_no_ordering_and_says_so():
    """Case (c): a list of what remains, explicitly not an ordering."""
    result = plan(C("AAA111", 3), C("BBB222", 3),
                  source=PrereqSource.NONE, max_units=15.0)

    assert len(result.terms) == 1                 # one unordered bucket
    assert result.blocked == []
    assert any("not an ordering" in n for n in result.notes)


def test_unknown_prereq_courses_are_scheduled_but_flagged():
    result = plan(C("AAA111", 3, confidence=PrereqConfidence.UNKNOWN),
                  C("BBB222", 3, confidence=PrereqConfidence.UNKNOWN))

    assert len(result.terms) == 1
    assert any("absence of information" in n for n in result.notes)


# --- Degenerate inputs ---------------------------------------------------------

def test_empty_curriculum_produces_an_empty_plan_not_a_crash():
    result = build_plan(cur(), set(), **LIMITS)

    assert result.terms == []
    assert result.blocked == []
    assert result.notes == []


def test_everything_taken_produces_zero_terms_and_a_note():
    result = plan(C("AAA111", taken=True), C("BBB222", taken=True))

    assert result.terms == []
    assert any("already marked as passed" in n for n in result.notes)


def test_zero_unit_course_is_scheduled():
    """NSTP and PE rows sometimes carry 0 units; they must not vanish."""
    result = plan(C("NSTP01", 0), C("AAA111", 3))

    scheduled = {c.code for term in result.terms for c in term.courses}
    assert scheduled == {"NSTP01", "AAA111"}
