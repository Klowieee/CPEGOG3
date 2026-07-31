"""
test_html_report.py — Tests for src/curriculum/html_report.py (Phase 15).

The plan page replaced a node-and-arrow diagram, so the properties worth
testing changed with it. What matters now:

  * it is genuinely self-contained — no network reference can sneak in, or the
    file stops working offline and on a printout;
  * every course in the plan actually appears, and its requirements are
    written on its own card (that text IS the arrows);
  * a hard requirement and a soft one are visibly different, because that
    distinction decides whether a failed attempt still clears the course;
  * anything the plan could not schedule is still shown, never dropped;
  * user text is HTML-escaped — course titles come from a PDF we do not control.

Dependencies:
    pytest, src.curriculum.html_report.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.curriculum.html_report import (          # noqa: E402
    render_plan_html,
    status_of,
    write_plan_html,
)
from src.curriculum.model import (                # noqa: E402
    Course,
    Curriculum,
    PlannedTerm,
    PrereqConfidence,
    PrereqSource,
    StudyPlan,
)


def C(code, units=3, **kw) -> Course:
    kw.setdefault("confidence", PrereqConfidence.STATED)
    return Course(code=code, title=kw.pop("title", f"Course {code}"),
                  units=units, **kw)


def curriculum(*courses, source=PrereqSource.COLUMN) -> Curriculum:
    return Curriculum("bscpe", "BS Computer Engineering", 3,
                      {c.code: c for c in courses}, source)


def plan(*terms, **kw) -> StudyPlan:
    return StudyPlan(terms=list(terms), available_now=kw.get("available_now", []),
                     deferred=kw.get("deferred", []), blocked=kw.get("blocked", []),
                     unreachable=kw.get("unreachable", []),
                     cycles=kw.get("cycles", []), notes=kw.get("notes", []))


def term(index, *courses, units=None, cap=0.0, checklist_term=None) -> PlannedTerm:
    label = "Next term" if index == 1 else f"Term +{index}"
    return PlannedTerm(index=index, label=label, courses=list(courses),
                       units=units if units is not None
                       else sum(c.units for c in courses),
                       cap=cap, checklist_term=checklist_term)


class FakeRule:
    """Duck-typed PolicyRule; html_report must not import the real one."""

    def __init__(self, statement, citation=None, excerpt=""):
        self.statement = statement
        self.citation = citation
        self.excerpt = excerpt


# --- Self-containment ----------------------------------------------------------

def test_document_is_well_formed_html():
    html = render_plan_html(plan(term(1, C("AAA111"))), curriculum(C("AAA111")))

    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert "<style>" in html and "</style>" in html


def test_nothing_is_fetched_from_the_network():
    """It has to work offline and when printed — no CDN, font, or image URL."""
    html = render_plan_html(plan(term(1, C("AAA111"))), curriculum(C("AAA111")))

    for token in ("http://", "https://", "<script", "<img", "@import", "src="):
        assert token not in html, f"external reference found: {token}"


# --- Every course appears, with its requirements -------------------------------

def test_every_planned_course_appears():
    cur = curriculum(C("AAA111"), C("BBB222"), C("CCC333"))
    html = render_plan_html(
        plan(term(1, cur.courses["AAA111"], cur.courses["BBB222"]),
             term(2, cur.courses["CCC333"])), cur)

    for code in ("AAA111", "BBB222", "CCC333"):
        assert code in html


def test_hard_requirement_is_written_on_the_card():
    """The card text is what replaced the arrows, so it has to carry them."""
    cur = curriculum(C("GEMATMW"), C("CSMATH2", prereqs=("GEMATMW",)))
    html = render_plan_html(plan(term(1, cur.courses["CSMATH2"])), cur)

    assert "needs" in html and "GEMATMW" in html


def test_soft_requirement_reads_differently_from_a_hard_one():
    """"after" vs "needs" is the difference between sat and passed."""
    cur = curriculum(C("CALENG1"), C("ECNOMIC", soft_prereqs=("CALENG1",)))
    html = render_plan_html(plan(term(1, cur.courses["ECNOMIC"])), cur)

    assert "after" in html
    assert "<b>needs</b> CALENG1" not in html


def test_corequisite_is_labelled_with():
    cur = curriculum(C("LOGDSGN"), C("LBYCPG4", 1, coreqs=("LOGDSGN",)))
    html = render_plan_html(plan(term(1, cur.courses["LBYCPG4"])), cur)

    assert "with" in html and "LOGDSGN" in html


def test_requirements_naming_unknown_courses_are_not_shown():
    """A dangling code would be noise on the card; the plan notes cover it."""
    cur = curriculum(C("AAA111", prereqs=("GONE999",)))
    html = render_plan_html(plan(term(1, cur.courses["AAA111"])), cur)

    assert "GONE999" not in html


def test_non_credit_units_are_bracketed():
    cur = curriculum(C("NSTP101", 3, credited=False))
    html = render_plan_html(plan(term(1, cur.courses["NSTP101"])), cur)

    assert "(3u)" in html


def test_unlocks_count_is_shown_when_supplied():
    cur = curriculum(C("MICPROS"))
    html = render_plan_html(plan(term(1, cur.courses["MICPROS"])), cur,
                            downstream={"MICPROS": 14})

    assert "unlocks 14" in html


# --- Statuses ------------------------------------------------------------------

@pytest.mark.parametrize("index,expected", [(0, "taken"), (1, "ready"), (3, "later")])
def test_status_follows_the_term_it_lands_in(index, expected):
    assert status_of(C("AAA111"), index, set()) == expected


def test_a_passed_course_is_taken_whatever_term_it_appears_in():
    assert status_of(C("AAA111"), 2, {"AAA111"}) == "taken"


def test_unknown_confidence_outranks_ready():
    """Uncertainty must stay visible; a confident blue would be a lie."""
    course = C("AAA111", confidence=PrereqConfidence.UNKNOWN)
    assert status_of(course, 1, set()) == "unknown"


def test_all_four_statuses_have_a_legend_entry():
    html = render_plan_html(plan(term(1, C("AAA111"))), curriculum(C("AAA111")))

    for label in ("Already passed", "Take next term", "Scheduled later",
                  "No prerequisite info"):
        assert label in html


# --- Term panels ---------------------------------------------------------------

def test_each_term_shows_its_own_cap_and_checklist_term():
    """The cap comes from the checklist, so it must be attributable."""
    cur = curriculum(C("AAA111"))
    html = render_plan_html(
        plan(term(1, cur.courses["AAA111"], cap=16.0, checklist_term=7)), cur)

    assert "of 16 units" in html
    assert "checklist term 7" in html


def test_already_passed_panel_is_shown_and_can_be_omitted():
    cur = curriculum(C("AAA111", taken=True), C("BBB222"))
    scheduled = plan(term(1, cur.courses["BBB222"]))

    with_panel = render_plan_html(scheduled, cur, include_taken=True)
    without = render_plan_html(scheduled, cur, include_taken=False)

    assert "Already passed" in with_panel and "AAA111" in with_panel
    assert "AAA111" not in without.split('class="legend"')[1]


def test_taken_argument_overrides_the_curriculum_flags():
    """A checklist with no grade column carries no flags — the caller knows."""
    cur = curriculum(C("AAA111"), C("BBB222"))
    html = render_plan_html(plan(term(1, cur.courses["BBB222"])), cur,
                            taken={"AAA111"})

    assert "1 course" in html          # remaining count reflects the override


# --- Nothing is silently dropped -----------------------------------------------

def test_deferred_blocked_and_unreachable_are_all_rendered():
    cur = curriculum(C("AAA111"), C("BBB222"), C("CCC333"), C("DDD444"))
    html = render_plan_html(
        plan(term(1, cur.courses["AAA111"]),
             deferred=["BBB222"], blocked=["CCC333"], unreachable=["DDD444"]), cur)

    assert "over the unit cap" in html and "BBB222" in html
    assert "Still blocked" in html and "CCC333" in html
    assert "Cannot be scheduled" in html and "DDD444" in html


def test_empty_sections_are_omitted_entirely():
    html = render_plan_html(plan(term(1, C("AAA111"))), curriculum(C("AAA111")))

    assert "Still blocked" not in html
    assert "Cannot be scheduled" not in html


def test_caveats_are_rendered():
    html = render_plan_html(
        plan(term(1, C("AAA111")), notes=["CALENG1 will be retaken."]),
        curriculum(C("AAA111")))

    assert "Caveats" in html and "CALENG1 will be retaken." in html


def test_the_page_always_says_it_is_not_advice():
    html = render_plan_html(plan(term(1, C("AAA111"))), curriculum(C("AAA111")))
    assert "not advice" in html


def test_an_empty_plan_still_renders():
    html = render_plan_html(plan(), curriculum(C("AAA111", taken=True)))

    assert html.startswith("<!DOCTYPE html>")
    assert "Nothing to plan" in html


# --- Policy citations ----------------------------------------------------------

def test_policy_rules_and_citations_are_shown():
    html = render_plan_html(
        plan(term(1, C("AAA111"))), curriculum(C("AAA111")),
        [FakeRule("Terms are capped at 16 units.",
                  "Undergraduate, Section 10, pp. 101-102",
                  "the maximum academic load ... is 15 units")])

    assert "Constraints applied" in html
    assert "Terms are capped at 16 units." in html
    assert "Undergraduate, Section 10, pp. 101-102" in html
    assert "the maximum academic load" in html


def test_a_rule_without_a_citation_says_so():
    """Silence would imply grounding the plan does not have."""
    html = render_plan_html(plan(term(1, C("AAA111"))), curriculum(C("AAA111")),
                            [FakeRule("Terms are capped at 15 units.")])

    assert "citation unavailable" in html


def test_constraints_section_is_omitted_when_there_are_no_rules():
    html = render_plan_html(plan(term(1, C("AAA111"))), curriculum(C("AAA111")))
    assert "Constraints applied" not in html


# --- Escaping ------------------------------------------------------------------

def test_titles_from_the_pdf_are_escaped():
    """Course titles come from a PDF we do not control."""
    cur = curriculum(C("AAA111", title='Data <b>Structures</b> & "Algorithms"'))
    html = render_plan_html(plan(term(1, cur.courses["AAA111"])), cur)

    assert "<b>Structures</b>" not in html
    assert "&lt;b&gt;Structures&lt;/b&gt;" in html
    assert "&amp;" in html


def test_a_note_containing_markup_is_escaped():
    html = render_plan_html(plan(term(1, C("AAA111")), notes=["<script>x</script>"]),
                            curriculum(C("AAA111")))

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# --- Writing -------------------------------------------------------------------

def test_write_creates_parents_and_returns_the_path(tmp_path):
    path = write_plan_html("<!DOCTYPE html><html></html>",
                           tmp_path / "deep" / "er" / "plan.html")

    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_write_uses_utf8(tmp_path):
    path = write_plan_html("<!DOCTYPE html><p>§10.2 — 19u</p>",
                           tmp_path / "plan.html")

    assert "§10.2 — 19u" in path.read_text(encoding="utf-8")
