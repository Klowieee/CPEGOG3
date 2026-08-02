"""
test_plan_view.py — Tests for src/chat/plan_view.py (Phase 15).

The view layer has one job that matters to correctness: never let the terminal
imply more certainty than the plan has. So these tests are mostly about what
IS printed — the caveats, the "citation unavailable" notice, the banner when
ordering came from the checklist's layout rather than real prerequisites —
rather than about layout.

Rendering is checked by capturing a rich Console to a buffer; no terminal, no
index, no model, no network.

Dependencies:
    pytest, rich, src.chat.plan_view.
"""

import sys
from pathlib import Path

import pytest
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat.core import CoursePlan                     # noqa: E402
from src.chat.plan_view import (                         # noqa: E402
    _discover_checklists,
    _summarize,
    courses_in_terms,
    parse_code_list,
    parse_term_ranges,
    render_course_catalog,
    render_extraction_summary,
    render_plan,
)
from src.curriculum.model import (                       # noqa: E402
    Course,
    Curriculum,
    PrereqConfidence,
    PrereqSource,
)
from src.curriculum.planner import build_plan            # noqa: E402
from src.curriculum.policy import PolicyRule             # noqa: E402


def render(fn, *args, **kwargs) -> str:
    """Run a renderer against a captured console and return the plain text."""
    console = Console(width=100, record=True, force_terminal=False)
    fn(console, *args, **kwargs)
    return console.export_text()


def curriculum(source=PrereqSource.COLUMN) -> Curriculum:
    courses = [
        Course("GEMATMW", "Mathematics in the Modern World", 3, 1, 1,
               (), (), PrereqConfidence.STATED, True, "3.5"),
        Course("CSMATH2", "Discrete Structures", 3, 1, 2,
               ("GEMATMW",), (), PrereqConfidence.STATED, False, "0.0"),
        Course("CSADPRG", "Advanced Programming", 3, 2, 1,
               ("CSMATH2",), (), PrereqConfidence.STATED),
    ]
    return Curriculum("bscs-st", "BS Computer Science", 3,
                      {c.code: c for c in courses}, source)


def planned(source=PrereqSource.COLUMN, policy=None, **limits) -> CoursePlan:
    cur = curriculum(source)
    plan = build_plan(cur, set(), **{"max_units": 15.0, "min_units": 12.0,
                                     "max_terms": 8, **limits})
    return CoursePlan(plan=plan, policy=policy or [], curriculum=cur)


RULE = PolicyRule(key="max_units",
                  statement="Terms are capped at 15 units.",
                  value=15.0,
                  citation="Undergraduate, Section 10, pp. 101-102",
                  similarity=0.71,
                  excerpt="the maximum academic load ... is 15 units")


# --- parse_code_list -----------------------------------------------------------

def test_parse_code_list_normalizes_and_dedupes():
    assert parse_code_list(" gematmw, cSmAtH2  ccprog1,,gematmw ") == [
        "GEMATMW", "CSMATH2", "CCPROG1"]


def test_parse_code_list_of_nothing_is_empty():
    assert parse_code_list("") == []
    assert parse_code_list("   ,  , ") == []


# --- Picking whole terms -------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("1-6", {1, 2, 3, 4, 5, 6}),
    ("1-5,7", {1, 2, 3, 4, 5, 7}),
    ("3 1 2", {1, 2, 3}),
    ("1", {1}),
    ("1 to 3", {1, 2, 3}),
    ("6-1", {1, 2, 3, 4, 5, 6}),        # reversed range still means the range
])
def test_parse_term_ranges(text, expected):
    assert parse_term_ranges(text, 12) == expected


@pytest.mark.parametrize("text", ["", "banana", "0", "99", "-", "13"])
def test_parse_term_ranges_drops_what_it_cannot_use(text):
    """A typo must be visible in the confirmation, not fatal."""
    assert parse_term_ranges(text, 12) == set()


def test_parse_term_ranges_clamps_to_the_program_length():
    assert parse_term_ranges("1-99", 12) == set(range(1, 13))


def test_courses_in_terms_uses_the_running_term_index():
    cur = curriculum()          # trimester: Y1T1=1, Y1T2=2, Y2T1=4
    assert courses_in_terms(cur, {1}) == {"GEMATMW"}
    assert courses_in_terms(cur, {1, 2}) == {"GEMATMW", "CSMATH2"}
    assert courses_in_terms(cur, {4}) == {"CSADPRG"}


def test_courses_in_terms_of_nothing_is_empty():
    assert courses_in_terms(curriculum(), set()) == set()


@pytest.mark.parametrize("terms,expected", [
    ({1, 2, 3}, "1-3"), ({1, 2, 3, 5}, "1-3, 5"), ({4}, "4"),
    ({1, 3, 5}, "1, 3, 5"),
])
def test_summarize_reads_back_like_the_input(terms, expected):
    assert _summarize(terms) == expected


# --- The course catalog --------------------------------------------------------

def test_catalog_lists_every_course_grouped_by_term():
    text = render(render_course_catalog, curriculum())

    for code in ("GEMATMW", "CSMATH2", "CSADPRG"):
        assert code in text
    assert "Term  1" in text and "Term  2" in text


def test_catalog_brackets_non_credit_units():
    """The checklist's own notation, so the term totals still add up."""
    cur = curriculum()
    cur.courses["NSTP101"] = Course("NSTP101", "NSTP", 3, 1, 1, credited=False)
    text = render(render_course_catalog, cur)

    assert "NSTP101 [3]" in text


def test_catalog_ticks_courses_already_marked():
    text = render(render_course_catalog, curriculum(), {"GEMATMW"})
    assert "✓GEMATMW" in text


def test_catalog_shows_the_per_term_unit_limit():
    cur = curriculum()
    cur.term_caps = {1: 17.0}
    assert "17u" in render(render_course_catalog, cur)


# --- Extraction summary --------------------------------------------------------

def test_extraction_summary_leads_with_the_three_decisive_lines():
    """These three answer "which extraction case am I in?" at a glance."""
    text = render(render_extraction_summary, curriculum())

    assert "PREREQUISITE SOURCE:" in text
    assert "YEAR/TERM GROUPING:" in text
    assert "ALREADY TAKEN:" in text


def test_extraction_summary_excludes_failed_courses_from_taken():
    """CSMATH2 has a 0.0 — one of three courses is passed, not two."""
    text = render(render_extraction_summary, curriculum())
    assert "1 course(s)" in text


def test_extraction_summary_shows_warnings():
    cur = curriculum()
    cur.warnings.append("p.3 row 27: prereq cell contained 'or'")
    text = render(render_extraction_summary, cur)

    assert "Warnings (1)" in text
    assert "row 27" in text


def test_extraction_summary_points_at_the_editable_file():
    """The escape hatch is worthless if the user is not told where it is."""
    text = render(render_extraction_summary, curriculum(),
                  Path("data/checklists/bscs-st.curriculum.yaml"))

    assert "bscs-st.curriculum.yaml" in text
    assert "not the PDF" in text


# --- Plan rendering ------------------------------------------------------------

def test_render_plan_lists_terms_and_courses():
    text = render(render_plan, planned())

    assert "Next term" in text
    assert "CSMATH2" in text
    assert "CSADPRG" in text


def test_render_plan_shows_the_citation_for_each_constraint():
    text = render(render_plan, planned(policy=[RULE]))

    assert "Terms are capped at 15 units." in text
    assert "Undergraduate, Section 10" in text


def test_render_plan_flags_a_constraint_with_no_citation():
    """A missing citation must be visible, not silently implied."""
    ungrounded = PolicyRule("max_units", "Terms are capped at 15 units.",
                            15.0, None, 0.10, "")
    text = render(render_plan, planned(policy=[ungrounded]))

    assert "Terms are capped at 15 units." in text
    assert "unavailable" in text.lower()


def test_render_plan_prints_caveats():
    text = render(render_plan, planned())
    assert "Caveats:" in text


def test_render_plan_always_says_it_is_not_advice():
    """The plan does not know what is actually offered; it must say so."""
    text = render(render_plan, planned())
    assert "not advice" in text


def test_render_plan_warns_when_ordering_came_from_the_sheet_layout():
    """Case (b) must not look like a real dependency graph."""
    text = render(render_plan, planned(source=PrereqSource.YEAR_TERM))
    assert "year/term layout" in text


def test_render_plan_warns_when_nothing_could_be_established():
    text = render(render_plan, planned(source=PrereqSource.NONE))
    assert "not an ordering" in text


def test_render_plan_shows_deferred_courses():
    """Cap-bumped courses are a choice the student has, so they must be shown."""
    courses = [Course(f"AAA{i:03d}", f"Course {i}", 3) for i in range(7)]
    cur = Curriculum("x", "X", 3, {c.code: c for c in courses},
                     PrereqSource.COLUMN)
    plan = build_plan(cur, set(), max_units=15.0, min_units=12.0, max_terms=8)
    text = render(render_plan, CoursePlan(plan, [], cur))

    # Collapsed, because rich wraps a table title and the assertion is about
    # the words being present, not about where the line breaks fall.
    assert "over the unit cap" in " ".join(text.split())


def test_render_plan_reports_an_error_instead_of_a_plan():
    result = planned()
    result.error = "Could not read your curriculum file."
    text = render(render_plan, result)

    assert "Could not read your curriculum file." in text
    assert "Next term" not in text


def test_render_plan_handles_an_empty_curriculum():
    empty = Curriculum("x", "X", 3, {}, PrereqSource.NONE)
    plan = build_plan(empty, set(), max_units=15.0, min_units=12.0, max_terms=8)
    text = render(render_plan, CoursePlan(plan, [], empty))

    assert "nothing left to schedule" in text


def test_render_plan_mentions_the_written_flowchart():
    text = render(render_plan, planned(), Path("data/plans/bscs-st-plan.md"))
    assert "bscs-st-plan.md" in text


# --- Finding the checklist without being asked ---------------------------------

class FakePlannerSettings:
    def __init__(self, checklist_dir):
        self.checklist_dir = checklist_dir


class FakeSettings:
    def __init__(self, checklist_dir):
        self.planner = FakePlannerSettings(checklist_dir)


def test_discovery_prefers_the_corrected_artifact_over_its_pdf(tmp_path):
    """Re-parsing a PDF that already has an artifact would discard hand edits."""
    (tmp_path / "bscs.pdf").write_bytes(b"%PDF-")
    (tmp_path / "bscs.curriculum.yaml").write_text("x", encoding="utf-8")

    found = _discover_checklists(FakeSettings(tmp_path))

    assert [p.name for p in found] == ["bscs.curriculum.yaml"]


def test_discovery_offers_a_pdf_that_has_not_been_extracted(tmp_path):
    (tmp_path / "bscpe.pdf").write_bytes(b"%PDF-")
    found = _discover_checklists(FakeSettings(tmp_path))

    assert [p.name for p in found] == ["bscpe.pdf"]


def test_discovery_of_an_empty_directory_finds_nothing(tmp_path):
    assert _discover_checklists(FakeSettings(tmp_path)) == []


def test_discovery_survives_a_missing_directory(tmp_path):
    assert _discover_checklists(FakeSettings(tmp_path / "nope")) == []


def test_header_counts_courses_the_planner_treated_as_passed():
    """Regression: the header read the artifact's `taken:` flags, so a checklist
    with no grade column (the real CpE one) reported "0 units completed" for a
    student who had typed in two years of courses. It must report the set the
    planner actually used.
    """
    cur = curriculum()
    for course in cur.courses.values():          # nothing flagged in the file
        cur.courses[course.code] = type(course)(
            **{**course.__dict__, "taken": False})
    plan = build_plan(cur, {"GEMATMW"}, max_units=15.0, min_units=12.0,
                      max_terms=8)
    result = CoursePlan(plan, [], cur, taken=frozenset({"GEMATMW"}))

    text = " ".join(render(render_plan, result).split())

    assert "3 of 9 units completed" in text
    assert "2 course(s) remaining" in text
