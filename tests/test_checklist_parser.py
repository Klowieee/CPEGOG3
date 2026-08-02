"""
test_checklist_parser.py — Tests for src/curriculum/checklist_parser.py (Phase 15).

This module carries all of the feature's risk, because a checklist's layout is
outside our control (AD-8). The cell parsers and the column-role vote are pure
functions, so almost everything here runs on synthetic rows with no PDF at all.

Two cases get special attention because getting them wrong would actively
mislead a student:
  * a failed or deferred grade must never count as "passed";
  * the requirement-TYPE column pairs positionally with the code column, so
    'H/C' beside 'A/B' means A is a prerequisite and B is a corequisite —
    mixing those up would either block a course wrongly or clear it too early.

A real-PDF test runs only when a checklist is present, the same idiom as
test_parser.py's HANDBOOK skip.

Dependencies:
    pytest, src.curriculum.checklist_parser.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.curriculum.checklist_parser import (      # noqa: E402
    ColumnRoles,
    RawRow,
    _rows_from_words,
    clean_cell,
    infer_column_roles,
    is_course_code,
    pair_requirements,
    parse_checklist,
    parse_grade,
    parse_term_banner,
    parse_units,
    parse_year_term,
    split_codes,
    split_prereq_types,
    synthetic_code,
    units_are_credited,
)
from src.curriculum.model import PrereqSource       # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKLISTS = sorted((PROJECT_ROOT / "data" / "checklists").glob("*.pdf"))


def rows(*cell_lists) -> list[RawRow]:
    return [RawRow([clean_cell(c) for c in cells], 1, "table_lines")
            for cells in cell_lists]


# --- Course-code recognition ---------------------------------------------------

@pytest.mark.parametrize("code", [
    "GEMATMW", "GERIZAL", "CSMATH2", "CSOPESY", "LBYCPA1", "ENG501M",
    "THSST1", "NSTP101", "SAS1000", "FNDMATH", "CALENG1", "THSCP4B",
    "LBYEC2A", "NSTPCW1", "PRCGECP",
])
def test_regex_accepts_real_dlsu_course_codes(code):
    assert is_course_code(code)


@pytest.mark.parametrize("token", [
    "COURSE", "TITLE", "UNITS", "GRADE", "TOTAL", "REMARKS", "PREREQ",
    "PREREQUISITES", "ELECTIVE", "NONE", "AND", "OF", "STUDENT", "LEGEND",
    "3.0", "Y1T1", "GeMatMw", "", "   ", "A", "-",
])
def test_regex_rejects_headers_and_prose(token):
    assert not is_course_code(token)


# --- Units ---------------------------------------------------------------------

@pytest.mark.parametrize("cell,expected", [
    ("3", 3.0), ("3.0", 3.0), ("0", 0.0), ("(3)", 3.0), (" 5 ", 5.0),
    ("1.5", 1.5),
])
def test_parse_units_reads_a_number(cell, expected):
    assert parse_units(cell) == expected


@pytest.mark.parametrize("cell", ["", "-", "N/A", "#REF!", "TOTAL", "abc",
                                  "178", "99"])
def test_parse_units_rejects_junk_and_totals(cell):
    """178 is a misread program total, not a course; #REF! is an Excel artifact."""
    assert parse_units(cell) is None


@pytest.mark.parametrize("cell,credited", [
    ("3", True), ("3.0", True), ("0", True), (" 5 ", True),
    ("(3)", False), (" (1) ", False),
])
def test_parenthesised_units_mark_a_non_credit_course(cell, credited):
    """The sheet writes NSTP and the Lasallian series as "(3)".

    They must be taken but do not count toward the term's unit load — which is
    how the checklist's own "18 (3)" term totals add up.
    """
    assert units_are_credited(cell) is credited
    assert parse_units(cell) is not None      # the value is still read


# --- Grades: the correctness point ---------------------------------------------

@pytest.mark.parametrize("cell", ["4.0", "3.5", "3.0", "2.5", "2.0", "1.5",
                                  "1.0", "P", "S", "CR"])
def test_passing_grade_marks_taken(cell):
    assert parse_grade(cell)[1] is True


@pytest.mark.parametrize("cell", ["0.0", "9.9", "INC", "INP", "W", "WP", "WF"])
def test_failing_deferred_and_withdrawn_are_not_taken(cell):
    """Crediting these would send a student into a course they are blocked from."""
    assert parse_grade(cell)[1] is False


def test_blank_grade_is_not_taken():
    assert parse_grade("") == (None, False)


def test_check_mark_column_marks_taken():
    for mark in ("X", "YES", "OK", "✓"):
        assert parse_grade(mark) == ("✓", True)


# --- Requirement pairing -------------------------------------------------------

def test_single_hard_prerequisite():
    assert pair_requirements("H", "FNDMATH") == (("FNDMATH",), (), (), None)


def test_single_corequisite_goes_to_coreqs():
    """'C' is how this sheet states a lab travelling with its lecture (§10.10.1)."""
    assert pair_requirements("C", "DATSRAL") == ((), (), ("DATSRAL",), None)


def test_soft_requirement_is_kept_separate_from_hard():
    """'S' means the course must have been SAT, not passed.

    Having failed Differential Calculus still clears you for Engineering
    Economics, so a soft requirement cannot live in `prereqs` — it is satisfied
    by a different set entirely (see planner._unmet_edges).
    """
    hard, soft, coreqs, _ = pair_requirements("S", "CALENG1")
    assert soft == ("CALENG1",)
    assert hard == () and coreqs == ()


def test_compound_types_pair_positionally_with_codes():
    """'H/H' + 'A/B' is two prerequisites — on this sheet '/' means AND."""
    hard, soft, coreqs, warning = pair_requirements("H/H", "FUNDLEC/LOGDSGN")
    assert hard == ("FUNDLEC", "LOGDSGN")
    assert soft == () and coreqs == ()
    assert warning is None


def test_mixed_types_split_three_ways():
    hard, soft, coreqs, _ = pair_requirements("H/C", "EMBDSYS/THSCP4A")
    assert hard == ("EMBDSYS",)
    assert coreqs == ("THSCP4A",)
    assert soft == ()


def test_soft_and_hard_in_one_cell_are_separated():
    """The real row: ENGPHYS is 'S / H' over 'CALENG1 / BASPHYS'."""
    hard, soft, coreqs, _ = pair_requirements("S / H", "CALENG1 / BASPHYS")
    assert soft == ("CALENG1",)
    assert hard == ("BASPHYS",)
    assert coreqs == ()


def test_spaced_and_multiline_requirement_cells_parse():
    hard, _, _, _ = pair_requirements("H/H/H", "ENGDATA/ GEPCOMM/\nLOGDSGN")
    assert hard == ("ENGDATA", "GEPCOMM", "LOGDSGN")


def test_mismatched_type_and_code_counts_warn_and_default_to_hard():
    """The conservative direction: a hard requirement can only delay a course."""
    hard, soft, coreqs, warning = pair_requirements("H", "AAA111/BBB222")
    assert hard == ("AAA111", "BBB222")
    assert soft == () and coreqs == ()
    assert warning and "do not line up" in warning


def test_empty_requirement_cell_yields_nothing():
    assert pair_requirements("", "") == ((), (), (), None)
    assert pair_requirements("", "NONE") == ((), (), (), None)


@pytest.mark.parametrize("cell", ["", "-", "--", "—", "N/A", "NONE", "NIL"])
def test_none_markers_yield_no_codes(cell):
    assert split_codes(cell) == []


def test_split_codes_handles_every_separator():
    assert split_codes("AAA111, BBB222; CCC333 / DDD444 & EEE555 and FFF666") == [
        "AAA111", "BBB222", "CCC333", "DDD444", "EEE555", "FFF666"]


def test_split_prereq_types_normalizes_case_and_spacing():
    assert split_prereq_types(" h / c ") == ["H", "C"]


# --- Term banners --------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("FIRST TERM", 1), ("SECOND TERM", 2), ("THIRD TERM", 3),
    ("FOURTH TERM", 4), ("SEVENTH TERM", 7), ("ELEVENTH TERM", 11),
    ("TWELFTH TERM", 12), ("Term 5", 5), ("2ND TERM", 2),
])
def test_parse_term_banner(text, expected):
    assert parse_term_banner(text) == expected


@pytest.mark.parametrize("text", ["", "COURSE", "TOTAL", "GEMATMW", "TERMINAL"])
def test_non_banners_are_rejected(text):
    assert parse_term_banner(text) is None


@pytest.mark.parametrize("text,expected", [
    ("Y3 T2", (3, 2)), ("Y1T1", (1, 1)), ("3-1", (3, 1)),
])
def test_parse_year_term(text, expected):
    assert parse_year_term(text) == expected


# --- Column-role voting --------------------------------------------------------

def test_columns_are_voted_not_guessed_per_cell():
    """The real ambiguity: units and grades are both small decimals.

    Here the grade column sits LEFT of the units column, so per-cell guessing
    would read grades as units. The column-level vote must not.
    """
    table = rows(
        ["AAA111", "Alpha", "4.0", "3"],
        ["BBB222", "Beta", "3.5", "3"],
        ["CCC333", "Gamma", "2.0", "1"],
        ["DDD444", "Delta", "", "3"],
    )
    roles = infer_column_roles(table)

    assert roles.code == 0
    assert roles.grade == 2
    assert roles.units == 3


def test_prereq_type_column_is_recognized():
    table = rows(
        ["AAA111", "Alpha", "3", "H", "BBB222"],
        ["CCC333", "Gamma", "3", "C", "AAA111"],
        ["DDD444", "Delta", "3", "H/H", "AAA111/BBB222"],
    )
    roles = infer_column_roles(table)

    assert roles.code == 0
    assert roles.units == 2
    assert roles.prereq_type == 3
    assert roles.prereq == 4


def test_title_column_is_whatever_is_left():
    table = rows(["AAA111", "A rather long course title here", "3"],
                 ["BBB222", "Another long course title here", "3"])
    assert infer_column_roles(table).title == 1


def test_header_row_names_a_column_voting_left_empty():
    table = rows(["COURSE", "COURSE TITLE", "UNITS", "GRADE"],
                 ["AAA111", "Alpha", "3", ""],
                 ["BBB222", "Beta", "3", ""])
    roles = infer_column_roles(table)

    assert roles.code == 0
    assert roles.grade == 3          # no data to vote on; the header settled it


def test_voting_wins_over_a_merged_header_that_spans_two_columns():
    """The real sheet's 'PREREQUISITES' header spans the type and code columns.

    If the header were allowed to override, the type column would be relabelled
    and every corequisite would silently become a prerequisite.
    """
    table = rows(
        ["COURSE", "COURSE TITLE", "UNITS", "PREREQUISITES", ""],
        ["AAA111", "Alpha", "3", "H", "BBB222"],
        ["CCC333", "Gamma", "3", "C", "AAA111"],
        ["DDD444", "Delta", "3", "H", "AAA111"],
    )
    roles = infer_column_roles(table)

    assert roles.prereq_type == 3
    assert roles.prereq == 4


def test_empty_rows_do_not_crash_the_vote():
    assert infer_column_roles([]) == ColumnRoles()


# --- Tier C: geometry fallback -------------------------------------------------

def word(text, x0, top=100.0):
    return {"text": text, "x0": x0, "x1": x0 + 6.0 * len(text), "top": top}


def test_x0_clustering_recovers_columns_without_a_grid():
    """Tier C rebuilds rows from word geometry, with no table lines at all."""
    words = [
        word("AAA111", 50), word("Alpha", 120), word("Course", 155), word("3", 300),
        word("BBB222", 50, 130), word("Beta", 120, 130), word("3", 300, 130),
    ]
    built = _rows_from_words(words, page_number=1)

    assert len(built) == 2
    assert built[0].cells == ["AAA111", "Alpha Course", "3"]
    assert built[1].cells == ["BBB222", "Beta", "3"]


def test_words_on_the_same_baseline_join_into_one_row():
    built = _rows_from_words([word("AAA111", 50, 100.0),
                              word("Alpha", 120, 102.0)], 1)
    assert len(built) == 1


# --- Placeholders --------------------------------------------------------------

def test_placeholder_code_is_synthesized_from_the_title():
    assert synthetic_code("GE ELECTIVE 1", set()) == "PLACEHOLDER_GE_ELECTIVE_1"


def test_synthetic_codes_do_not_collide():
    used = {"PLACEHOLDER_MAJOR_ELECTIVE"}
    assert synthetic_code("MAJOR ELECTIVE", used) == "PLACEHOLDER_MAJOR_ELECTIVE_2"


def test_clean_cell_collapses_wrapped_values():
    assert clean_cell("Data Structures\nand  Algorithms") == \
        "Data Structures and Algorithms"


# --- Against the real checklist ------------------------------------------------

@pytest.mark.skipif(not CHECKLISTS, reason="no checklist PDF in data/checklists")
def test_parse_real_checklist_invariants():
    """Whatever checklist is present must parse into a coherent curriculum.

    Deliberately asserts shape rather than exact contents, so a different
    program's checklist does not fail the suite.
    """
    curriculum, report = parse_checklist(CHECKLISTS[0])

    assert report.tier in {"table_lines", "table_text", "words_x0"}
    assert len(curriculum.courses) >= 20
    assert len(curriculum.courses) == len({c.code for c
                                           in curriculum.courses.values()})
    assert 60 <= sum(c.units for c in curriculum.courses.values()) <= 300
    assert all(c.title for c in curriculum.courses.values())
    assert all(0 <= c.units <= 15 for c in curriculum.courses.values())

    # Every prerequisite must name a course that exists on the same checklist;
    # dangling codes are the signature of a column read wrongly.
    known = set(curriculum.courses)
    dangling = {p for c in curriculum.courses.values()
                for p in tuple(c.prereqs) + tuple(c.coreqs) if p not in known}
    assert not dangling, f"prerequisites naming unknown courses: {sorted(dangling)}"


@pytest.mark.skipif(not CHECKLISTS, reason="no checklist PDF in data/checklists")
def test_real_checklist_states_prerequisites():
    curriculum, _ = parse_checklist(CHECKLISTS[0])

    assert curriculum.prereq_source is PrereqSource.COLUMN
    assert any(c.prereqs for c in curriculum.courses.values())


@pytest.mark.skipif(not CHECKLISTS, reason="no checklist PDF in data/checklists")
def test_real_checklist_yields_a_unit_limit_for_every_term():
    """The per-term loads are what §10.2 defers to, so they must all be read."""
    curriculum, _ = parse_checklist(CHECKLISTS[0])
    placed = {c.term_index(curriculum.terms_per_year)
              for c in curriculum.courses.values()} - {None}

    assert curriculum.term_caps
    assert set(curriculum.term_caps) == placed
    assert all(0 < cap <= 30 for cap in curriculum.term_caps.values())
