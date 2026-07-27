"""
test_curriculum_model.py — Tests for src/curriculum/model.py (Phase 15).

The curriculum YAML is the contract the whole planner depends on
(Architectural Decision AD-8): the checklist PDF is parsed once into a file
the user can correct by hand, and nothing downstream ever reads the PDF again.
So these tests are mostly about the round trip surviving a hand edit, and
about the two ways this artifact could silently ruin someone's plan —
overwriting their corrections, and counting a failed course as passed.

No PDF, no network, no model needed.

Dependencies:
    pytest, pyyaml, src.curriculum.model.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.curriculum.model import (            # noqa: E402
    SCHEMA_VERSION,
    Course,
    Curriculum,
    CurriculumError,
    PrereqConfidence,
    PrereqSource,
    credited_units,
    derive_term_caps,
    load_curriculum_yaml,
    remaining_courses,
    taken_codes,
    total_units,
    write_curriculum_yaml,
)


def make_curriculum(*courses: Course, source=PrereqSource.COLUMN,
                    terms_per_year: int = 3, max_units=None,
                    warnings=None) -> Curriculum:
    """A Curriculum from positional Courses, keyed in the order given."""
    return Curriculum(
        program_id="bscs-st",
        program_name="BS Computer Science major in Software Technology",
        terms_per_year=terms_per_year,
        courses={c.code: c for c in courses},
        prereq_source=source,
        max_units_override=max_units,
        warnings=list(warnings or []),
    )


MATH = Course("GEMATMW", "Mathematics in the Modern World", 3, 1, 1,
              (), (), PrereqConfidence.STATED, True, "3.5", False, 1)
DISC = Course("CSMATH2", "Discrete Structures", 3, 1, 2,
              ("GEMATMW",), (), PrereqConfidence.STATED, False, None, False, 1)
LAB = Course("LBYCPA1", "Logic Formulation Laboratory", 1, 1, 1,
             (), ("CCPROG1",), PrereqConfidence.STATED, False, None, False, 1)
LEC = Course("CCPROG1", "Logic Formulation", 3, 1, 1,
             (), ("LBYCPA1",), PrereqConfidence.STATED, False, None, False, 1)


# --- Round trip ----------------------------------------------------------------

def test_yaml_round_trip_is_lossless(tmp_path):
    original = make_curriculum(MATH, DISC, LAB, LEC, max_units=18.0)
    path = write_curriculum_yaml(original, tmp_path / "p.curriculum.yaml",
                                 source_pdf="data/checklists/p.pdf",
                                 tier="table_lines")

    back = load_curriculum_yaml(path)

    assert list(back.courses) == list(original.courses)   # order preserved
    assert back.prereq_source is PrereqSource.COLUMN
    assert back.terms_per_year == 3
    assert back.max_units_override == 18.0
    for code, course in original.courses.items():
        assert back.courses[code] == course


def test_written_yaml_has_the_provenance_header_comment(tmp_path):
    """The header is why this artifact is YAML: provenance travels with data."""
    path = write_curriculum_yaml(make_curriculum(MATH), tmp_path / "p.yaml",
                                 source_pdf="data/checklists/p.pdf",
                                 tier="table_text")
    text = path.read_text(encoding="utf-8")

    assert text.startswith("#")
    assert "HAND-EDITABLE" in text
    assert "table_text" in text
    assert "data/checklists/p.pdf" in text.replace("\\", "/")
    assert "PREREQUISITE SOURCE: column" in text
    # The user must be told the planner reads this file and not the PDF.
    assert "NOT THE PDF" in text


def _edit_prereqs_of(path: Path, code: str, replacement: str) -> None:
    """Rewrite exactly one course's `prereqs:` line, the way a user would.

    Anchored on `code: <CODE>` rather than replacing the first match in the
    file: an unanchored replace lands on whichever course happens to be dumped
    first, which makes the test pass for the wrong reason.
    """
    text = path.read_text(encoding="utf-8")
    anchor = f"code: {code}"
    head, found, tail = text.partition(anchor)
    assert found, f"{code} is not in the written file; fixture is wrong"
    assert "prereqs: []" in tail, "nothing to hand-edit; fixture is wrong"
    path.write_text(head + anchor + tail.replace("prereqs: []", replacement, 1),
                    encoding="utf-8")


def test_written_yaml_is_reloadable_after_a_hand_edit(tmp_path):
    """THE test for the escape hatch (AD-8).

    If a hand edit cannot round-trip, the entire "parse once, correct by hand"
    design is a fiction — a mis-parsed prerequisite would be unfixable.
    """
    lonely = Course("CSMATH2", "Discrete Structures", 3, 1, 2)
    path = write_curriculum_yaml(make_curriculum(MATH, lonely),
                                 tmp_path / "p.yaml")

    _edit_prereqs_of(path, "CSMATH2", "prereqs: [GEMATMW]")

    back = load_curriculum_yaml(path)
    assert back.courses["CSMATH2"].prereqs == ("GEMATMW",)
    assert back.courses["GEMATMW"].prereqs == ()   # the other course untouched


def test_hand_edited_scalar_prereq_is_accepted(tmp_path):
    """`prereqs: GEMATMW` is what a human types; punishing it would be hostile."""
    path = write_curriculum_yaml(
        make_curriculum(MATH, Course("CSMATH2", "Discrete Structures", 3)),
        tmp_path / "p.yaml")

    _edit_prereqs_of(path, "CSMATH2", "prereqs: GEMATMW")

    back = load_curriculum_yaml(path)
    assert back.courses["CSMATH2"].prereqs == ("GEMATMW",)


def test_codes_are_uppercased_on_load(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": SCHEMA_VERSION,
        "program": {"id": "x", "name": "X", "terms_per_year": 3},
        "extraction": {"prereq_source": "column"},
        "courses": [
            {"code": "gematmw", "title": "M", "units": 3},
            {"code": "csmath2", "title": "D", "units": 3, "prereqs": ["gematmw"]},
        ],
    }), encoding="utf-8")

    back = load_curriculum_yaml(path)
    assert set(back.courses) == {"GEMATMW", "CSMATH2"}
    assert back.courses["CSMATH2"].prereqs == ("GEMATMW",)


# --- The overwrite guard -------------------------------------------------------

def test_write_refuses_to_overwrite_without_force(tmp_path):
    """Clobbering hand corrections is the one unforgivable bug here (AD-8)."""
    path = tmp_path / "p.yaml"
    write_curriculum_yaml(make_curriculum(MATH), path)

    with pytest.raises(CurriculumError, match="already exists"):
        write_curriculum_yaml(make_curriculum(DISC), path)

    assert "GEMATMW" in path.read_text(encoding="utf-8")   # untouched


def test_write_with_force_overwrites(tmp_path):
    path = tmp_path / "p.yaml"
    write_curriculum_yaml(make_curriculum(MATH), path)
    write_curriculum_yaml(make_curriculum(DISC), path, force=True)

    assert "CSMATH2" in path.read_text(encoding="utf-8")


def test_write_creates_missing_parent_directories(tmp_path):
    path = write_curriculum_yaml(make_curriculum(MATH),
                                 tmp_path / "deep" / "er" / "p.yaml")
    assert path.exists()


# --- Validation ----------------------------------------------------------------

def _write_raw(path: Path, courses: list[dict], **program) -> Path:
    body = {
        "schema_version": program.pop("schema_version", SCHEMA_VERSION),
        "program": {"id": "x", "name": "X", "terms_per_year": 3, **program},
        "extraction": {"prereq_source": "column"},
        "courses": courses,
    }
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(CurriculumError, match="not found"):
        load_curriculum_yaml(tmp_path / "nope.yaml")


def test_load_rejects_unknown_schema_version(tmp_path):
    path = _write_raw(tmp_path / "p.yaml",
                      [{"code": "A1234", "title": "A", "units": 3}],
                      schema_version=99)
    with pytest.raises(CurriculumError, match="schema_version"):
        load_curriculum_yaml(path)


def test_load_rejects_duplicate_codes(tmp_path):
    path = _write_raw(tmp_path / "p.yaml", [
        {"code": "GEMATMW", "title": "A", "units": 3},
        {"code": "GEMATMW", "title": "B", "units": 3},
    ])
    with pytest.raises(CurriculumError, match="duplicate"):
        load_curriculum_yaml(path)


def test_load_rejects_course_without_code(tmp_path):
    path = _write_raw(tmp_path / "p.yaml", [{"title": "A", "units": 3}])
    with pytest.raises(CurriculumError, match="code"):
        load_curriculum_yaml(path)


def test_load_rejects_course_without_units(tmp_path):
    path = _write_raw(tmp_path / "p.yaml", [{"code": "A1234", "title": "A"}])
    with pytest.raises(CurriculumError, match="units"):
        load_curriculum_yaml(path)


def test_load_rejects_implausible_units(tmp_path):
    """A units cell of 178 is a misread total, not a course."""
    path = _write_raw(tmp_path / "p.yaml",
                      [{"code": "A1234", "title": "A", "units": 178}])
    with pytest.raises(CurriculumError, match="units"):
        load_curriculum_yaml(path)


def test_load_rejects_non_numeric_year(tmp_path):
    path = _write_raw(tmp_path / "p.yaml",
                      [{"code": "A1234", "title": "A", "units": 3, "year": "first"}])
    with pytest.raises(CurriculumError, match="year"):
        load_curriculum_yaml(path)


def test_load_rejects_bad_prereq_confidence(tmp_path):
    path = _write_raw(tmp_path / "p.yaml", [
        {"code": "A1234", "title": "A", "units": 3, "prereq_confidence": "vibes"},
    ])
    with pytest.raises(CurriculumError, match="prereq_confidence"):
        load_curriculum_yaml(path)


def test_unresolvable_prereq_becomes_a_warning_not_an_error(tmp_path):
    """A checklist may reference a code from another curriculum version.

    Treating that as fatal would make a correct file unloadable; blocking on it
    forever would hide a course the student can actually take. So: drop the
    edge, record it, and say so.
    """
    path = _write_raw(tmp_path / "p.yaml", [
        {"code": "CSMATH2", "title": "D", "units": 3, "prereqs": ["GONE999"]},
    ])

    back = load_curriculum_yaml(path)

    assert back.courses["CSMATH2"].prereqs == ()
    assert back.unresolved_prereqs == {"CSMATH2": ["GONE999"]}
    assert any("GONE999" in w for w in back.warnings)


def test_unresolvable_coreq_is_also_dropped_and_warned(tmp_path):
    path = _write_raw(tmp_path / "p.yaml", [
        {"code": "LBYCPA1", "title": "Lab", "units": 1, "coreqs": ["GONE999"]},
    ])
    back = load_curriculum_yaml(path)

    assert back.courses["LBYCPA1"].coreqs == ()
    assert "LBYCPA1" in back.unresolved_prereqs


def test_resolvable_prereqs_are_untouched(tmp_path):
    """Regression guard: the unresolved-edge sweep must not drop good edges."""
    path = _write_raw(tmp_path / "p.yaml", [
        {"code": "GEMATMW", "title": "M", "units": 3},
        {"code": "CSMATH2", "title": "D", "units": 3, "prereqs": ["GEMATMW"]},
    ])
    back = load_curriculum_yaml(path)

    assert back.courses["CSMATH2"].prereqs == ("GEMATMW",)
    assert back.unresolved_prereqs == {}


def test_program_max_units_override_is_loaded(tmp_path):
    """§10.2 defers to "the number of units indicated on the program checklist"."""
    path = _write_raw(tmp_path / "p.yaml",
                      [{"code": "A1234", "title": "A", "units": 3}], max_units=18)
    assert load_curriculum_yaml(path).max_units_override == 18.0


def test_absent_max_units_means_use_the_configured_default(tmp_path):
    path = _write_raw(tmp_path / "p.yaml",
                      [{"code": "A1234", "title": "A", "units": 3}])
    assert load_curriculum_yaml(path).max_units_override is None


def test_load_rejects_implausible_terms_per_year(tmp_path):
    path = _write_raw(tmp_path / "p.yaml",
                      [{"code": "A1234", "title": "A", "units": 3}],
                      terms_per_year=17)
    with pytest.raises(CurriculumError, match="terms_per_year"):
        load_curriculum_yaml(path)


def test_absent_prereq_source_defaults_to_none(tmp_path):
    """The safe default: claim nothing about ordering we cannot establish."""
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": SCHEMA_VERSION,
        "program": {"id": "x", "name": "X"},
        "courses": [{"code": "A1234", "title": "A", "units": 3}],
    }), encoding="utf-8")

    assert load_curriculum_yaml(path).prereq_source is PrereqSource.NONE


# --- Per-term unit limits ------------------------------------------------------

def test_term_caps_round_trip(tmp_path):
    original = make_curriculum(MATH, DISC)
    original.term_caps = {1: 17.0, 2: 18.0}
    path = write_curriculum_yaml(original, tmp_path / "p.yaml")

    assert "term_units:" in path.read_text(encoding="utf-8")
    assert load_curriculum_yaml(path).term_caps == {1: 17.0, 2: 18.0}


def test_term_caps_are_derived_when_the_file_omits_them(tmp_path):
    """Files written before term_units existed still get per-term limits."""
    path = _write_raw(tmp_path / "p.yaml", [
        {"code": "AAA111", "title": "A", "units": 3, "year": 1, "term": 1},
        {"code": "BBB222", "title": "B", "units": 2, "year": 1, "term": 1},
        {"code": "CCC333", "title": "C", "units": 3, "year": 1, "term": 2},
    ])
    assert load_curriculum_yaml(path).term_caps == {1: 5.0, 2: 3.0}


def test_hand_edited_term_units_beat_the_derived_ones(tmp_path):
    """The artifact is the contract: a hand edit must win (AD-8)."""
    original = make_curriculum(MATH, DISC)
    original.term_caps = {1: 17.0}
    path = write_curriculum_yaml(original, tmp_path / "p.yaml")
    path.write_text(path.read_text(encoding="utf-8")
                    .replace("1: 17.0", "1: 21.0"), encoding="utf-8")

    assert load_curriculum_yaml(path).term_caps == {1: 21.0}


def test_derive_term_caps_excludes_non_credit_courses():
    """"18 (3)" on the sheet means 18 counted units plus a 3-unit NSTP."""
    courses = [
        Course("AAA111", "A", 15, 1, 1),
        Course("NSTP101", "NSTP", 3, 1, 1, credited=False),
    ]
    assert derive_term_caps(courses, terms_per_year=3) == {1: 15.0}


def test_derive_term_caps_uses_a_running_term_index():
    courses = [Course("AAA111", "A", 3, 1, 1), Course("BBB222", "B", 3, 2, 1)]
    # Year 2 term 1 is the fourth term of a trimester program.
    assert derive_term_caps(courses, terms_per_year=3) == {1: 3.0, 4: 3.0}


def test_term_index_is_none_without_a_placement():
    assert Course("AAA111", "A", 3).term_index(3) is None
    assert Course("AAA111", "A", 3, 2, 2).term_index(3) == 5


def test_credited_units_ignores_non_credit_courses():
    courses = [Course("AAA111", "A", 3), Course("NSTP101", "N", 3, credited=False)]
    assert credited_units(courses) == 3.0
    assert total_units(courses) == 6.0


def test_credited_defaults_to_true_when_the_file_omits_it(tmp_path):
    path = _write_raw(tmp_path / "p.yaml",
                      [{"code": "AAA111", "title": "A", "units": 3}])
    assert load_curriculum_yaml(path).courses["AAA111"].credited is True


def test_load_rejects_non_numeric_term_units(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": SCHEMA_VERSION,
        "program": {"id": "x", "name": "X", "terms_per_year": 3},
        "term_units": {1: "heavy"},
        "courses": [{"code": "AAA111", "title": "A", "units": 3}],
    }), encoding="utf-8")

    with pytest.raises(CurriculumError, match="term_units"):
        load_curriculum_yaml(path)


# --- Derived helpers -----------------------------------------------------------

def test_taken_codes_excludes_failed_courses():
    """`taken` is the planning input, and a 0.0 must never satisfy a prereq.

    Regression for the whole point of grade parsing: crediting a failure would
    send a student into a course they are actually blocked from.
    """
    failed = Course("CSMATH2", "Discrete Structures", 3, taken=False, grade="0.0")
    curriculum = make_curriculum(MATH, failed)

    assert taken_codes(curriculum) == {"GEMATMW"}


def test_total_units_sums_and_rounds():
    assert total_units([MATH, LAB]) == 4.0
    assert total_units([]) == 0


def test_remaining_courses_keeps_checklist_order():
    curriculum = make_curriculum(MATH, DISC, LAB)
    assert [c.code for c in remaining_courses(curriculum)] == ["CSMATH2", "LBYCPA1"]


def test_checklist_order_sorts_unknown_years_last():
    known = Course("A1234", "A", 3, 1, 2)
    unknown = Course("B1234", "B", 3)

    assert known.checklist_order == (1, 2)
    assert unknown.checklist_order > known.checklist_order


def test_course_is_hashable_and_immutable():
    """Frozen so a plan cannot mutate its own inputs mid-run."""
    assert len({MATH, DISC}) == 2
    with pytest.raises(Exception):
        MATH.units = 99
