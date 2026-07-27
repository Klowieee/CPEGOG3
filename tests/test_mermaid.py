"""
test_mermaid.py — Tests for src/curriculum/mermaid.py.

StudyPlan/PlannedTerm/Course are constructed by hand; the planner is never
involved, so these tests describe the RENDERER's contract only.

The emphasis is on the ways Mermaid fails silently — a reserved word or a
hyphen in an id, a bracket in a title, an arrow pointing at a node nobody
declared — because every one of them renders as a blank diagram rather than
an error message.

Dependencies:
    pytest, src.curriculum.mermaid, src.curriculum.model.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.curriculum.model import (                              # noqa: E402
    Course,
    Curriculum,
    PlannedTerm,
    PrereqConfidence,
    PrereqSource,
    StudyPlan,
    total_units,
)
from src.curriculum.mermaid import (                            # noqa: E402
    escape_label,
    node_id,
    render_mermaid,
    render_plan_markdown,
    write_plan_markdown,
)

DECLARATION = re.compile(r'^\s*([A-Za-z0-9_]+)\["')
# Every id token on an edge line, NOT just the C_-prefixed ones. Matching only
# /C_\w+/ would presuppose the prefix the invariant is there to verify: a bare
# `MATH1 --> C_MATH2`, or an edge pointing at a subgraph id (`T1 --> T2`),
# would contribute nothing to `referenced` and the subset assertion would pass
# on a diagram that is actually broken.
NODE_REF = re.compile(r"[A-Za-z0-9_]+")
EDGE_ARROW = re.compile(r"-\.->|-->")


# --- Fixtures ------------------------------------------------------------------

def course(code, title="Some Course", units=3.0, *, prereqs=(), coreqs=(),
           confidence=PrereqConfidence.STATED, taken=False) -> Course:
    return Course(code=code, title=title, units=units,
                  prereqs=tuple(prereqs), coreqs=tuple(coreqs),
                  confidence=confidence, taken=taken)


def curriculum_of(*courses, name="BS Computer Science") -> Curriculum:
    return Curriculum(
        program_id="bscs",
        program_name=name,
        terms_per_year=3,
        courses={c.code: c for c in courses},
        prereq_source=PrereqSource.COLUMN,
    )


def plan_of(*term_courses, notes=()) -> StudyPlan:
    """One PlannedTerm per argument; each argument is a list of Courses."""
    terms = [
        PlannedTerm(
            index=i + 1,
            label="Next term" if i == 0 else f"Term +{i + 1}",
            courses=list(courses),
            units=total_units(courses),
        )
        for i, courses in enumerate(term_courses)
    ]
    return StudyPlan(terms=terms, available_now=[], deferred=[], blocked=[],
                     unreachable=[], cycles=[], notes=list(notes))


def declared_ids(text: str) -> list[str]:
    return [m.group(1) for m in
            (DECLARATION.match(line) for line in text.splitlines()) if m]


class FakeRule:
    """Stands in for a policy rule; duck-typed, no import of src.curriculum.policy."""

    def __init__(self, statement, citation=None, excerpt=""):
        self.statement = statement
        self.citation = citation
        self.excerpt = excerpt


# --- Header and direction ------------------------------------------------------

def test_output_starts_with_flowchart_direction():
    curriculum = curriculum_of(course("GEMATMW"))
    text = render_mermaid(plan_of([course("GEMATMW")]), curriculum,
                          direction="TB")
    assert text.splitlines()[0] == "flowchart TB"


def test_invalid_direction_falls_back_to_lr():
    # A bad direction must not reach the output: Mermaid answers an
    # unparseable first line with a blank diagram, not a complaint.
    curriculum = curriculum_of(course("GEMATMW"))
    text = render_mermaid(plan_of([course("GEMATMW")]), curriculum,
                          direction="sideways")
    assert text.splitlines()[0] == "flowchart LR"


def test_empty_plan_still_renders_a_valid_header():
    text = render_mermaid(plan_of(), curriculum_of())
    assert text.splitlines()[0] == "flowchart LR"
    assert "subgraph" not in text
    assert "classDef taken" in text
    # A header alone is not a diagram, so an empty plan declares one node.
    assert declared_ids(text) == ["C_EMPTY"]


# --- Subgraph structure --------------------------------------------------------

def test_one_subgraph_per_term_plus_taken():
    curriculum = curriculum_of(
        course("GEMATMW", taken=True),
        course("CCPROG1"), course("CCPROG2"),
    )
    text = render_mermaid(plan_of([course("CCPROG1")], [course("CCPROG2")]),
                          curriculum)
    heads = [ln for ln in text.splitlines() if ln.startswith("subgraph")]
    assert len(heads) == 3
    assert heads[0].startswith('subgraph T0["Already taken — 1 course, 3u"]')
    assert heads[1].startswith('subgraph T1["Next term — 3u"]')
    assert heads[2].startswith('subgraph T2["Term +2 — 3u"]')
    assert len([ln for ln in text.splitlines() if ln.strip() == "end"]) == 3


def test_include_taken_false_omits_that_subgraph():
    curriculum = curriculum_of(course("GEMATMW", taken=True), course("CCPROG1"))
    text = render_mermaid(plan_of([course("CCPROG1")]), curriculum,
                          include_taken=False)
    assert "T0[" not in text
    assert "Already taken" not in text
    assert "C_GEMATMW" not in text
    assert 'subgraph T1["Next term — 3u"]' in text


# --- Edges ---------------------------------------------------------------------

def test_prereq_edge_is_solid_and_coreq_edge_is_dotted():
    lab = course("PHYLAB1", coreqs=["PHYSC1"])
    lecture = course("PHYSC1")
    math1 = course("MATH1")
    math2 = course("MATH2", prereqs=["MATH1"])
    curriculum = curriculum_of(lab, lecture, math1, math2)
    text = render_mermaid(plan_of([lab, lecture, math1], [math2]), curriculum)

    assert "C_MATH1 --> C_MATH2" in text
    assert "C_PHYLAB1 -.-> C_PHYSC1" in text
    # Edges live after the last subgraph, otherwise they join whichever
    # subgraph is still open and get drawn inside its box.
    assert text.index("end") < text.index("C_MATH1 --> C_MATH2")


def test_coreq_pair_emits_exactly_one_edge():
    # A coreq is normally stated on BOTH courses. Emitting each direction
    # gives a double-headed arrow that reads as a prerequisite cycle.
    lab = course("PHYLAB1", coreqs=["PHYSC1"])
    lecture = course("PHYSC1", coreqs=["PHYLAB1"])
    text = render_mermaid(plan_of([lab, lecture]), curriculum_of(lab, lecture))
    assert text.count("-.->") == 1
    assert "C_PHYLAB1 -.-> C_PHYSC1" in text


def test_include_taken_false_drops_edges_from_undeclared_taken_courses():
    # Mermaid auto-creates a node for an unknown id, so an edge from a course
    # we chose not to draw would silently put it back — unstyled and outside
    # every subgraph.
    done = course("MATH1", taken=True)
    next_up = course("MATH2", prereqs=["MATH1"])
    curriculum = curriculum_of(done, next_up)

    with_taken = render_mermaid(plan_of([next_up]), curriculum)
    assert "C_MATH1 --> C_MATH2" in with_taken

    without = render_mermaid(plan_of([next_up]), curriculum, include_taken=False)
    assert "-->" not in without
    assert "C_MATH1" not in without


def test_every_node_referenced_by_an_edge_is_declared():
    # THE structural invariant. A reference to an id that was never declared
    # is the single most common way this output breaks, and it breaks by
    # rendering a stray box (or nothing at all) rather than by raising.
    done = course("MATH1", taken=True)
    curriculum = curriculum_of(
        done,
        course("MATH2", prereqs=["MATH1"]),
        course("PHYLAB1", coreqs=["PHYSC1"]),
        course("PHYSC1", coreqs=["PHYLAB1"]),
        course("CSALGCM", prereqs=["MATH2", "PHYSC1"]),
    )
    plan = plan_of(
        [curriculum.courses["MATH2"], curriculum.courses["PHYLAB1"],
         curriculum.courses["PHYSC1"]],
        [curriculum.courses["CSALGCM"]],
    )

    for include_taken in (True, False):
        text = render_mermaid(plan, curriculum, include_taken=include_taken)
        declared = set(declared_ids(text))
        referenced = {
            ref
            for line in text.splitlines() if EDGE_ARROW.search(line)
            for endpoint in EDGE_ARROW.split(line)
            for ref in NODE_REF.findall(endpoint)
        }
        assert referenced, "the fixture must produce at least one edge"
        assert referenced <= declared, (
            f"undeclared ids in edges: {sorted(referenced - declared)}")


def test_edge_survives_a_prereq_code_spelled_in_another_case():
    # node_id() normalizes case and punctuation, so "math1" and "MATH1" are the
    # SAME node. Deciding "is this endpoint declared?" on the raw code instead
    # of the id dropped the edge: the diagram then shows MATH2 with no incoming
    # arrow, i.e. announces it has no prerequisite.
    math1 = course("MATH1")
    math2 = course("MATH2", prereqs=["math1"])
    text = render_mermaid(plan_of([math1], [math2]), curriculum_of(math1, math2))
    assert "C_MATH1 --> C_MATH2" in text

    # Same for a coreq, and it must still be exactly one edge.
    lab = course("PHYLAB1", coreqs=["physc1"])
    lec = course("PHYSC1", coreqs=["PHYLAB1"])
    text = render_mermaid(plan_of([lab, lec]), curriculum_of(lab, lec))
    assert "C_PHYLAB1 -.-> C_PHYSC1" in text and text.count("-.->") == 1


def test_two_codes_that_sanitize_to_one_id_are_declared_once():
    # "MATH-1" and "MATH 1" both sanitize to C_MATH_1. Declaring both emits
    # that id twice — Mermaid keeps whichever label came last, and the class
    # line lists the id twice.
    dash = course("MATH-1", title="Dash")
    space = course("MATH 1", title="Space")
    text = render_mermaid(plan_of([dash, space]), curriculum_of(dash, space))
    ids = declared_ids(text)
    assert ids == ["C_MATH_1"]
    members = [ln.split(" ")[1] for ln in text.splitlines()
               if ln.startswith("class ")]
    assert members == ["C_MATH_1"]


# --- Identifiers and labels ----------------------------------------------------

def test_node_ids_are_sanitized():
    assert node_id("NSTP-CWTS 1") == "C_NSTP_CWTS_1"
    nstp = course("NSTP-CWTS 1", units=0.0)
    text = render_mermaid(plan_of([nstp]), curriculum_of(nstp))
    ids = declared_ids(text)
    assert ids == ["C_NSTP_CWTS_1"]
    assert all(" " not in i and "-" not in i for i in ids)


def test_reserved_word_code_does_not_break_ids():
    # "end" closes a subgraph. A course coded END declared as a bare `end`
    # would terminate the subgraph early and swallow the rest of the diagram.
    assert node_id("END") == "C_END"
    reserved = course("END", title="Capstone Endgame")
    text = render_mermaid(plan_of([reserved]), curriculum_of(reserved))
    lines = text.splitlines()
    assert declared_ids(text) == ["C_END"]
    assert not [ln for ln in lines if ln.strip().lower().startswith("end[")]
    # The only bare `end` is the one subgraph's terminator.
    assert len([ln for ln in lines if ln.strip() == "end"]) == 1


def test_title_brackets_and_quotes_are_escaped():
    assert escape_label('The "Big" [Idea]') == "The &quot;Big&quot; Idea"
    assert escape_label("a\nb\n c") == "a b c"
    assert escape_label("<script>") == "&lt;script&gt;"

    messy = course("GEETHIC", title='Ethics [Sec. 3] "Honor" (Lab) <b> 50% {x}')
    text = render_mermaid(plan_of([messy]), curriculum_of(messy))
    label = next(ln for ln in text.splitlines() if ln.strip().startswith("C_"))
    assert "&quot;Honor&quot;" in label
    assert "&lt;b&gt;" in label
    assert "[Sec. 3]" not in label and "(Lab)" not in label
    assert "{x}" not in label and "%" not in label
    # The shape brackets Mermaid needs must survive exactly once each.
    assert label.count("[") == 1 and label.count("]") == 1


# --- Styling -------------------------------------------------------------------

def test_class_assignment_per_status():
    done = course("MATH1", taken=True)
    ready = course("MATH2", prereqs=["MATH1"])
    later = course("MATH3", prereqs=["MATH2"])
    fuzzy = course("GEELECT", confidence=PrereqConfidence.UNKNOWN)
    curriculum = curriculum_of(done, ready, later, fuzzy)
    text = render_mermaid(plan_of([ready], [later, fuzzy]), curriculum)

    for name in ("taken", "ready", "later", "unknown"):
        assert f"classDef {name} fill:#" in text
    assert "stroke-dasharray:4 3" in text          # unknown must LOOK uncertain

    # "class C_MATH1,C_MATH2 later" -> {"later": "C_MATH1,C_MATH2"}
    assignments = {}
    for line in text.splitlines():
        if line.startswith("class "):
            _, members, status = line.split(" ")
            assignments[status] = members
    assert assignments["taken"] == "C_MATH1"
    assert assignments["ready"] == "C_MATH2"        # term 1 only
    assert assignments["later"] == "C_MATH3"
    assert assignments["unknown"] == "C_GEELECT"


def test_unknown_outranks_ready_for_a_term_one_course():
    # The precedence claim that test_class_assignment_per_status does NOT pin:
    # its UNKNOWN fixture sits in term 2, where the competing status is `later`,
    # so swapping the `unknown`/`ready` checks leaves that test green. A term-1
    # course is the only place the two statuses actually compete. Blue means
    # "take this next term"; on a course whose position was guessed, it lies.
    fuzzy = course("GEELECT", confidence=PrereqConfidence.UNKNOWN)
    text = render_mermaid(plan_of([fuzzy]), curriculum_of(fuzzy))
    assert "class C_GEELECT unknown" in text
    assert "ready" not in [ln.split(" ")[-1] for ln in text.splitlines()
                           if ln.startswith("class ")]


def test_class_line_is_omitted_for_a_status_with_no_members():
    solo = course("CCPROG1")
    text = render_mermaid(plan_of([solo]), curriculum_of(solo))
    assert "class C_CCPROG1 ready" in text
    assert not [ln for ln in text.splitlines() if ln.startswith("class ")
                and ln.endswith((" taken", " later", " unknown"))]


# --- Markdown wrapper and persistence ------------------------------------------

def test_markdown_wrapper_contains_the_policy_citations():
    done = course("MATH1", taken=True)
    todo = course("MATH2", units=3.0, prereqs=["MATH1"])
    curriculum = curriculum_of(done, todo)
    policy = [
        FakeRule("A regular load is at most 18 units.",
                 "Undergraduate, Section 10.2, p. 101", "excerpt text"),
        FakeRule("Corequisites are taken in the same term."),
    ]
    plan = plan_of([todo], notes=["Prereqs were inferred."])
    md = render_plan_markdown(plan, curriculum, policy)

    assert md.startswith("# BS Computer Science — study plan\n")
    assert "**3u taken** · **3u remaining** · **1 term planned**" in md
    assert "```mermaid\nflowchart LR" in md and md.count("```") == 2
    assert "## Constraints applied" in md
    assert ("- A regular load is at most 18 units. — "
            "Undergraduate, Section 10.2, p. 101") in md
    assert "- Corequisites are taken in the same term." in md
    assert "## Caveats" in md and "- Prereqs were inferred." in md


def test_markdown_omits_constraints_section_when_policy_is_empty():
    todo = course("MATH1")
    curriculum = curriculum_of(todo)
    for policy in (None, []):
        md = render_plan_markdown(plan_of([todo]), curriculum, policy)
        assert "## Constraints applied" not in md
        assert "```mermaid" in md
    # No notes either -> no Caveats heading to explain.
    assert "## Caveats" not in render_plan_markdown(plan_of([todo]), curriculum)


def test_a_backtick_in_a_title_cannot_break_out_of_the_mermaid_fence():
    # The one character that escapes the WRAPPER rather than the label. Three of
    # them close the ```mermaid block early, so the rest of the diagram renders
    # as prose and the document below it is swallowed into a code block.
    sneaky = course("CCPROG1", title="Uses ```mermaid in class")
    md = render_plan_markdown(plan_of([sneaky]), curriculum_of(sneaky))
    assert md.count("```") == 2
    fences = [i for i, ln in enumerate(md.splitlines()) if ln.startswith("```")]
    assert len(fences) == 2 and md.splitlines()[fences[0]] == "```mermaid"
    assert "`" not in escape_label("a`b")


def test_write_plan_markdown_creates_parents_and_returns_path(tmp_path):
    todo = course("MATH1")
    md = render_plan_markdown(plan_of([todo]), curriculum_of(todo))
    out = tmp_path / "reports" / "plans" / "bscs.plan.md"
    returned = write_plan_markdown(md, out)
    assert returned == out and out.exists()
    # UTF-8: the em dash in the heading and the subgraph labels must survive.
    assert out.read_text(encoding="utf-8") == md
    assert "— study plan" in out.read_text(encoding="utf-8")
