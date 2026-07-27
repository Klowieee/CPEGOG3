"""
mermaid.py — Render a StudyPlan as a Mermaid flowchart inside Markdown.

Purpose:
    Turn a finished StudyPlan into a diagram a student (or a panelist) can
    read at a glance: one subgraph per planned term, prerequisite arrows
    between the courses, and a colour per status — including a distinct,
    dashed colour for courses whose prerequisites are UNKNOWN.

Inputs:
    A StudyPlan already built by the planner, the Curriculum it was built
    from, and optionally the policy rules the planner applied (duck-typed
    objects exposing .statement / .citation / .excerpt).

Outputs:
    render_mermaid() returns the `flowchart` body; render_plan_markdown()
    wraps it in a self-contained Markdown document; write_plan_markdown()
    persists that document as UTF-8 (mirroring chunker.write_chunks_jsonl).

Dependencies:
    src.curriculum.model only (re/pathlib from the standard library). This
    module deliberately does NOT import src.curriculum.planner: it receives
    a plan, it does not make one.

Why this file exists:
    Architectural Decision AD-7 keeps course ordering in deterministic code
    rather than in the LLM — but a deterministic answer is only trustworthy
    if it can be audited, and a wall of course codes cannot be. The diagram
    is that audit surface: the student sees the actual prerequisite edges the
    planner used and can spot a wrong one immediately. Two consequences shape
    the whole module:

      * Uncertainty is drawn, not dropped. A course with
        PrereqConfidence.UNKNOWN gets its own dashed style, because a plan
        that hides what it had to guess is worse than one that admits it.
      * Output is text, and text can be syntactically invalid. Mermaid fails
        by rendering NOTHING, so every structural rule here (unconditional
        "C_" id prefix, label escaping, edges only between declared nodes,
        no set iteration) exists to make an unparseable diagram impossible
        rather than merely unlikely.
"""

from __future__ import annotations

import re

from pathlib import Path

from src.curriculum.model import (
    Course,
    Curriculum,
    PlannedTerm,
    PrereqConfidence,
    StudyPlan,
    total_units,
)

# Mermaid's own set of flowchart orientations. Anything else is a typo, and a
# typo here kills the whole diagram, so it is corrected rather than emitted.
VALID_DIRECTIONS = ("TB", "TD", "BT", "LR", "RL")
DEFAULT_DIRECTION = "LR"

# Styles keyed by status, in the order they are emitted. Kept as data so the
# classDef block and the class assignments cannot drift apart.
CLASS_DEFS: tuple[tuple[str, str], ...] = (
    ("taken", "fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20"),
    ("ready", "fill:#e3f2fd,stroke:#1565c0,color:#0d47a1,stroke-width:2px"),
    ("later", "fill:#f5f5f5,stroke:#9e9e9e,color:#424242"),
    ("unknown", "fill:#fff8e1,stroke:#f9a825,color:#e65100,stroke-dasharray:4 3"),
)

_ID_UNSAFE = re.compile(r"[^A-Za-z0-9_]")
_WHITESPACE = re.compile(r"\s+")

# Brackets and braces close a node shape early; '%' starts a Mermaid comment;
# a backtick can close the ```mermaid fence render_plan_markdown wraps this in,
# which ends the diagram and swallows the rest of the document. None of them can
# be escaped inside a quoted label, so they are removed.
_STRIP_TABLE = {ord(ch): None for ch in "[](){}%`"}

_INDENT = "    "

# Shown when there is literally nothing to draw. `flowchart LR` with no nodes
# is not a diagram, and an empty render must still be a valid one.
_PLACEHOLDER_LABEL = "Nothing to plan"


# --- Identifiers and labels ----------------------------------------------------

def node_id(code: str) -> str:
    """Mermaid-safe node id for a course code.

    Uppercases, replaces every character Mermaid will not accept in an id
    with '_', and prefixes "C_" UNCONDITIONALLY.

    The prefix is not conditional on purpose. Mermaid ids may not contain
    spaces or hyphens ("NSTP-CWTS 1" would parse as three tokens), and a
    handful of bare words — end, graph, class, subgraph, o, x, style, click —
    are reserved and break the parser SILENTLY when used as ids. Prefixing
    always is one line shorter than maintaining a reserved-word list, and it
    cannot be defeated by a word we failed to think of.

        "NSTP-CWTS 1" -> "C_NSTP_CWTS_1"
        "END"         -> "C_END"
    """
    return "C_" + _ID_UNSAFE.sub("_", (code or "").upper())


def escape_label(text: str) -> str:
    """Make arbitrary text safe inside a quoted Mermaid label.

    A quoted label tolerates far less than it looks like it should: an
    unescaped double quote ends the label, '<' and '>' are read as markup,
    brackets/braces close the node shape, '%' begins a comment, and a backtick
    can close the surrounding Markdown fence. Quotes and angle brackets become
    HTML entities; the rest are removed, since there is no escape for them.
    Newlines collapse to single spaces so a multi-line title cannot split one
    declaration across two lines.

    A bare '&' is deliberately left alone: it separates node lists only OUTSIDE
    a quoted string, and browsers render a stray '&' in an HTML label literally.
    Anyone adding '&' -> '&amp;' must do it BEFORE the substitutions below, or
    the entities they emit get escaped a second time into '&amp;quot;'.
    """
    cleaned = (text or "")
    cleaned = cleaned.replace('"', "&quot;")
    cleaned = cleaned.replace("<", "&lt;").replace(">", "&gt;")
    cleaned = cleaned.translate(_STRIP_TABLE)
    return _WHITESPACE.sub(" ", cleaned).strip()


def _node_line(course: Course) -> str:
    """One node declaration: C_CODE["CODE<br/>Title<br/>3u"].

    <br/> rather than a literal newline is this repo's Mermaid idiom (see the
    diagrams in docs/architecture.md): it is what renders on GitHub.
    """
    label = (f"{escape_label(course.code)}<br/>"
             f"{escape_label(course.title)}<br/>"
             f"{course.units:g}u")
    return f'{node_id(course.code)}["{label}"]'


def _status_of(course: Course, term_index: int) -> str:
    """Which classDef a course belongs to.

    Precedence is taken > unknown > ready > later. `unknown` outranks the
    scheduling statuses because the point of the colour is to warn that the
    course's position was guessed — a confident blue "take this next term"
    on a course with no known prerequisites would be a lie.
    """
    if course.taken or term_index == 0:
        return "taken"
    if course.confidence is PrereqConfidence.UNKNOWN:
        return "unknown"
    return "ready" if term_index == 1 else "later"


# --- Flowchart -----------------------------------------------------------------

def _taken_set(curriculum: Curriculum,
               taken: set[str] | frozenset[str] | None) -> set[str]:
    """Which codes count as passed: the caller's set, else the artifact's flags.

    A checklist with no grade column carries no flags at all — the student
    supplies them at the prompt — so a diagram that trusted only the flags would
    show an empty "Already taken" box for a graduating senior.
    """
    if taken is not None:
        return {str(c).upper() for c in taken}
    return {c.code for c in curriculum.courses.values() if c.taken}


def render_mermaid(plan: StudyPlan, curriculum: Curriculum, *,
                   direction: str = "LR", include_taken: bool = True,
                   taken: set[str] | frozenset[str] | None = None) -> str:
    """Render the plan as the body of a Mermaid `flowchart`.

    Args:
        plan: An already-built StudyPlan. Never modified.
        curriculum: The curriculum it was planned from; supplies the
            prerequisite edges.
        direction: One of VALID_DIRECTIONS; anything else falls back to "LR".
        include_taken: Draw the "Already taken" subgraph and the edges that
            start in it. False gives a diagram of the work that remains.
        taken: Codes to treat as passed. None falls back to the curriculum's
            own `taken` flags.

    Returns:
        The flowchart text, no trailing newline and no ``` fence.
    """
    direction = str(direction or "").strip().upper()
    if direction not in VALID_DIRECTIONS:
        direction = DEFAULT_DIRECTION

    lines: list[str] = [f"flowchart {direction}"]

    # node id -> the Course drawn for it: the single source of truth for "is
    # this node declared?", which the edge pass depends on. Keyed by the ID and
    # not by the code, because the id is what the output actually contains.
    # Keying by code let two spellings of one course ("MATH1" declared, "math1"
    # named as a prereq) disagree about whether a node exists, dropping the edge
    # between them, and let two codes that sanitize to one id both be declared.
    declared: dict[str, Course] = {}
    by_status: dict[str, list[str]] = {name: [] for name, _ in CLASS_DEFS}

    passed = _taken_set(curriculum, taken)
    taken_courses = [c for c in curriculum.courses.values()
                     if c.code in passed]
    if include_taken and taken_courses:
        lines.extend(_taken_subgraph(taken_courses, declared, by_status))
    for term in plan.terms:
        lines.extend(_term_subgraph(term, declared, by_status))

    if not declared:
        lines.append(f'{node_id("EMPTY")}["{escape_label(_PLACEHOLDER_LABEL)}"]')

    # Edges come after every subgraph: an edge mentioned earlier would pull
    # its endpoints into whichever subgraph was open at the time.
    lines.extend(_edge_lines(declared))

    for name, style in CLASS_DEFS:
        lines.append(f"classDef {name} {style}")
    for name, _ in CLASS_DEFS:
        members = by_status[name]
        if members:
            lines.append(f"class {','.join(members)} {name}")

    return "\n".join(lines)


def _taken_subgraph(taken: list[Course], declared: dict[str, Course],
                    by_status: dict[str, list[str]]) -> list[str]:
    """The T0 subgraph of completed courses, placed before every planned term.

    It exists so the prerequisite arrows into term 1 have a visible origin;
    without it the first term looks like it came from nowhere.
    """
    label = (f"Already taken — {len(taken)} "
             f"course{'' if len(taken) == 1 else 's'}, {total_units(taken):g}u")
    out = [f'subgraph T0["{escape_label(label)}"]']
    for course in taken:
        nid = node_id(course.code)
        if nid in declared:
            continue                      # another code sanitized to this id
        declared[nid] = course
        by_status["taken"].append(nid)
        out.append(_INDENT + _node_line(course))
    out.append("end")
    return out


def _term_subgraph(term: PlannedTerm, declared: dict[str, Course],
                   by_status: dict[str, list[str]]) -> list[str]:
    """One subgraph per planned term, labelled with its own unit total."""
    # Show the limit that governed this term, not a global one: the checklist
    # prescribes its own per-term load, and "15 of 19u" would misstate it.
    label = (f"{term.label} — {term.units:g} of {term.cap:g}u" if term.cap
             else f"{term.label} — {term.units:g}u")
    out = [f'subgraph T{term.index}["{escape_label(label)}"]']
    for course in term.courses:
        nid = node_id(course.code)
        if nid in declared:
            continue                      # already drawn (e.g. also flagged taken)
        declared[nid] = course
        by_status[_status_of(course, term.index)].append(nid)
        out.append(_INDENT + _node_line(course))
    out.append("end")
    return out


def _edge_lines(declared: dict[str, Course]) -> list[str]:
    """Prerequisite (solid) then corequisite (dotted) edges.

    Two rules keep the output parseable:

      * An edge is emitted only when BOTH endpoints are declared nodes.
        Mermaid would happily invent a node for an unknown id, producing a
        bare, unstyled box for a course that is not in the diagram — which is
        exactly what include_taken=False must not do.
      * A corequisite pair yields exactly ONE edge. Coreqs are usually stated
        on both courses, so the pair is ordered and only smaller->larger is
        emitted; otherwise the pair renders as a double-headed arrow that
        reads like a prerequisite cycle.

    Both rules are applied in node-id space, not code space: `declared` is
    keyed by id, and a referenced code is resolved through node_id() before it
    is looked up. Deciding "is it declared?" on the raw code while emitting the
    sanitized id meant the two could disagree.
    """
    prereq_edges: list[str] = []
    coreq_edges: list[str] = []
    seen: set[str] = set()               # membership only; never iterated

    for nid, course in declared.items():
        for source in course.prereqs:
            source_id = node_id(source)
            if source_id not in declared:
                continue
            edge = f"{source_id} --> {nid}"
            if edge not in seen:
                seen.add(edge)
                prereq_edges.append(edge)
        for other in course.coreqs:
            other_id = node_id(other)
            if other_id not in declared or other_id == nid:
                continue
            low, high = sorted((nid, other_id))
            edge = f"{low} -.-> {high}"
            if edge not in seen:
                seen.add(edge)
                coreq_edges.append(edge)

    return prereq_edges + coreq_edges


# --- Markdown document ---------------------------------------------------------

def render_plan_markdown(plan: StudyPlan, curriculum: Curriculum,
                         policy: list | None = None, *,
                         direction: str = "LR",
                         include_taken: bool = True,
                         taken: set[str] | frozenset[str] | None = None) -> str:
    """Wrap the flowchart in a self-contained Markdown document.

    The file is meant to be committed and read as-is: a fenced ```mermaid
    block renders natively on GitHub and in VS Code, so the diagram needs no
    build step and no image asset.

    Args:
        policy: Rules the planner applied, duck-typed on .statement and
            .citation (.excerpt is available but deliberately not rendered —
            the section is a summary of what constrained the plan, not the
            handbook text). Empty or None omits the section entirely, because
            an empty "Constraints applied" heading implies we looked and found
            nothing, which is a different claim.
    """
    passed = _taken_set(curriculum, taken)
    taken_courses = [c for c in curriculum.courses.values() if c.code in passed]
    remaining = [c for c in curriculum.courses.values()
                 if c.code not in passed]
    term_count = len(plan.terms)

    out: list[str] = [
        f"# {curriculum.program_name} — study plan",
        "",
        f"**{total_units(taken_courses):g}u taken** · "
        f"**{total_units(remaining):g}u remaining** · "
        f"**{term_count} term{'' if term_count == 1 else 's'} planned**",
        "",
        "```mermaid",
        render_mermaid(plan, curriculum, direction=direction, taken=passed,
                       include_taken=include_taken),
        "```",
    ]

    if policy:
        out += ["", "## Constraints applied", ""]
        for rule in policy:
            statement = str(getattr(rule, "statement", "") or "").strip()
            if not statement:
                continue
            citation = getattr(rule, "citation", None)
            out.append(f"- {statement} — {citation}" if citation
                       else f"- {statement}")

    if plan.notes:
        out += ["", "## Caveats", ""]
        out += [f"- {note}" for note in plan.notes]

    return "\n".join(out) + "\n"


def write_plan_markdown(text: str, path: Path | str) -> Path:
    """Persist a rendered plan as UTF-8; returns the path written.

    Mirrors chunker.write_chunks_jsonl: create the parent directory, write
    the artifact, hand back the path so a caller can print it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
