"""
planner.py — Turn a curriculum plus "what I've passed" into an ordered plan.

Purpose:
    Decide which courses are eligible now and lay the rest out term by term,
    respecting prerequisites, corequisites, and the unit cap. Pure computation:
    same inputs always produce the same plan, byte for byte.

Inputs:
    A Curriculum (from the hand-editable artifact), the set of passed course
    codes, and the numeric limits from PlannerSettings.

Outputs:
    A StudyPlan carrying the schedule AND everything it could not schedule
    (deferred, blocked, unreachable, cycles, notes).

Dependencies:
    graphlib (standard library), src.curriculum.model. No LLM, no network.

Why this file exists:
    Architectural Decision AD-7: course ordering is deterministic code and the
    LLM is not in the loop. A prerequisite graph has an exact answer, so
    generating one would add risk for nothing, and a confidently wrong schedule
    is worse than no schedule. Everything here is therefore reproducible and
    auditable — see docs/course_planner.md §5, and §8 for the determinism
    guarantees this module is required to hold.

    The other rule this module obeys: build_plan NEVER raises for a data
    reason. A cyclic graph, an over-cap corequisite pair, an empty checklist —
    each becomes an entry in the returned plan rather than an exception, the
    same way Answer(error=...) keeps src/chat/core.py's UI always able to show
    something honest.
"""

from __future__ import annotations

import logging

from graphlib import TopologicalSorter

from src.curriculum.model import (
    Bundle,
    Course,
    Curriculum,
    PlannedTerm,
    PrereqConfidence,
    PrereqSource,
    StudyPlan,
    attempted_codes,
    credited_units,
    taken_codes,
)

log = logging.getLogger(__name__)

# Undergraduate §10.2 permits a graduating student an overload of up to this
# many units. The planner never applies it — it only says so in a note.
GRADUATING_OVERLOAD_UNITS = 6.0

_LAB_CODE_PREFIX = "LBY"
_LAB_TITLE_SUFFIX = "laboratory"


# --- Graph primitives ----------------------------------------------------------

def find_cycles(edges: dict[str, set[str]]) -> list[list[str]]:
    """Every prerequisite cycle in `edges`, canonicalized for stable output.

    `edges` maps a course to the set of courses it DEPENDS ON. Returns each
    strongly connected component of size > 1, plus every self-loop, with each
    cycle's members sorted and the outer list sorted — so two runs on the same
    data produce identical output.

    Iterative Tarjan rather than recursion: a long prerequisite chain would
    otherwise risk the recursion limit.

    Rejected: TopologicalSorter.prepare() and catching CycleError. That is
    shorter, but it surfaces one cycle at a time as a DFS *path* rather than a
    canonical component, so the members reported look arbitrary and the
    "delete the backward edge" rule loses its meaning. Reporting every cycle at
    once is exactly the diagnostic value this feature needs.
    """
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    found: list[list[str]] = []
    counter = 0

    def successors(node: str):
        # Sorted, and restricted to known nodes, so traversal order is fixed.
        return iter(sorted(d for d in edges.get(node, ()) if d in edges))

    for root in sorted(edges):
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        work: list[tuple[str, object]] = [(root, successors(root))]

        while work:
            node, iterator = work[-1]
            descended = False
            for dep in iterator:                      # type: ignore[union-attr]
                if dep not in index:
                    index[dep] = low[dep] = counter
                    counter += 1
                    stack.append(dep)
                    on_stack[dep] = True
                    work.append((dep, successors(dep)))
                    descended = True
                    break
                if on_stack.get(dep):
                    low[node] = min(low[node], index[dep])
            if descended:
                continue

            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])

            if low[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    found.append(sorted(component))
                elif node in edges.get(node, ()):
                    found.append([node])              # a self-loop is a cycle

    return sorted(found)


def break_cycles(edges: dict[str, set[str]], cycles: list[list[str]],
                 order_key) -> list[tuple[str, str]]:
    """Make `edges` acyclic in place; return the (course, prereq) pairs removed.

    Within each cycle, members are sorted by `order_key` and every prerequisite
    edge pointing BACKWARD in that order is deleted. When the checklist states
    year/term, that order is the program's own intended sequence, so the repair
    preserves its intent; when it does not, the alphabetical fallback at least
    makes the repair deterministic.
    """
    removed: list[tuple[str, str]] = []
    for cycle in cycles:
        ordered = sorted(cycle, key=order_key)
        position = {code: i for i, code in enumerate(ordered)}
        for code in ordered:
            for dep in sorted(edges.get(code, ())):
                # A self-loop is always backward; so is depending on a member
                # that the chosen order places later.
                if dep in position and position[dep] >= position[code]:
                    edges[code].discard(dep)
                    removed.append((code, dep))
    return removed


def levels(edges: dict[str, set[str]]) -> dict[str, int]:
    """How many prerequisite layers deep each course sits. Requires acyclicity.

    level 0 means "nothing unmet is blocking it", i.e. takeable now.
    """
    order = list(TopologicalSorter(
        {node: set(deps) for node, deps in edges.items()}).static_order())

    result: dict[str, int] = {}
    for node in order:
        if node not in edges:
            continue                       # a dep outside the checklist
        depths = [result[d] for d in edges[node] if d in result]
        result[node] = 1 + max(depths) if depths else 0
    return result


def downstream_counts(edges: dict[str, set[str]]) -> dict[str, int]:
    """How many courses transitively depend on each course.

    This is the "Unlocks" number the UI shows: among otherwise equal choices,
    taking what unblocks the most is what makes a plan short.
    """
    dependents: dict[str, set[str]] = {node: set() for node in edges}
    for node, deps in edges.items():
        for dep in deps:
            if dep in dependents:
                dependents[dep].add(node)

    order = list(TopologicalSorter(
        {node: set(deps) for node, deps in edges.items()}).static_order())

    # Walk dependencies-last: a node's dependents are resolved before it is.
    reachable: dict[str, set[str]] = {}
    for node in reversed(order):
        if node not in edges:
            continue
        acc: set[str] = set()
        for child in dependents.get(node, ()):
            acc.add(child)
            acc |= reachable.get(child, set())
        reachable[node] = acc
    return {node: len(found) for node, found in reachable.items()}


# --- Corequisite bundling ------------------------------------------------------

def _infer_lab_pairs(courses: dict[str, Course]) -> tuple[list[tuple[str, str]],
                                                          list[str]]:
    """Pair a laboratory course with its lecture by code prefix and title.

    Undergraduate §10.10.1 requires a laboratory and its corresponding lecture
    in the same term. Checklists often state that only implicitly, so it is
    inferred — and, being a DLSU-specific heuristic rather than a stated fact,
    every inference is reported so the user can overrule it.
    """
    pairs: list[tuple[str, str]] = []
    notes: list[str] = []

    for code in sorted(courses):
        course = courses[code]
        title = course.title.strip().lower()
        looks_like_lab = (code.startswith(_LAB_CODE_PREFIX)
                          or title.endswith(_LAB_TITLE_SUFFIX))
        if not looks_like_lab or not title.endswith(_LAB_TITLE_SUFFIX):
            continue

        stem = title[: -len(_LAB_TITLE_SUFFIX)].strip()
        if not stem:
            continue

        matches = [
            other.code for other in courses.values()
            if other.code != code
            and other.title.strip().lower() == stem
            and (other.year, other.term) == (course.year, course.term)
        ]
        if len(matches) == 1:
            pairs.append((code, matches[0]))
        elif matches:
            notes.append(
                f"{code} could be the laboratory for any of "
                f"{', '.join(sorted(matches))}; not paired. Set `coreqs:` by "
                "hand if they must be taken together (Undergraduate §10.10.1)."
            )
    return pairs, notes


def coreq_bundles(courses: dict[str, Course], levels_map: dict[str, int],
                  pair_labs: bool) -> tuple[list[Bundle], list[str]]:
    """Group courses that must share a term into bundles.

    Ordinary courses become one-element bundles, so the packer has exactly one
    kind of thing to place. Bundles are returned sorted by their first code, so
    set iteration order can never leak into the plan.
    """
    notes: list[str] = []
    adjacency: dict[str, set[str]] = {code: set() for code in courses}
    inferred_links: set[frozenset[str]] = set()

    for code, course in courses.items():
        for other in course.coreqs:
            if other in adjacency:
                adjacency[code].add(other)
                adjacency[other].add(code)      # corequisite is symmetric

    if pair_labs:
        pairs, pair_notes = _infer_lab_pairs(courses)
        notes.extend(pair_notes)
        for lab, lecture in pairs:
            if lecture in adjacency[lab]:
                continue                        # already stated; not inferred
            adjacency[lab].add(lecture)
            adjacency[lecture].add(lab)
            inferred_links.add(frozenset((lab, lecture)))
            notes.append(
                f"Paired {lab} with {lecture} in the same term because it looks "
                "like its laboratory (Undergraduate §10.10.1 requires a lab and "
                "its lecture together). This is inferred, not stated on your "
                "checklist — set `planner.pair_labs: false` or edit `coreqs:` "
                "if it is wrong."
            )

    bundles: list[Bundle] = []
    seen: set[str] = set()
    for code in sorted(courses):
        if code in seen:
            continue
        component: list[str] = []
        queue = [code]
        seen.add(code)
        while queue:
            current = queue.pop()
            component.append(current)
            for neighbour in sorted(adjacency[current]):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)

        codes = tuple(sorted(component))
        bundles.append(Bundle(
            codes=codes,
            units=round(sum(courses[c].units for c in codes), 1),
            level=max(levels_map.get(c, 0) for c in codes),
            inferred=any(frozenset(pair) in inferred_links
                         for pair in zip(codes, codes[1:])
                         ) or any(
                link <= set(codes) for link in inferred_links),
        ))

    bundles.sort(key=lambda b: b.codes[0])
    return bundles, notes


# --- The plan ------------------------------------------------------------------

def _term_label(index: int) -> str:
    return "Next term" if index == 1 else f"Term +{index}"


def _units(value: float) -> str:
    """"1 unit" / "3 units" — these notes are read by a person."""
    return f"{value:g} unit" + ("" if value == 1 else "s")


def resolve_taken(curriculum: Curriculum, extra: set[str] | None = None,
                  notes: list[str] | None = None) -> set[str]:
    """Passed courses from the artifact, plus anything the user typed.

    Public because the UI must report the same "completed" figure the planner
    actually used; computing it twice from different rules is how a plan and its
    own header end up disagreeing.
    """
    resolved = taken_codes(curriculum)
    for raw in sorted(extra or ()):
        code = str(raw).strip().upper()
        if not code:
            continue
        if code in curriculum.courses:
            resolved.add(code)
        elif notes is not None:
            notes.append(
                f"{code} is not on your checklist, so it was ignored. Check the "
                "code, or add the course to your curriculum file."
            )
    return resolved


def _flag_impossible_history(curriculum: Curriculum, taken: set[str],
                             attempted: set[str], notes: list[str]) -> None:
    """Warn when a passed course's own prerequisite is not passed.

    That combination cannot happen in reality, so it means the record is wrong:
    usually a whole term was marked complete and then one course inside it was
    removed as failed, leaving its dependents still marked as passed. Saying so
    is the difference between a plan built on a contradiction and one the
    student can trust — but it is a note, not a refusal, because only they know
    which half is right.
    """
    contradictions: list[tuple[str, list[str]]] = []
    for code in sorted(taken):
        course = curriculum.courses.get(code)
        if course is None:
            continue
        # Only HARD prerequisites make a history impossible. Passing a course
        # whose soft prerequisite you merely failed is perfectly legitimate —
        # that is what "soft" means.
        missing = sorted(p for p in course.prereqs
                         if p in curriculum.courses and p not in taken)
        missing += sorted(p for p in course.soft_prereqs
                          if p in curriculum.courses and p not in attempted)
        if missing:
            contradictions.append((code, missing))

    for code, missing in contradictions[:5]:
        notes.append(
            f"You have {code} marked as passed, but not its prerequisite "
            f"{', '.join(missing)}. One of those is wrong — either {code} is "
            f"not actually passed, or {missing[0]} is. I planned "
            f"{missing[0]} as a retake and left {code} alone."
        )
    if len(contradictions) > 5:
        notes.append(f"...and {len(contradictions) - 5} more course(s) are "
                     "marked passed without their prerequisites.")


def _unmet_edges(curriculum: Curriculum, taken: set[str], attempted: set[str],
                 notes: list[str]) -> dict[str, set[str]]:
    """Requirement edges among not-yet-passed courses only.

    A requirement already satisfied is simply absent, so `level 0` means exactly
    "takeable now". The two kinds of requirement are satisfied by different
    sets, which is the whole point of tracking both:

      * a HARD prerequisite must be in `taken` — actually passed;
      * a SOFT prerequisite need only be in `attempted` — sat, pass or fail.

    Both still produce an edge when unsatisfied, because either way the course
    has to come first; they differ only in what clears them.

    A requirement naming a course that is not on the checklist is dropped with a
    note rather than treated as blocking: blocking forever would hide a course
    the student can actually take, and explain nothing.
    """
    edges: dict[str, set[str]] = {}
    for code, course in curriculum.courses.items():
        if code in taken:
            continue
        unmet = set()
        for prereq, satisfied_by in ((p, taken) for p in course.prereqs):
            if prereq in satisfied_by:
                continue
            if prereq in curriculum.courses:
                unmet.add(prereq)
            else:
                notes.append(
                    f"{code} lists {prereq} as a prerequisite, but {prereq} is "
                    "not on your checklist; that requirement was ignored."
                )
        for prereq in course.soft_prereqs:
            if prereq in attempted:
                continue          # sat it, even if failed — that is enough
            if prereq in curriculum.courses:
                unmet.add(prereq)
            else:
                notes.append(
                    f"{code} lists {prereq} as a soft prerequisite, but "
                    f"{prereq} is not on your checklist; it was ignored."
                )
        edges[code] = unmet
    return edges


def _year_term_levels(curriculum: Curriculum,
                      pending: dict[str, Course]) -> dict[str, int]:
    """Levels taken from the checklist's own year/term layout.

    Used when the checklist has no prerequisite column. Crucially this
    synthesizes NO edges: "every first-term course precedes every second-term
    course" would draw a dense, confident, wrong graph. Using the term index
    directly as the level yields the same ordering with zero fabricated claims.
    """
    per_year = max(1, curriculum.terms_per_year)
    raw: dict[str, int] = {}
    for code, course in pending.items():
        if course.year is not None and course.term is not None:
            raw[code] = (course.year - 1) * per_year + (course.term - 1)

    if not raw:
        return {code: 0 for code in pending}

    base = min(raw.values())
    result = {code: value - base for code, value in raw.items()}
    unplaced = max(result.values()) + 1        # unknown year/term schedules last
    for code in pending:
        result.setdefault(code, unplaced)
    return result


def build_plan(curriculum: Curriculum, taken: set[str] | None = None, *,
               max_units: float, min_units: float, max_terms: int,
               pair_labs: bool = True,
               attempted: set[str] | None = None) -> StudyPlan:
    """Plan the remaining courses. Never raises for a data reason.

    Args:
        taken: Extra PASSED codes beyond those flagged in the curriculum.
        attempted: Codes the student has SAT, whether or not they passed —
            a failed course belongs here but not in `taken`. Soft ("S")
            prerequisites are satisfied by this set. Everything passed is
            implicitly attempted; None means "nothing beyond what was passed",
            which is the conservative reading.
        max_units: The general per-term cap, used only when the checklist states
            no term loads of its own. The checklist's own per-term limits win
            when present, and `curriculum.max_units_override` beats both —
            §10.2's "15 units, or the number of units indicated on the program
            checklist", in precedence order.
        min_units: Full-time floor (§10.1). Warned about, never packed toward.
        max_terms: Horizon, and the guard against a malformed graph looping.
    """
    notes: list[str] = []
    passed = resolve_taken(curriculum, taken, notes)
    # Passing a course means you sat it, so `attempted` always contains
    # `passed`; the caller adds the ones that were sat and failed.
    sat = passed | resolve_taken(curriculum, attempted) | attempted_codes(curriculum)
    caps = _term_caps_for(curriculum, max_units, notes)

    pending = {code: course for code, course in curriculum.courses.items()
               if code not in passed}
    if not pending:
        if curriculum.courses:
            notes.append("Every course on your checklist is already marked as "
                         "passed — there is nothing left to plan.")
        return StudyPlan([], [], [], [], [], [], notes)

    _flag_impossible_history(curriculum, passed, sat, notes)
    edges = _unmet_edges(curriculum, passed, sat, notes)

    # --- Cycles: report every one, repair deterministically, keep planning.
    cycles = find_cycles(edges)
    if cycles:
        order_key = _checklist_sort_key(curriculum)
        break_cycles(edges, cycles, order_key)
        for cycle in cycles:
            members = " and ".join(cycle) if len(cycle) == 2 else ", ".join(cycle)
            if len(cycle) == 1:
                notes.append(
                    f"{cycle[0]} lists itself as its own prerequisite — almost "
                    "certainly an extraction error. I ignored that and scheduled "
                    "it. Check `prereqs:` for it in your curriculum file."
                )
            else:
                notes.append(
                    f"{members} list each other as prerequisites, which cannot "
                    "be satisfied and is almost certainly an extraction error. I "
                    "used your checklist's term order to break the tie and "
                    "scheduled them all. Check `prereqs:` for these in your "
                    "curriculum file."
                )

    # --- Levels: from the graph, or from the sheet's layout when that is all
    # we have. Either way, no edge is ever invented.
    if curriculum.prereq_source is PrereqSource.YEAR_TERM:
        levels_map = _year_term_levels(curriculum, pending)
        notes.append(
            "Your checklist states no prerequisites, so the ordering below "
            "follows its own year/term layout rather than a dependency graph. "
            "Verify with your adviser before enrolling."
        )
    elif curriculum.prereq_source is PrereqSource.NONE:
        levels_map = {code: 0 for code in pending}
        notes.append(
            "No prerequisite or year/term information could be read from your "
            "checklist, so this is a list of what remains, not an ordering. "
            "Fill in `prereqs:` or `year:`/`term:` in your curriculum file and "
            "run /plan again."
        )
    else:
        levels_map = levels(edges)

    unknown = sorted(code for code, course in pending.items()
                     if course.confidence is PrereqConfidence.UNKNOWN
                     and not course.prereqs)
    if unknown and curriculum.prereq_source is PrereqSource.COLUMN:
        notes.append(
            f"No prerequisite information was available for {len(unknown)} "
            f"course(s) ({', '.join(unknown[:5])}"
            f"{', ...' if len(unknown) > 5 else ''}). They are treated as "
            "takeable, but that is an absence of information, not a clearance — "
            "verify before enrolling."
        )

    downstream = downstream_counts(edges)
    bundles, bundle_notes = coreq_bundles(pending, levels_map, pair_labs)
    notes.extend(bundle_notes)

    # --- A bundle bigger than the cap can never be scheduled. Say so, and take
    # it out of the loop rather than letting it spin forever.
    largest_cap = max(caps.values(), default=max_units)
    unreachable: list[str] = []
    schedulable: list[Bundle] = []
    for bundle in bundles:
        credited = credited_units(pending[c] for c in bundle.codes)
        if credited > largest_cap:
            unreachable.extend(bundle.codes)
            notes.append(
                f"{' + '.join(bundle.codes)} must be taken together and total "
                f"{credited:g} credited units, which exceeds even the largest "
                f"term limit on your checklist ({largest_cap:g}), so they cannot "
                "be scheduled. This usually means a misparsed units column, or a "
                "corequisite that should not be one — check your curriculum file."
            )
        else:
            schedulable.append(bundle)

    plan_terms, available_now, deferred, blocked = _pack(
        schedulable, pending, edges, downstream, curriculum,
        caps=caps, fallback_cap=largest_cap, min_units=min_units,
        max_terms=max_terms, notes=notes)

    return StudyPlan(
        terms=plan_terms,
        available_now=available_now,
        deferred=deferred,
        blocked=blocked,
        unreachable=sorted(unreachable),
        cycles=cycles,
        notes=notes,
    )


def _term_caps_for(curriculum: Curriculum, configured_max: float,
                   notes: list[str]) -> dict[int, float]:
    """The unit limit to apply to each of the checklist's terms.

    Undergraduate §10.2 caps a regular term at 15 units "OR the number of units
    indicated on the program checklist". For an engineering program those are
    not the same number — the BS CpE checklist prescribes 16-19 credited units
    per term — and the checklist's number is the one that governs. So the
    checklist's own per-term loads win when it states them; `program.max_units`
    overrides everything; and the configured default is the last resort for a
    curriculum that says nothing about terms at all.
    """
    if curriculum.max_units_override:
        return {index: curriculum.max_units_override
                for index in (curriculum.term_caps or {1: 0.0})}
    if curriculum.term_caps:
        heaviest = max(curriculum.term_caps.values())
        if heaviest > configured_max:
            notes.append(
                f"Your checklist prescribes up to {heaviest:g} units in a term, "
                f"above the general {configured_max:g}-unit maximum. That is "
                "what Undergraduate §10.2 allows when it defers to \"the number "
                "of units indicated on the program checklist\", so the "
                "checklist's own limits are the ones applied below."
            )
        return dict(curriculum.term_caps)
    return {}


def _resume_position(curriculum: Curriculum, pending: dict[str, Course]) -> int:
    """The checklist term the student is picking up from.

    The earliest term that still holds unfinished work: someone who has cleared
    the first six terms is entering the seventh, and the seventh's prescribed
    load is the one that applies to them.
    """
    indexes = [course.term_index(curriculum.terms_per_year)
               for course in pending.values()]
    known = [i for i in indexes if i is not None]
    return min(known) if known else 1


def _checklist_sort_key(curriculum: Curriculum):
    """Order courses by the program's own intended sequence, then by code."""
    def key(code: str):
        course = curriculum.courses.get(code)
        return (course.checklist_order if course else (99, 99), code)
    return key


def _pack(bundles: list[Bundle], pending: dict[str, Course],
          edges: dict[str, set[str]], downstream: dict[str, int],
          curriculum: Curriculum, *, caps: dict[int, float],
          fallback_cap: float, min_units: float, max_terms: int,
          notes: list[str]):
    """Greedy, level-respecting, first-fit term packing.

    The ordering key below is deliberately TOTAL: its last component breaks
    every remaining tie, so shuffling the input dict cannot change one byte of
    the output (docs/course_planner.md §8, and the test named
    test_tie_break_is_stable_under_input_reordering).
    """
    def sort_key(bundle: Bundle):
        return (
            bundle.level,                                    # correctness first
            min(pending[c].checklist_order for c in bundle.codes),  # the program's sequence
            all(pending[c].placeholder for c in bundle.codes),      # real courses first
            -max(downstream.get(c, 0) for c in bundle.codes),       # unblock the most
            -bundle.units,
            bundle.codes[0],                                 # always decides
        )

    remaining = list(bundles)
    scheduled: set[str] = set()
    terms: list[PlannedTerm] = []
    available_now: list[str] = []
    deferred: list[str] = []
    position = _resume_position(curriculum, pending)

    for index in range(1, max_terms + 1):
        if not remaining:
            break

        # The checklist's own limit for the term being entered. A student who
        # has cleared six terms is entering the seventh, so the seventh's
        # prescribed load governs, then the eighth's, and so on.
        checklist_term = position + index - 1
        cap = caps.get(checklist_term, fallback_cap)

        eligible = [b for b in remaining
                    if all(dep in scheduled
                           for code in b.codes for dep in edges.get(code, ()))]
        if not eligible:
            break                       # nothing can advance; leftovers are blocked

        eligible.sort(key=sort_key)
        if index == 1:
            # Everything takeable next term, whether or not the cap fitted it:
            # a student with ten options can only take five, and hiding the
            # other five would hide the choice they actually have.
            available_now = sorted(c for b in eligible for c in b.codes)

        picked: list[Bundle] = []
        units = 0.0
        for bundle in eligible:
            # Only credited units count against the limit; NSTP and the
            # Lasallian series must still be taken but sit outside the load,
            # exactly as the checklist's own "18 (3)" totals show.
            weight = credited_units(pending[c] for c in bundle.codes)
            if units + weight <= cap:
                picked.append(bundle)
                units = round(units + weight, 1)

        if not picked:
            break                       # defensive; an empty term always fits one

        if index == 1:
            taken_this_term = {c for b in picked for c in b.codes}
            deferred = sorted(c for b in eligible for c in b.codes
                              if c not in taken_this_term)

        courses = sorted((pending[c] for b in picked for c in b.codes),
                         key=lambda c: (c.checklist_order, c.code))
        terms.append(PlannedTerm(index=index, label=_term_label(index),
                                 courses=courses, units=units, cap=cap,
                                 checklist_term=checklist_term
                                 if checklist_term in caps else None))
        for bundle in picked:
            scheduled.update(bundle.codes)
            remaining.remove(bundle)

        if units < min_units and len(picked) == len(eligible):
            # Below the full-time floor because nothing else was ELIGIBLE, not
            # because the cap stopped us — so the cap is not the thing to relax.
            notes.append(
                f"{_term_label(index).lower().capitalize()} comes to only "
                f"{_units(units)}, below the {min_units:g}-unit full-time "
                "minimum for undergraduates (Undergraduate §10.1). That is "
                "expected if you are graduating; otherwise ask your College "
                "Associate Dean."
            )

    blocked = sorted(code for bundle in remaining for code in bundle.codes)
    if blocked:
        _explain_blocked(blocked, edges, scheduled, notes, max_terms,
                         len(terms))

    if not blocked and len(terms) >= 2 and terms[-1].units <= GRADUATING_OVERLOAD_UNITS:
        notes.append(
            f"The final term is only {_units(terms[-1].units)}. Undergraduate "
            f"§10.2 lets a graduating student overload by up to "
            f"{GRADUATING_OVERLOAD_UNITS:g} units with approval, so this plan "
            "could finish one term earlier — that is a conversation with your "
            "Associate Dean, not something the planner decides."
        )

    return terms, available_now, deferred, blocked


def _explain_blocked(blocked: list[str], edges: dict[str, set[str]],
                     scheduled: set[str], notes: list[str], max_terms: int,
                     terms_used: int) -> None:
    """Say WHY each unscheduled course is stuck — the actionable form."""
    if terms_used >= max_terms:
        notes.append(
            f"Planning stopped at the {max_terms}-term horizon "
            f"(`planner.max_terms`); {len(blocked)} course(s) remain unplanned.")
        return

    for code in blocked[:8]:
        stuck = sorted(dep for dep in edges.get(code, ())
                       if dep not in scheduled)
        if stuck:
            notes.append(
                f"{code} could not be scheduled: it still needs "
                f"{', '.join(stuck)}, which could not be scheduled either.")
        else:
            notes.append(f"{code} could not be scheduled within the horizon.")
    if len(blocked) > 8:
        notes.append(f"...and {len(blocked) - 8} more could not be scheduled.")
