"""
model.py — The curriculum data model and its hand-editable YAML artifact.

Purpose:
    Define Course/Curriculum (what a program checklist contains) and the
    round trip to `<program>.curriculum.yaml`: the file the user inspects
    and corrects, and the ONLY thing the planner reads.

Inputs:
    A Curriculum (from src.curriculum.checklist_parser, or hand-authored);
    or a path to a curriculum YAML file.

Outputs:
    write_curriculum_yaml() persists the artifact; load_curriculum_yaml()
    validates and returns a Curriculum.

Dependencies:
    pyyaml (external), dataclasses/pathlib/enum (standard library).

Why this file exists:
    Architectural Decision AD-8: the layout of a MyLaSalle checklist export
    is outside our control, so the checklist is parsed ONCE into a file the
    user can fix by hand, and the planner reads that file rather than the
    PDF. The escape hatch is the design, not a fallback — which makes this
    module, not the parser, the contract everything downstream depends on.

    YAML rather than the .jsonl used for ingestion intermediates: those hold
    hundreds of machine-written records and are never hand-edited, whereas
    this file exists to be hand-edited, and YAML carries the comments that
    tell the user what was inferred and what to fix (the same reasoning that
    made tests/golden_set.yaml a YAML file).
"""

from __future__ import annotations

import dataclasses

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

import yaml

# Bumped only for a breaking change to the file format. load_curriculum_yaml
# refuses anything else rather than silently misreading a future schema.
SCHEMA_VERSION = 1

# Grades that count as having passed a course, so it satisfies a prerequisite.
# Everything else — 0.0 (failed), 9.9 (deferred), INC, W/WP/WF — does NOT.
PASSING_GRADES = frozenset({
    "4.0", "3.5", "3.0", "2.5", "2.0", "1.5", "1.0",   # numeric scale
    "P", "S", "CR", "✓",                          # pass / satisfactory / credited
})

# A units value above this is certainly a misparse (a total, or a page number).
MAX_PLAUSIBLE_UNITS = 15.0


class PrereqConfidence(str, Enum):
    """How a course's prerequisites came to be what they are."""

    STATED = "stated"                          # read from a prerequisite column
    INFERRED_YEAR_TERM = "inferred_year_term"  # ordering came from the sheet's layout
    UNKNOWN = "unknown"                        # nothing could be established


class PrereqSource(str, Enum):
    """What the whole checklist yielded — the three cases the planner handles."""

    COLUMN = "column"        # a real prerequisite column: full dependency graph
    YEAR_TERM = "year_term"  # only year/term grouping: order by the sheet's own layout
    NONE = "none"            # neither: nothing may be claimed about ordering


@dataclass(frozen=True)
class Course:
    """One row of a program checklist.

    Frozen, with tuples rather than lists, so a Course is hashable and cannot
    be mutated part-way through planning — a plan that changed its own inputs
    would not be reproducible.
    """

    code: str
    title: str
    units: float
    year: int | None = None
    term: int | None = None
    prereqs: tuple[str, ...] = ()      # HARD: must be passed first
    coreqs: tuple[str, ...] = ()       # must be taken in the same term (§10.10.1)
    confidence: PrereqConfidence = PrereqConfidence.UNKNOWN
    taken: bool = False
    grade: str | None = None           # informational; `taken` is what planning uses
    placeholder: bool = False          # "GE ELECTIVE 1" — a slot, not a real course
    source_page: int | None = None     # page of the checklist PDF it was read from
    # False for a course whose units the checklist shows in parentheses — NSTP
    # and the Lasallian series. They must still be taken, but they do not count
    # against a term's unit limit, which is exactly how the sheet's own totals
    # arithmetic works ("18 (3)" = 18 credited plus 3 non-credit).
    credited: bool = True
    # SOFT prerequisites: must have been SAT, but not necessarily passed. A
    # checklist marks these "S": having taken Differential Calculus — even
    # having failed it — clears you for Engineering Economics. Satisfied by
    # `attempted`, not by `taken`, which is the whole reason those two sets are
    # tracked separately. Last in the field order deliberately: tests and
    # fixtures across this repo build Course positionally, so a new field goes
    # on the end rather than shifting every existing call.
    soft_prereqs: tuple[str, ...] = ()

    @property
    def checklist_order(self) -> tuple[int, int]:
        """Sort key from the sheet's own layout; unknowns sort last.

        Used as a planner tie-break so that, all else equal, the university's
        own intended sequence wins over anything we inferred.
        """
        return (self.year or 99, self.term or 99)

    def term_index(self, terms_per_year: int) -> int | None:
        """Position in the checklist's own term sequence (1-based), if known."""
        if self.year is None or self.term is None:
            return None
        return (self.year - 1) * max(1, terms_per_year) + self.term


@dataclass
class Curriculum:
    """A whole program checklist, plus what extraction managed to establish."""

    program_id: str
    program_name: str
    terms_per_year: int
    courses: dict[str, Course]           # code -> Course, in checklist order
    prereq_source: PrereqSource
    # Undergraduate §10.2 caps the load at 15 units "or the number of units
    # indicated on the program checklist" — so a checklist that states its own
    # cap overrides planner.max_units. None means "use the configured default".
    max_units_override: float | None = None
    # Credited units the checklist prescribes for each of its own terms, keyed
    # by term index (1-based, running across years). This is the "or the number
    # indicated on the program checklist" clause made concrete: engineering
    # programs routinely prescribe 16-19 units, well above the general 15.
    term_caps: dict[int, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # code -> prereq codes that name a course absent from this checklist. Those
    # edges are dropped rather than treated as permanently blocking; see
    # load_curriculum_yaml for why that is a warning and not an error.
    unresolved_prereqs: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Bundle:
    """Courses that must be scheduled in the same term (§10.10.1).

    Ordinary courses are one-element bundles, so the packer has exactly one
    kind of thing to place.
    """

    codes: tuple[str, ...]        # sorted, so iteration order cannot leak out
    units: float
    level: int
    inferred: bool = False        # the coreq link was guessed, not stated


@dataclass
class PlannedTerm:
    """One term of the recommended schedule."""

    index: int                    # 1 = the next term the student will enroll in
    label: str                    # "Next term", "Term +2", ...
    courses: list[Course]         # deterministically ordered
    units: float                  # credited units only — see Course.credited
    cap: float = 0.0              # the unit limit that applied to THIS term
    checklist_term: int | None = None   # which of the checklist's terms it maps to


@dataclass
class StudyPlan:
    """The whole recommendation, including everything it could NOT schedule.

    build_plan never raises for data reasons; every problem becomes an entry
    here. That mirrors Answer(error=...) in src/chat/core.py — the UI must
    always have something honest to show.
    """

    terms: list[PlannedTerm]
    available_now: list[str]      # eligible next term BEFORE the unit cap applies
    deferred: list[str]           # eligible now, bumped out by the cap
    blocked: list[str]            # prerequisites unmet within the horizon
    unreachable: list[str]        # can never be scheduled (e.g. over-cap bundle)
    cycles: list[list[str]]       # every prerequisite cycle found, canonicalized
    notes: list[str]              # honest caveats; the UI prints these in yellow


class CurriculumError(Exception):
    """Raised when a curriculum YAML is missing, malformed, or inconsistent."""


def taken_codes(curriculum: Curriculum) -> set[str]:
    """Codes the student has PASSED, per the `taken` flag."""
    return {c.code for c in curriculum.courses.values() if c.taken}


def attempted_codes(curriculum: Curriculum) -> set[str]:
    """Codes the student has SAT, whether or not they passed.

    A recorded grade means the course was taken, so a 0.0 counts here and not
    in taken_codes(). That distinction is what makes a soft prerequisite work:
    it asks whether you have been through the course, not whether you cleared it.
    """
    return {c.code for c in curriculum.courses.values() if c.taken or c.grade}


def total_units(courses: Iterable[Course]) -> float:
    """Sum of units, rounded to one decimal to keep 0.1-style float noise out."""
    return round(sum(c.units for c in courses), 1)


def credited_units(courses: Iterable[Course]) -> float:
    """Units that count against a term's limit — non-credit courses excluded."""
    return round(sum(c.units for c in courses if c.credited), 1)


def derive_term_caps(courses: Iterable[Course],
                     terms_per_year: int) -> dict[int, float]:
    """The credited load the checklist prescribes for each of its terms.

    Summing the courses the sheet places in a term reproduces its own TOTAL row
    (verified on the BS CpE checklist: 10 of 12 terms match exactly, and the two
    that do not are broken cells in the source spreadsheet — a literal 0 and an
    Excel #REF! — where this sum is the correct figure). Deriving the cap rather
    than parsing the TOTAL row is therefore both simpler and more robust.
    """
    caps: dict[int, float] = {}
    for course in courses:
        index = course.term_index(terms_per_year)
        if index is None or not course.credited:
            continue
        caps[index] = round(caps.get(index, 0.0) + course.units, 1)
    return caps


def remaining_courses(curriculum: Curriculum) -> list[Course]:
    """Courses still to be taken, in checklist order."""
    return [c for c in curriculum.courses.values() if not c.taken]


# --- Writing -------------------------------------------------------------------

def _header(curriculum: Curriculum, source_pdf: Path | None, tier: str,
            path: Path) -> str:
    """The comment block that tells the user what this file is and how to fix it.

    pyyaml cannot emit comments, so the header is composed as text and
    prepended to the dumped body. It is the whole reason this artifact is YAML
    rather than JSON: provenance and warnings travel WITH the data.
    """
    stated = sum(1 for c in curriculum.courses.values() if c.prereqs)
    n = len(curriculum.courses)
    lines = [
        "# " + "-" * 74,
        f"# {path.name} — extracted program checklist. HAND-EDITABLE.",
        "#",
    ]
    if source_pdf is not None:
        lines.append(f"# Generated by scripts/inspect_checklist.py from {source_pdf}.")
    lines += [
        f"# Extraction tier: {tier}. {n} course(s) parsed.",
        f"# PREREQUISITE SOURCE: {curriculum.prereq_source.value} "
        f"({stated} of {n} course(s) have stated prerequisites).",
        "#",
        "# THIS FILE, NOT THE PDF, IS WHAT THE PLANNER READS. If anything below is",
        "# wrong, fix it here and run /plan again — the PDF is never parsed again.",
        "# Re-running inspect_checklist.py will NOT overwrite this file (--force).",
    ]
    if curriculum.warnings:
        lines.append("#")
        lines.append(f"# {len(curriculum.warnings)} warning(s) — see extraction.warnings below.")
    lines.append("# " + "-" * 74)
    return "\n".join(lines) + "\n"


def _course_to_dict(course: Course) -> dict:
    """A Course as plain YAML-safe types, in a deliberate field order.

    Order matters: this file is read by a human, so code/title/units come
    first and bookkeeping last.
    """
    return {
        "code": course.code,
        "title": course.title,
        "units": course.units,
        "year": course.year,
        "term": course.term,
        "prereqs": list(course.prereqs),
        "soft_prereqs": list(course.soft_prereqs),
        "coreqs": list(course.coreqs),
        "prereq_confidence": course.confidence.value,
        "taken": course.taken,
        "grade": course.grade,
        "credited": course.credited,
        "placeholder": course.placeholder,
        "source_page": course.source_page,
    }


def write_curriculum_yaml(curriculum: Curriculum, path: Path | str, *,
                          source_pdf: Path | str | None = None,
                          tier: str = "manual",
                          columns: dict | None = None,
                          force: bool = False) -> Path:
    """Persist a Curriculum as the hand-editable artifact.

    Args:
        columns: The column roles extraction voted for, echoed into the file so
            the user can correct a mis-voted column.
        force: Required to overwrite an existing file.

    Returns:
        The path written.

    Raises:
        CurriculumError: If the file exists and force is False. Silently
            clobbering hand corrections is the one unforgivable failure for
            this artifact (AD-8), so it is refused rather than warned about.
    """
    path = Path(path)
    if path.exists() and not force:
        raise CurriculumError(
            f"{path} already exists and may contain your corrections. "
            "Pass --force (or force=True) to regenerate it from the PDF."
        )

    body = {
        "schema_version": SCHEMA_VERSION,
        "program": {
            "id": curriculum.program_id,
            "name": curriculum.program_name,
            "source_pdf": str(source_pdf) if source_pdf else None,
            "terms_per_year": curriculum.terms_per_year,
            "max_units": curriculum.max_units_override,
        },
        # The credited load this checklist prescribes per term. Undergraduate
        # §10.2 caps a regular term at 15 units "or the number of units
        # indicated on the program checklist" — these ARE that number, so they
        # govern. Edit one to change the cap the planner applies to that term.
        "term_units": {index: caps for index, caps
                       in sorted(curriculum.term_caps.items())},
        "extraction": {
            "strategy": tier,
            "prereq_source": curriculum.prereq_source.value,
            "courses_parsed": len(curriculum.courses),
            "columns": columns,
            "warnings": list(curriculum.warnings),
        },
        "courses": [_course_to_dict(c) for c in curriculum.courses.values()],
    }

    dumped = yaml.safe_dump(body, sort_keys=False, default_flow_style=False,
                            allow_unicode=True, width=88)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _header(curriculum, Path(source_pdf) if source_pdf else None, tier, path)
        + dumped,
        encoding="utf-8",
    )
    return path


# --- Loading -------------------------------------------------------------------

def _as_int_or_none(value, field_name: str, where: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise CurriculumError(
            f"{where}: '{field_name}' must be a whole number or null (got {value!r})"
        ) from None


def _code_list(value, field_name: str, where: str) -> tuple[str, ...]:
    """Normalize a prereqs/coreqs cell into uppercase codes.

    A bare string is accepted as a one-item list, because that is what a
    hand-edit like `prereqs: GEMATMW` produces and rejecting it would punish
    the user for using the file the way it is meant to be used.
    """
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise CurriculumError(
            f"{where}: '{field_name}' must be a list of course codes (got {value!r})"
        )
    out, seen = [], set()
    for item in value:
        code = str(item).strip().upper()
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return tuple(out)


def _enum_or_default(raw, enum_cls, default, field_name: str, where: str):
    if raw is None or raw == "":
        return default
    try:
        return enum_cls(str(raw).strip().lower())
    except ValueError:
        valid = sorted(m.value for m in enum_cls)
        raise CurriculumError(
            f"{where}: '{field_name}' must be one of {valid} (got {raw!r})"
        ) from None


def load_curriculum_yaml(path: Path | str) -> Curriculum:
    """Load, validate, and return a Curriculum from its YAML artifact.

    Validation is loud and names the offender, in the same spirit as
    src.utils.config._validate: a malformed curriculum should fail before any
    planning happens, not produce a quietly wrong schedule.

    One deliberate exception: a prerequisite naming a course that is not on the
    checklist is a WARNING, not an error. Checklists legitimately reference
    codes from other curriculum versions or shifted-in courses, and treating
    that as fatal would make a correct file unloadable. The edge is dropped and
    recorded in `unresolved_prereqs` so the planner can say so.

    Raises:
        CurriculumError: Missing file, unparseable YAML, wrong schema version,
            duplicate codes, or a course missing code/title/units.
    """
    path = Path(path)
    if not path.exists():
        raise CurriculumError(f"Curriculum file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CurriculumError(f"Could not parse {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise CurriculumError(f"{path} did not contain a mapping at the top level")

    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise CurriculumError(
            f"{path}: schema_version {version!r} is not supported "
            f"(this build reads version {SCHEMA_VERSION})"
        )

    program = raw.get("program") or {}
    extraction = raw.get("extraction") or {}
    raw_courses = raw.get("courses")
    if not isinstance(raw_courses, list):
        raise CurriculumError(f"{path}: 'courses' must be a list")

    terms_per_year = _as_int_or_none(
        program.get("terms_per_year", 3), "terms_per_year", f"{path}: program") or 3
    if not (1 <= terms_per_year <= 6):
        raise CurriculumError(
            f"{path}: program.terms_per_year must be between 1 and 6 "
            f"(got {terms_per_year})"
        )

    max_units_override = program.get("max_units")
    if max_units_override is not None:
        try:
            max_units_override = float(max_units_override)
        except (TypeError, ValueError):
            raise CurriculumError(
                f"{path}: program.max_units must be a number or null "
                f"(got {max_units_override!r})"
            ) from None

    courses: dict[str, Course] = {}
    for index, entry in enumerate(raw_courses):
        where = f"{path}: courses[{index}]"
        if not isinstance(entry, dict):
            raise CurriculumError(f"{where} is not a mapping")

        code = str(entry.get("code") or "").strip().upper()
        if not code:
            raise CurriculumError(f"{where} has no 'code'")
        if code in courses:
            raise CurriculumError(
                f"{where}: duplicate course code {code!r} (already defined earlier). "
                "Delete or rename one of them."
            )

        title = str(entry.get("title") or "").strip()
        if not title:
            raise CurriculumError(f"{where} ({code}) has no 'title'")

        if "units" not in entry or entry["units"] is None:
            raise CurriculumError(f"{where} ({code}) has no 'units'")
        try:
            units = float(entry["units"])
        except (TypeError, ValueError):
            raise CurriculumError(
                f"{where} ({code}): 'units' must be a number (got {entry['units']!r})"
            ) from None
        if not (0.0 <= units <= MAX_PLAUSIBLE_UNITS):
            raise CurriculumError(
                f"{where} ({code}): 'units' must be between 0 and "
                f"{MAX_PLAUSIBLE_UNITS:g} (got {units:g})"
            )

        courses[code] = Course(
            code=code,
            title=title,
            units=units,
            year=_as_int_or_none(entry.get("year"), "year", f"{where} ({code})"),
            term=_as_int_or_none(entry.get("term"), "term", f"{where} ({code})"),
            prereqs=_code_list(entry.get("prereqs"), "prereqs", f"{where} ({code})"),
            soft_prereqs=_code_list(entry.get("soft_prereqs"), "soft_prereqs",
                                    f"{where} ({code})"),
            coreqs=_code_list(entry.get("coreqs"), "coreqs", f"{where} ({code})"),
            confidence=_enum_or_default(
                entry.get("prereq_confidence"), PrereqConfidence,
                PrereqConfidence.UNKNOWN, "prereq_confidence", f"{where} ({code})"),
            taken=bool(entry.get("taken", False)),
            grade=(str(entry["grade"]).strip() if entry.get("grade") not in (None, "")
                   else None),
            credited=bool(entry.get("credited", True)),
            placeholder=bool(entry.get("placeholder", False)),
            source_page=_as_int_or_none(
                entry.get("source_page"), "source_page", f"{where} ({code})"),
        )

    warnings = [str(w) for w in (extraction.get("warnings") or [])]
    unresolved = _drop_unresolved_edges(courses, warnings)

    # A hand-edited term_units block wins; otherwise derive the caps from the
    # courses, so a file written before this section existed still gets them.
    raw_caps = raw.get("term_units") or {}
    term_caps: dict[int, float] = {}
    for key, value in raw_caps.items():
        index = _as_int_or_none(key, "term_units key", f"{path}")
        if index is None:
            continue
        try:
            term_caps[index] = float(value)
        except (TypeError, ValueError):
            raise CurriculumError(
                f"{path}: term_units[{key}] must be a number (got {value!r})"
            ) from None
    if not term_caps:
        term_caps = derive_term_caps(courses.values(), terms_per_year)

    return Curriculum(
        program_id=str(program.get("id") or path.stem.replace(".curriculum", "")),
        program_name=str(program.get("name") or program.get("id") or "Unnamed program"),
        terms_per_year=terms_per_year,
        courses=courses,
        prereq_source=_enum_or_default(
            extraction.get("prereq_source"), PrereqSource, PrereqSource.NONE,
            "prereq_source", f"{path}: extraction"),
        max_units_override=max_units_override,
        term_caps=term_caps,
        warnings=warnings,
        unresolved_prereqs=unresolved,
    )


def _drop_unresolved_edges(courses: dict[str, Course],
                           warnings: list[str]) -> dict[str, list[str]]:
    """Remove prereq/coreq edges pointing at codes absent from the checklist.

    Mutates `courses` in place (replacing frozen Courses with trimmed copies)
    and appends a warning per affected course. Dropping beats blocking: an
    unresolvable reference that blocked forever would hide a course the student
    can actually take, and say nothing about why.
    """
    unresolved: dict[str, list[str]] = {}
    for code, course in list(courses.items()):
        missing_pre = [p for p in course.prereqs if p not in courses]
        missing_soft = [p for p in course.soft_prereqs if p not in courses]
        missing_co = [c for c in course.coreqs if c not in courses]
        if not (missing_pre or missing_soft or missing_co):
            continue

        unresolved[code] = sorted(set(missing_pre + missing_soft + missing_co))
        courses[code] = dataclasses.replace(
            course,
            prereqs=tuple(p for p in course.prereqs if p in courses),
            soft_prereqs=tuple(p for p in course.soft_prereqs if p in courses),
            coreqs=tuple(c for c in course.coreqs if c in courses),
        )
        warnings.append(
            f"{code}: {', '.join(unresolved[code])} "
            f"{'is' if len(unresolved[code]) == 1 else 'are'} not on this checklist; "
            "that requirement was ignored. Add the course, or fix the code."
        )
    return unresolved
