"""
checklist_parser.py — Read a DLSU program checklist PDF into a Curriculum.

Purpose:
    Extract course code, title, units, term, and prerequisite relationships
    from a checklist export, and report honestly how much of that succeeded.

Inputs:
    A text-based checklist PDF (data/checklists/<program>.pdf).

Outputs:
    (Curriculum, ExtractionReport). The caller writes the Curriculum to the
    hand-editable YAML that the planner actually reads.

Dependencies:
    pdfplumber (already required for the handbook), src.curriculum.model.

Why this file exists:
    All of this feature's risk lives here, because a checklist's layout is
    outside our control (Architectural Decision AD-8). Three consequences shape
    the design:

      * Three extraction tiers are tried in order, and the winner is recorded,
        so a sheet without ruled cells still yields something.
      * Column ROLES are decided once per table by voting across rows, not
        guessed per cell — units and grades are both small decimals, and
        per-cell guessing is what fails on real sheets.
      * Everything uncertain becomes a warning carried into the artifact rather
        than a silent assumption, because the user is the one who corrects it.

    Measured against the BS Computer Engineering (ID 122) checklist: tier A
    wins, the sheet is landscape with two side-by-side term tables per page,
    and it carries a prerequisite-TYPE column (H/C/S) that pairs positionally
    with the prerequisite codes. See docs/course_planner.md §3.4.
"""

from __future__ import annotations

import logging
import re

from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from src.curriculum.model import (
    Course,
    Curriculum,
    PrereqConfidence,
    PrereqSource,
    derive_term_caps,
)

log = logging.getLogger(__name__)

# --- Extraction tiers ----------------------------------------------------------

TABLE_LINES = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
TABLE_TEXT = {"vertical_strategy": "text", "horizontal_strategy": "text",
              "text_x_tolerance": 2, "snap_tolerance": 3}

# A tier must find at least this many course-code-bearing rows to be believed.
MIN_COURSE_ROWS = 8
# A units value above this in a course row is a misread term total, not a course.
MAX_PLAUSIBLE_COURSE_UNITS = 15.0
# Words whose left edges differ by more than this start a new cell (tier C).
COLUMN_GAP = 12.0
# Same-line tolerance, matching src/ingestion/parser.py's proven value.
LINE_TOLERANCE = 3.0

# --- Vocabulary ----------------------------------------------------------------

# Deliberately loose: it also matches STUDENT and REMARKS. Two further gates
# (the stoplist below, and column context) do the real work — the same
# pattern-plus-context approach as chunker._provision_matches_section.
COURSE_CODE_RE = re.compile(r"""
    ^(?=[A-Z0-9]{4,9}$)
    [A-Z]{2,7}
    (?: [0-9]{1,4}[A-Z]? | [A-Z]{0,5}[0-9]? )$
""", re.VERBOSE)

HEADER_WORDS = frozenset("""
COURSE COURSES CODE CODES TITLE TITLES UNITS UNIT CREDIT CREDITS GRADE GRADES
TERM TERMS YEAR YEARS TOTAL TOTALS SUBTOTAL PREREQ PREREQS PREREQUISITE
PREREQUISITES COREQ COREQUISITE COREQUISITES REMARKS REMARK LECTURE LEC LAB
LABORATORY SUBJECT SUBJECTS STUDENT NUMBER NAME PROGRAM COLLEGE DEGREE MAJOR
MINOR ELECTIVE ELECTIVES NONE TAKEN PASSED CREDITED PENDING SUMMER AND OR THE
OF FOR WITH DLSU GPA CGPA SIGNATURE APPROVED CHECKLIST FLOWCHART LEGEND
""".split())

NONE_MARKERS = frozenset({"", "-", "--", "---", "—", "–", "N/A", "NA", "NONE",
                          "NIL", "#REF!"})

# Requirement markers, as used by the Gokongwei College of Engineering
# checklists:
#   H — hard prerequisite: must have been PASSED before enrolling.
#   S — soft prerequisite: must have been TAKEN, pass or fail. Sitting
#       Differential Calculus and failing it still clears you for Engineering
#       Economics, so this is satisfied by `attempted`, not by `taken`.
#   C — corequisite: taken in the same term (Undergraduate §10.10.1).
PREREQ_TYPES = {"H": "prereq", "S": "soft", "C": "coreq"}

GRADE_VALUES = frozenset({"4.0", "3.5", "3.0", "2.5", "2.0", "1.5", "1.0",
                          "0.0", "9.9", "P", "S", "W", "WP", "WF", "INC",
                          "INP", "CR", "✓"})
PASSING_GRADES = frozenset({"4.0", "3.5", "3.0", "2.5", "2.0", "1.5", "1.0",
                            "P", "S", "CR", "✓"})

ORDINALS = {
    "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5, "SIXTH": 6,
    "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9, "TENTH": 10, "ELEVENTH": 11,
    "TWELFTH": 12, "THIRTEENTH": 13, "FOURTEENTH": 14, "FIFTEENTH": 15,
}
_ORDINAL_RE = re.compile(
    r"^(?P<word>" + "|".join(ORDINALS) + r")\s+TERM\b", re.IGNORECASE)
_NUMERIC_TERM_RE = re.compile(r"^(?:TERM\s+(\d{1,2})|(\d{1,2})(?:ST|ND|RD|TH)\s+TERM)\b",
                              re.IGNORECASE)
_YEAR_TERM_RE = re.compile(
    r"^(?:Y(?P<y>\d)\s*[-/ ]?\s*T(?P<t>\d)|(?P<y2>\d)\s*-\s*(?P<t2>\d))$",
    re.IGNORECASE)

_UNITS_RE = re.compile(r"^\(?\s*(\d{1,2}(?:\.\d)?)\s*\)?$")
_WHITESPACE = re.compile(r"\s+")
_SEPARATORS = re.compile(r"[/,;&+]|\band\b|\bor\b", re.IGNORECASE)


# --- Data ----------------------------------------------------------------------

@dataclass
class RawRow:
    """One extracted table row, before anything is interpreted."""

    cells: list[str]
    page: int
    tier: str


@dataclass
class ColumnRoles:
    """Which column index holds which kind of value, decided by vote."""

    code: int | None = None
    title: int | None = None
    units: int | None = None
    grade: int | None = None
    prereq_type: int | None = None
    prereq: int | None = None
    term: int | None = None
    votes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"code": self.code, "title": self.title, "units": self.units,
                "grade": self.grade, "prereq_type": self.prereq_type,
                "prereq": self.prereq, "term": self.term}


@dataclass
class ExtractionReport:
    """What extraction did, in enough detail for the user to judge it."""

    tier: str
    pages: int
    tables: int
    rows_seen: int
    courses_parsed: int
    columns: dict
    roles: ColumnRoles
    header_row: list[str] | None
    warnings: list[str]
    program_name: str


# --- Cell parsers (pure; unit-tested without a PDF) ----------------------------

def clean_cell(value: str | None) -> str:
    """Collapse whitespace (cells wrap mid-value) and strip."""
    return _WHITESPACE.sub(" ", (value or "").replace("\n", " ")).strip()


def is_course_code(token: str) -> bool:
    """Shape plus vocabulary. Column context is applied by the caller."""
    token = token.strip()
    if not token or token.upper() in HEADER_WORDS:
        return False
    return bool(COURSE_CODE_RE.match(token))


def parse_units(cell: str) -> float | None:
    """'3', '3.0', '(3)', '0' -> float. Junk and totals -> None.

    An Excel '#REF!' is read as nothing at all. Whether the value was
    parenthesised is a separate question — see units_are_credited().
    """
    cell = clean_cell(cell)
    if cell.upper() in NONE_MARKERS:
        return None
    match = _UNITS_RE.match(cell)
    if not match:
        return None
    value = float(match.group(1))
    return value if 0.0 <= value <= MAX_PLAUSIBLE_COURSE_UNITS else None


def units_are_credited(cell: str) -> bool:
    """False when the checklist shows a course's units in parentheses.

    That is how these sheets mark a course that must be taken but does not
    count toward the term's unit load — NSTP and the Lasallian series. The
    term totals confirm the convention: "18 (3)" is 18 credited units plus a
    3-unit non-credit course.
    """
    return not clean_cell(cell).startswith("(")


def parse_grade(cell: str) -> tuple[str | None, bool]:
    """Return (normalized grade, whether it counts as passed).

    A course is taken only when its grade PASSES. Treating any non-empty grade
    cell as "done" would credit 0.0 failures, 9.9 deferrals, INC, and
    withdrawals — and then send the student into a course they are blocked from.
    """
    cell = clean_cell(cell).upper()
    if cell in NONE_MARKERS:
        return None, False
    if cell in {"X", "YES", "OK", "✓", "P", "S", "CR"}:
        return ("✓" if cell in {"X", "YES", "OK", "✓"} else cell), True
    if cell in GRADE_VALUES:
        return cell, cell in PASSING_GRADES
    return None, False


def parse_term_banner(text: str) -> int | None:
    """'FIRST TERM' -> 1, 'TERM 4' -> 4, '2ND TERM' -> 2. Else None."""
    text = clean_cell(text)
    if not text:
        return None
    match = _ORDINAL_RE.match(text)
    if match:
        return ORDINALS[match.group("word").upper()]
    match = _NUMERIC_TERM_RE.match(text)
    if match:
        value = int(match.group(1) or match.group(2))
        return value if 1 <= value <= 15 else None
    return None


def parse_year_term(text: str) -> tuple[int, int] | None:
    """'Y3 T2' or '3-1' -> (year, term)."""
    match = _YEAR_TERM_RE.match(clean_cell(text))
    if not match:
        return None
    year = match.group("y") or match.group("y2")
    term = match.group("t") or match.group("t2")
    return (int(year), int(term)) if year and term else None


def split_codes(cell: str) -> list[str]:
    """Split a prerequisite cell into tokens, in order, preserving case.

    Case is preserved deliberately: column-role voting asks whether a cell's
    tokens LOOK like course codes, and codes are written in capitals. Upper-
    casing here first would make every single-word course title ("Alpha") match
    the code shape, and the title column would win the prerequisite vote.
    Callers that want codes uppercase do that themselves.

    The whole cell is checked against the none-markers BEFORE splitting,
    because "N/A" would otherwise split on its own slash into "N" and "A".
    """
    cleaned = clean_cell(cell)
    if cleaned.upper() in NONE_MARKERS:
        return []

    out, seen = [], set()
    for piece in _SEPARATORS.split(cleaned):
        token = piece.strip()
        if not token or token.upper() in NONE_MARKERS or token.upper() in seen:
            continue
        seen.add(token.upper())
        out.append(token)
    return out


def split_prereq_types(cell: str) -> list[str]:
    """Split an 'H/H' or 'S / H' marker cell into per-code relation types."""
    return [piece.strip().upper()
            for piece in _SEPARATORS.split(clean_cell(cell))
            if piece.strip()]


def pair_requirements(types_cell: str, codes_cell: str
                      ) -> tuple[tuple[str, ...], tuple[str, ...],
                                 tuple[str, ...], str | None]:
    """Split a row's requirements into (hard, soft, coreqs, warning).

    The type column pairs POSITIONALLY with the code column: 'H/H' beside
    'FUNDLEC/LOGDSGN' means both are hard prerequisites; 'H/C' means the first
    must be passed and the second taken alongside; 'S' means the course need
    only have been SAT, not passed. Note this makes '/' an AND separator on
    this sheet, not the disjunction it can be on others — which is why the two
    lists are only trusted when their lengths agree.

    An unpaired list falls back to hard prerequisites, the conservative
    direction: a hard requirement can only delay a course, whereas mistaking a
    hard one for soft would clear a student into a course they cannot take.
    """
    codes = [c.upper() for c in split_codes(codes_cell)]
    if not codes:
        return (), (), (), None

    types = split_prereq_types(types_cell)
    if len(types) != len(codes):
        if types and any(t in PREREQ_TYPES for t in types):
            return tuple(codes), (), (), (
                f"requirement types {types_cell!r} do not line up with "
                f"{codes_cell!r}; all were recorded as hard prerequisites")
        return tuple(codes), (), (), None

    buckets: dict[str, list[str]] = {"prereq": [], "soft": [], "coreq": []}
    for kind, code in zip(types, codes):
        buckets[PREREQ_TYPES.get(kind, "prereq")].append(code)
    return (tuple(buckets["prereq"]), tuple(buckets["soft"]),
            tuple(buckets["coreq"]), None)


def synthetic_code(title: str, used: set[str]) -> str:
    """A stable placeholder code for a row that states no course code."""
    slug = re.sub(r"[^A-Z0-9]+", "_", clean_cell(title).upper()).strip("_")
    base = f"PLACEHOLDER_{slug}" if slug else "PLACEHOLDER"
    candidate, suffix = base, 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


# --- Row extraction ------------------------------------------------------------

def _rows_from_tables(page, settings: dict, tier: str) -> list[RawRow]:
    """Tables on one page, ordered left-to-right so terms read in sequence."""
    found = page.find_tables(settings)
    rows: list[RawRow] = []
    for table in sorted(found, key=lambda t: (round(t.bbox[0]))):
        for cells in table.extract() or []:
            rows.append(RawRow([clean_cell(c) for c in cells],
                               page.page_number, tier))
    return rows


def _rows_from_words(words: list[dict], page_number: int,
                     gap: float = COLUMN_GAP) -> list[RawRow]:
    """Tier C: rebuild rows from word geometry when no grid is detectable.

    Reuses the handbook parser's approach — bucket words onto a shared baseline,
    then split into cells wherever the horizontal gap exceeds `gap`.
    """
    buckets: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        placed = False
        for bucket in buckets:
            if abs(bucket[0]["top"] - word["top"]) <= LINE_TOLERANCE:
                bucket.append(word)
                placed = True
                break
        if not placed:
            buckets.append([word])

    rows: list[RawRow] = []
    for bucket in buckets:
        ordered = sorted(bucket, key=lambda w: w["x0"])
        cells, current, previous_end = [], [], None
        for word in ordered:
            if previous_end is not None and word["x0"] - previous_end > gap:
                cells.append(" ".join(current))
                current = []
            current.append(word["text"])
            previous_end = word["x1"]
        if current:
            cells.append(" ".join(current))
        rows.append(RawRow([clean_cell(c) for c in cells], page_number,
                           "words_x0"))
    return rows


def _course_row_count(rows: list[RawRow]) -> int:
    return sum(1 for row in rows
               if any(is_course_code(cell) for cell in row.cells))


def extract_rows(pdf_path: str | Path) -> tuple[list[RawRow], str, int, int]:
    """Try each tier in order; return (rows, winning tier, pages, tables)."""
    tables_seen = 0
    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = len(pdf.pages)

        for tier, settings in (("table_lines", TABLE_LINES),
                               ("table_text", TABLE_TEXT)):
            rows: list[RawRow] = []
            count = 0
            for page in pdf.pages:
                count += len(page.find_tables(settings))
                rows.extend(_rows_from_tables(page, settings, tier))
            if _course_row_count(rows) >= MIN_COURSE_ROWS:
                log.info("Checklist extraction tier %s: %d row(s) from %d table(s)",
                         tier, len(rows), count)
                return rows, tier, pages, count
            tables_seen = max(tables_seen, count)

        rows = []
        for page in pdf.pages:
            rows.extend(_rows_from_words(
                page.extract_words(extra_attrs=["size"]), page.page_number))
        log.info("Checklist extraction fell back to words_x0: %d row(s)", len(rows))
        return rows, "words_x0", pages, tables_seen


# --- Column-role voting --------------------------------------------------------

def _is_header_row(cells: list[str]) -> bool:
    hits = sum(1 for c in cells if c and c.upper() in HEADER_WORDS)
    return hits >= 2 and not any(is_course_code(c) for c in cells)


def infer_column_roles(rows: list[RawRow]) -> ColumnRoles:
    """Decide each column's role once, by voting across every row.

    Per-cell guessing is what fails on real sheets: units and grades are both
    small decimals, so a column-level decision is far more reliable. The tally
    is kept so the diagnostic can show WHY a column was chosen.
    """
    width = max((len(r.cells) for r in rows), default=0)
    if not width:
        return ColumnRoles()

    def column(index: int) -> list[str]:
        return [r.cells[index] for r in rows
                if len(r.cells) > index and r.cells[index]]

    votes = {"code": {}, "units": {}, "grade": {}, "prereq": {},
             "prereq_type": {}, "term": {}, "title": {}}
    for index in range(width):
        values = column(index)
        if not values:
            continue
        votes["code"][index] = sum(1 for v in values if is_course_code(v))
        votes["units"][index] = sum(1 for v in values
                                    if parse_units(v) is not None)
        votes["grade"][index] = sum(1 for v in values
                                    if v.upper() in GRADE_VALUES)
        votes["prereq_type"][index] = sum(
            1 for v in values
            if all(t in PREREQ_TYPES for t in split_prereq_types(v)))
        votes["prereq"][index] = sum(1 for v in values
                                     if any(is_course_code(c)
                                            for c in split_codes(v)))
        votes["term"][index] = sum(1 for v in values
                                   if parse_term_banner(v) is not None
                                   or parse_year_term(v) is not None)
        votes["title"][index] = round(
            sum(len(v) for v in values) / max(1, len(values)))

    roles = ColumnRoles(votes=votes)
    counts = {i: len(column(i)) for i in range(width)}

    def best(role: str, minimum: float, exclude: set[int]) -> int | None:
        candidates = [
            (index, hits) for index, hits in votes[role].items()
            if index not in exclude and counts.get(index)
            and hits / counts[index] >= minimum
        ]
        if not candidates:
            return None
        # Most hits wins; leftmost breaks the tie, so output is stable.
        return max(candidates, key=lambda kv: (kv[1], -kv[0]))[0]

    used: set[int] = set()
    roles.code = best("code", 0.5, used)
    if roles.code is not None:
        used.add(roles.code)
    roles.prereq = best("prereq", 0.5, used)
    if roles.prereq is not None:
        used.add(roles.prereq)
    roles.prereq_type = best("prereq_type", 0.6, used)
    if roles.prereq_type is not None:
        used.add(roles.prereq_type)
    roles.units = best("units", 0.7, used)
    if roles.units is not None:
        used.add(roles.units)
    roles.grade = best("grade", 0.6, used)
    if roles.grade is not None:
        used.add(roles.grade)
    roles.term = best("term", 0.5, used)
    if roles.term is not None:
        used.add(roles.term)

    remaining = [i for i in range(width) if i not in used and counts.get(i)]
    if remaining:
        roles.title = max(remaining, key=lambda i: votes["title"].get(i, 0))

    # A header row NAMES its columns, which beats any vote.
    for row in rows:
        if _is_header_row(row.cells):
            _apply_header(row.cells, roles)
            break
    return roles


_HEADER_ROLE_WORDS = {
    "COURSE": "code", "CODE": "code",
    "TITLE": "title", "SUBJECT": "title",
    "UNITS": "units", "UNIT": "units", "CREDIT": "units", "CREDITS": "units",
    "GRADE": "grade", "GRADES": "grade",
    "PREREQUISITE": "prereq", "PREREQUISITES": "prereq", "PREREQ": "prereq",
}


def _apply_header(cells: list[str], roles: ColumnRoles) -> None:
    """Let an explicit header override the vote, where it is unambiguous.

    Applied conservatively: a merged 'PREREQUISITES' header spanning the type
    and code columns would otherwise relabel the type column, so a header only
    moves a role that voting did not already fill.
    """
    for index, cell in enumerate(cells):
        role = _HEADER_ROLE_WORDS.get(cell.upper())
        if role and getattr(roles, role) is None:
            setattr(roles, role, index)


# --- Assembly ------------------------------------------------------------------

def _program_name(pdf_path: Path) -> str:
    """Best-effort program title from the largest text near the top of page 1."""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page = pdf.pages[0]
            words = [w for w in page.extract_words(extra_attrs=["size"])
                     if w["top"] < page.height * 0.2]
            if not words:
                return pdf_path.stem
            largest = max(w["size"] for w in words)
            line = [w["text"] for w in sorted(words, key=lambda w: w["x0"])
                    if abs(w["size"] - largest) < 0.2]
            return " ".join(line).strip() or pdf_path.stem
    except Exception:                       # a title is a nicety, never fatal
        return pdf_path.stem


def parse_checklist(pdf_path: str | Path, *, terms_per_year: int = 3
                    ) -> tuple[Curriculum, ExtractionReport]:
    """Read a checklist PDF into a Curriculum plus a report on how it went."""
    pdf_path = Path(pdf_path)
    rows, tier, pages, tables = extract_rows(pdf_path)
    roles = infer_column_roles(rows)
    warnings: list[str] = []

    header_row = next((r.cells for r in rows if _is_header_row(r.cells)), None)
    courses: dict[str, Course] = {}
    current_term: int | None = None
    saw_soft_prereq = False
    saw_grade_column = roles.grade is not None

    def cell(row: RawRow, index: int | None) -> str:
        if index is None or len(row.cells) <= index:
            return ""
        return row.cells[index]

    for row in rows:
        joined = " ".join(c for c in row.cells if c)
        if not joined:
            continue

        banner = next((parse_term_banner(c) for c in row.cells
                       if parse_term_banner(c) is not None), None)
        if banner is not None:
            current_term = banner
            continue
        if _is_header_row(row.cells):
            continue

        code = cell(row, roles.code).upper()
        title = cell(row, roles.title)
        units = parse_units(cell(row, roles.units))

        if code.upper() in HEADER_WORDS:          # TOTAL / SUBTOTAL rows
            continue
        if not is_course_code(code):
            # A titled row carrying units but no code is an elective placeholder.
            # `units is not None` rather than a truth test: a 0-unit row (NSTP,
            # the Lasallian recollection series) is a real requirement.
            if title and units is not None and not code:
                code = synthetic_code(title, set(courses))
            else:
                continue

        if units is None:
            units = 0.0
            warnings.append(
                f"p.{row.page} {code}: no units could be read; recorded as 0. "
                "Set `units:` by hand.")

        prereqs, soft_prereqs, coreqs, pair_warning = pair_requirements(
            cell(row, roles.prereq_type), cell(row, roles.prereq))
        if pair_warning:
            warnings.append(f"p.{row.page} {code}: {pair_warning}")
        if soft_prereqs:
            saw_soft_prereq = True

        grade, passed = parse_grade(cell(row, roles.grade))
        year = term = None
        if current_term is not None:
            year = (current_term - 1) // max(1, terms_per_year) + 1
            term = (current_term - 1) % max(1, terms_per_year) + 1

        if code in courses:
            warnings.append(
                f"p.{row.page} {code} appears more than once; kept the first.")
            continue

        courses[code] = Course(
            code=code,
            title=title or code,
            units=units,
            year=year,
            term=term,
            prereqs=prereqs,
            soft_prereqs=soft_prereqs,
            coreqs=coreqs,
            confidence=(PrereqConfidence.STATED if roles.prereq is not None
                        else PrereqConfidence.UNKNOWN),
            taken=passed,
            grade=grade,
            placeholder=code.startswith("PLACEHOLDER"),
            source_page=row.page,
            credited=units_are_credited(cell(row, roles.units)),
        )

    if saw_soft_prereq:
        warnings.append(
            "Some requirements are marked 'S' (soft): the course must have been "
            "TAKEN, but not necessarily passed. Those are recorded under "
            "`soft_prereqs:` and are satisfied by having sat the course, so a "
            "failed attempt still clears them. Hard 'H' requirements under "
            "`prereqs:` must actually be passed.")
    if not saw_grade_column:
        warnings.append(
            "This checklist has no grade column, so nothing could be marked as "
            "already passed. Type the courses you have passed when /plan asks, "
            "or set `taken: true` on them here.")

    if roles.prereq is not None and any(c.prereqs or c.coreqs
                                        for c in courses.values()):
        source = PrereqSource.COLUMN
    elif any(c.year is not None for c in courses.values()):
        source = PrereqSource.YEAR_TERM
    else:
        source = PrereqSource.NONE

    curriculum = Curriculum(
        program_id=pdf_path.stem.lower(),
        program_name=_program_name(pdf_path),
        terms_per_year=terms_per_year,
        courses=courses,
        prereq_source=source,
        term_caps=derive_term_caps(courses.values(), terms_per_year),
        warnings=warnings,
    )
    report = ExtractionReport(
        tier=tier, pages=pages, tables=tables, rows_seen=len(rows),
        courses_parsed=len(courses), columns=roles.as_dict(), roles=roles,
        header_row=header_row, warnings=warnings,
        program_name=curriculum.program_name,
    )
    return curriculum, report
