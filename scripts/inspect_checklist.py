"""
inspect_checklist.py — Inspect the checklist parser's output (Phase 15).

Purpose:
    Show exactly what was read from a program checklist PDF, and write the
    hand-editable curriculum artifact the planner will use.

Usage:
    uv run python scripts/inspect_checklist.py data/checklists/mine.pdf
    uv run python scripts/inspect_checklist.py data/checklists/mine.pdf --all
    uv run python scripts/inspect_checklist.py data/checklists/mine.pdf --raw --page 2
    uv run python scripts/inspect_checklist.py data/checklists/mine.pdf --force

Inputs:
    A text-based checklist PDF.

Outputs:
    A report on stdout, and data/checklists/<program>.curriculum.yaml
    (never overwritten without --force).

Dependencies:
    src.curriculum.checklist_parser, src.chat.plan_view, rich.

Why this file exists:
    The project values inspectable intermediate artifacts (AD-5), and the
    checklist parser is the one component whose input we do not control
    (AD-8). Before trusting any plan, you need to know whether prerequisites
    were actually found — and if the parser got a column wrong, you need to see
    which, so you can fix it in the YAML. The first three lines of output
    answer that; everything else is detail.
"""

import argparse
import sys

from pathlib import Path

# The report prints section signs, arrows, and en-dashes; without this the
# Windows console encodes them to '?' replacement characters.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console                              # noqa: E402

from src.chat.plan_view import render_extraction_summary      # noqa: E402
from src.curriculum.checklist_parser import (                 # noqa: E402
    TABLE_LINES,
    TABLE_TEXT,
    extract_rows,
    parse_checklist,
)
from src.curriculum.model import (                            # noqa: E402
    CurriculumError,
    write_curriculum_yaml,
)
from src.utils.config import load_settings                    # noqa: E402
from src.utils.logging_setup import setup_logging             # noqa: E402

ROLE_LABEL = {
    "code": "CODE", "title": "TITLE", "units": "UNITS", "grade": "GRADE",
    "prereq_type": "REQ TYPE", "prereq": "PREREQ", "term": "TERM",
}


def _dump_raw(pdf_path: Path, page_number: int) -> None:
    """Show one page's cells under each tier — the analogue of inspect_parse."""
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        if not (1 <= page_number <= len(pdf.pages)):
            raise SystemExit(f"Page {page_number} is out of range "
                             f"(the PDF has {len(pdf.pages)}).")
        page = pdf.pages[page_number - 1]
        print(f"\nPage {page_number}: {page.width:.0f}x{page.height:.0f}pt, "
              f"{len(page.chars)} chars")
        for tier, settings in (("table_lines", TABLE_LINES),
                               ("table_text", TABLE_TEXT)):
            tables = page.find_tables(settings)
            print(f"\n--- {tier}: {len(tables)} table(s) ---")
            for index, table in enumerate(
                    sorted(tables, key=lambda t: round(t.bbox[0]))):
                print(f"  table {index} at x0={table.bbox[0]:.0f}"
                      f" x1={table.bbox[2]:.0f}")
                for row in (table.extract() or [])[:12]:
                    print("   ", [(c or "").replace("\n", " ")[:24] for c in row])


def _print_votes(report) -> None:
    """Show WHY each column was chosen, so a bad vote is visible and fixable."""
    votes = report.roles.votes
    if not votes:
        return
    chosen = {index: role for role, index in report.columns.items()
              if index is not None}
    width = max((max(v) for v in votes.values() if v), default=-1) + 1

    print("\nColumn roles (voted across all rows):")
    for index in range(width):
        label = ROLE_LABEL.get(chosen.get(index), "")
        detail = (f"code={votes['code'].get(index, 0)} "
                  f"units={votes['units'].get(index, 0)} "
                  f"grade={votes['grade'].get(index, 0)} "
                  f"reqtype={votes['prereq_type'].get(index, 0)} "
                  f"prereq={votes['prereq'].get(index, 0)}")
        marker = f"<- {label}" if label else ""
        print(f"  col {index}  {detail:<62} {marker}")
    if report.header_row:
        print(f"  header row seen: {[c for c in report.header_row if c]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect a program checklist PDF and write its curriculum file.")
    ap.add_argument("pdf", help="Path to the checklist PDF.")
    ap.add_argument("--all", action="store_true",
                    help="List every course, not just the first 20.")
    ap.add_argument("--raw", action="store_true",
                    help="Dump raw table cells for one page and exit.")
    ap.add_argument("--page", type=int, default=1,
                    help="Page to dump with --raw (default 1).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing curriculum file.")
    ap.add_argument("--terms-per-year", type=int, default=3,
                    help="3 for a trimester program (default), 2 for semestral.")
    args = ap.parse_args()

    setup_logging()
    settings = load_settings()
    console = Console()

    pdf_path = Path(args.pdf).expanduser()
    if not pdf_path.exists():
        candidate = settings.planner.checklist_dir / args.pdf
        if not candidate.exists():
            raise SystemExit(
                f"Checklist not found: {pdf_path}. Put it in "
                f"{settings.planner.checklist_dir} and try again.")
        pdf_path = candidate

    if args.raw:
        _dump_raw(pdf_path, args.page)
        return

    print(f"Inspecting {pdf_path.name} ...")
    curriculum, report = parse_checklist(pdf_path,
                                         terms_per_year=args.terms_per_year)

    print(f"  {report.pages} page(s), {report.tables} table(s), "
          f"{report.rows_seen} row(s)")
    print(f"  extraction tier: {report.tier}")
    if not curriculum.courses:
        raise SystemExit(
            "No courses could be read from this checklist. Try "
            f"--raw --page 1 to see what the extractor found, or hand-write a "
            "curriculum file (docs/course_planner.md §4).")

    # The shared renderer: the chatbot shows this exact summary during /plan,
    # so the two can never disagree about what was extracted.
    render_extraction_summary(console, curriculum, columns=report.columns)
    _print_votes(report)

    courses = list(curriculum.courses.values())
    shown = courses if args.all else courses[:20]
    print(f"\nCourses ({len(shown)} of {len(courses)}"
          f"{'; --all for every row' if not args.all else ''}):")
    print(f"  {'TERM':<6}{'CODE':<9}{'U':>3}  {'TITLE':<44}{'REQUIRES'}")
    for course in shown:
        term = (f"Y{course.year}T{course.term}"
                if course.year is not None else "-")
        requires = ", ".join(course.prereqs)
        if course.coreqs:
            requires += ("  +with " if requires else "with ") + \
                ", ".join(course.coreqs)
        print(f"  {term:<6}{course.code:<9}{course.units:>3g}  "
              f"{course.title[:44]:<44}{requires}")

    with_pre = sum(1 for c in courses if c.prereqs)
    with_co = sum(1 for c in courses if c.coreqs)
    print(f"\nTotals: {len(courses)} courses, "
          f"{sum(c.units for c in courses):g} units; "
          f"{with_pre} with prerequisites, {with_co} with corequisites")

    yaml_path = (settings.planner.checklist_dir /
                 f"{curriculum.program_id}.curriculum.yaml")
    try:
        written = write_curriculum_yaml(
            curriculum, yaml_path, source_pdf=pdf_path, tier=report.tier,
            columns=report.columns, force=args.force)
    except CurriculumError as exc:
        print(f"\n{exc}")
        return

    print(f"\nWrote {written} ({len(courses)} courses).")
    print("  Edit that file to correct anything above, then run /plan in the "
          "chatbot.\n  Re-running this script will NOT overwrite it (use "
          "--force to regenerate).")


if __name__ == "__main__":
    main()
