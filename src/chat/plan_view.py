"""
plan_view.py — Terminal rendering and the interactive /plan flow.

Purpose:
    Everything the course planner needs from the terminal: prompting for a
    checklist, showing what extraction made of it, confirming what the student
    has passed, and rendering the resulting plan with its handbook citations.

Inputs:
    A rich Console, a ChatEngine, and Settings; then keyboard input.

Outputs:
    Terminal output, and an HTML study plan written under planner.plan_dir.

Dependencies:
    rich, src.chat.core, src.curriculum.*, src.utils.config.

Why this file exists:
    The same separation terminal.py exists for (AC-4): ChatEngine.plan_courses
    is one pure call a future GUI can reuse, so every prompt, table, and panel
    lives here instead. render_extraction_summary is deliberately shared with
    scripts/inspect_checklist.py, so the diagnostic script and the chatbot can
    never disagree about what was extracted from a checklist.
"""

from __future__ import annotations

import logging
import re

from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.chat.core import ChatEngine, CoursePlan
from src.curriculum.model import (
    Course,
    Curriculum,
    CurriculumError,
    PrereqSource,
    load_curriculum_yaml,
    remaining_courses,
    taken_codes,
    total_units,
    write_curriculum_yaml,
)
from src.curriculum.html_report import render_plan_html, write_plan_html
from src.curriculum.policy import MISSING_CITATION_NOTE
from src.utils.config import Settings

log = logging.getLogger(__name__)

CANCEL_WORDS = {"cancel", "quit", "exit", ":q"}

_SOURCE_BANNER = {
    PrereqSource.COLUMN: None,          # the normal case needs no warning
    PrereqSource.YEAR_TERM: (
        "Your checklist states no prerequisites, so the order below follows its "
        "own year/term layout. Nothing here claims a course unlocks another."
    ),
    PrereqSource.NONE: (
        "No prerequisite or year/term information could be read from your "
        "checklist, so this is a list of what remains — not an ordering."
    ),
}


def parse_code_list(text: str) -> list[str]:
    """Split typed course codes on commas, whitespace, or both.

    Deduplicated but order-preserving, so the echo back to the user reads in the
    order they typed.
    """
    out, seen = [], set()
    for raw in text.replace(",", " ").split():
        code = raw.strip().upper()
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def parse_term_ranges(text: str, highest: int) -> set[int]:
    """'1-6', '1-5,7', '1 2 3' -> the set of term numbers meant.

    Marking a whole term at a time is the only humane way to record two years
    of history: a student says "I've finished up to term six", not fifty-five
    course codes. Out-of-range and reversed values are dropped rather than
    raising — the caller reports what was actually matched, so a typo is
    visible in the confirmation rather than fatal.
    """
    # Fold the spelled-out forms and any spacing into a bare "1-3" before
    # splitting, so "1 to 3" and "1 - 3" survive the whitespace split intact.
    text = re.sub(r"\s*(?:-|–|—|\bto\b|\bthrough\b)\s*", "-", text,
                  flags=re.IGNORECASE)

    terms: set[int] = set()
    for piece in text.replace(",", " ").split():
        piece = piece.strip()
        if not piece:
            continue
        match = re.fullmatch(r"(\d{1,2})-(\d{1,2})", piece)
        if match:
            low, high = int(match.group(1)), int(match.group(2))
            if low > high:
                low, high = high, low
            terms.update(t for t in range(low, high + 1) if 1 <= t <= highest)
        elif piece.isdigit():
            value = int(piece)
            if 1 <= value <= highest:
                terms.add(value)
    return terms


def courses_in_terms(curriculum: Curriculum, terms: set[int]) -> set[str]:
    """Codes of every course the checklist places in the given terms."""
    return {
        course.code for course in curriculum.courses.values()
        if course.term_index(curriculum.terms_per_year) in terms
    }


# --- Extraction summary (shared with scripts/inspect_checklist.py) -------------

def render_extraction_summary(console: Console, curriculum: Curriculum,
                              yaml_path: Path | None = None,
                              columns: dict | None = None) -> None:
    """Show what was read from a checklist, and how confident that reading is.

    The three lines that matter come first — prerequisite source, year/term
    grouping, and how many courses were read as already passed — because they
    tell the user which of the three extraction cases they are in without
    reading anything else (docs/course_planner.md §4.1).
    """
    courses = curriculum.courses
    passed = taken_codes(curriculum)
    remaining = remaining_courses(curriculum)
    with_prereqs = sum(1 for c in courses.values() if c.prereqs)
    placed = sum(1 for c in courses.values()
                 if c.year is not None and c.term is not None)

    console.print()
    console.print(Panel.fit(
        f"[bold]{curriculum.program_name}[/bold]\n"
        f"{len(courses)} course(s), {total_units(courses.values()):g} units total",
        border_style="blue"))

    source = curriculum.prereq_source
    style = "green" if source is PrereqSource.COLUMN else "yellow"
    console.print(f"\n[bold]PREREQUISITE SOURCE:[/bold] [{style}]{source.value}"
                  f"[/{style}]   ({with_prereqs} of {len(courses)} course(s) have "
                  "stated prerequisites)")
    grouping = (f"found ({placed} of {len(courses)} placed)" if placed
                else "[yellow]none[/yellow]")
    console.print(f"[bold]YEAR/TERM GROUPING: [/bold] {grouping}")
    console.print(f"[bold]ALREADY TAKEN:      [/bold] {len(passed)} course(s) "
                  f"({total_units(c for c in courses.values() if c.taken):g} units); "
                  f"{len(remaining)} remaining "
                  f"({total_units(remaining):g} units)")

    if columns:
        shown = ", ".join(f"{role}={index}" for role, index in columns.items()
                          if index is not None)
        console.print(f"\n[dim]Columns read: {shown}[/dim]")

    if curriculum.warnings:
        console.print(f"\n[yellow]Warnings ({len(curriculum.warnings)}):[/yellow]")
        for warning in curriculum.warnings[:10]:
            console.print(f"  [yellow]- {warning}[/yellow]")
        if len(curriculum.warnings) > 10:
            console.print(f"  [yellow]...and {len(curriculum.warnings) - 10} "
                          "more (see the curriculum file).[/yellow]")

    if yaml_path is not None:
        console.print(f"\n[dim]Curriculum file: {yaml_path}[/dim]")
        console.print("[dim]That file, not the PDF, is what the planner reads — "
                      "edit it to correct anything above.[/dim]")


def render_course_catalog(console: Console, curriculum: Curriculum,
                          marked: set[str] | None = None) -> None:
    """Print the whole program, one line per term.

    Codes rather than titles: this list exists so the student can find the term
    they finished and the codes they need to name, and 103 titles would bury
    both. Titles are a `scripts/inspect_checklist.py --all` away.
    """
    marked = marked or set()
    per_year = max(1, curriculum.terms_per_year)
    by_term: dict[int | None, list[Course]] = {}
    for course in curriculum.courses.values():
        by_term.setdefault(course.term_index(per_year), []).append(course)

    console.print("\n[bold]The program[/bold] "
                  "[dim](units in brackets are not counted toward the load)[/dim]")
    for index in sorted(k for k in by_term if k is not None):
        courses = by_term[index]
        year, term = (index - 1) // per_year + 1, (index - 1) % per_year + 1
        cap = curriculum.term_caps.get(index)
        header = (f"[bold cyan]Term {index:>2}[/bold cyan] [dim](Y{year}T{term})[/dim]"
                  f"  [dim]{cap:g}u[/dim]" if cap else
                  f"[bold cyan]Term {index:>2}[/bold cyan] [dim](Y{year}T{term})[/dim]")
        items = []
        for course in courses:
            units = (f"{course.units:g}" if course.credited
                     else f"[{course.units:g}]")
            tick = "[green]✓[/green]" if course.code in marked else " "
            items.append(f"{tick}{course.code} {units}")
        console.print(f"  {header}  " + "  ".join(items))

    unplaced = by_term.get(None)
    if unplaced:
        console.print("  [bold cyan]No term[/bold cyan]  "
                      + "  ".join(c.code for c in unplaced))


def _pick_taken(console: Console,
                curriculum: Curriculum) -> tuple[set[str], set[str]] | None:
    """Ask what is done. Returns (passed, attempted), or None if cancelled.

    Three questions, in the order that does the most work first: whole terms,
    then extras, then exceptions. The third is what makes the common real case
    expressible — "I finished terms 1-5 but failed one of them" — because a
    term range alone cannot say that.

    That third answer carries more than a correction: a course inside a
    completed term that was not passed was still SAT, and a soft ("S")
    prerequisite is satisfied by having sat a course even after failing it. So
    it feeds `attempted` as well as being removed from `passed`.
    """
    per_year = max(1, curriculum.terms_per_year)
    indexes = [c.term_index(per_year) for c in curriculum.courses.values()]
    highest = max([i for i in indexes if i is not None], default=0)

    taken: set[str] = taken_codes(curriculum)      # anything the file already flags
    if not highest:
        console.print("\n[yellow]This checklist has no term grouping, so I "
                      "cannot offer whole terms.[/yellow]")
    else:
        console.print(f"\n[bold]Which terms have you completed?[/bold] "
                      f"[dim](1-{highest}; e.g. \"1-6\" or \"1-5,7\". "
                      f"Enter if none)[/dim]")
        try:
            raw = console.input("[bold cyan]Terms completed:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return None
        if raw.lower() in CANCEL_WORDS:
            console.print("[dim]Cancelled.[/dim]")
            return None

        terms = parse_term_ranges(raw, highest)
        if raw and not terms:
            console.print(f"[yellow]I couldn't read any term numbers in "
                          f"{raw!r} — treating it as none.[/yellow]")
        if terms:
            marked = courses_in_terms(curriculum, terms)
            taken |= marked
            console.print(f"  [green]✓[/green] {len(marked)} course(s) marked "
                          f"from term(s) {_summarize(terms)}")

    try:
        added = console.input(
            "\n[bold cyan]Anything else you've passed?[/bold cyan] "
            "[dim](codes, or Enter to skip)[/dim] ").strip()
        removed = console.input(
            "[bold cyan]Anything in there you HAVEN'T passed?[/bold cyan] "
            "[dim](e.g. a failed course — codes, or Enter)[/dim] ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None

    known = set(curriculum.courses)
    extra = set(parse_code_list(added))
    drop = set(parse_code_list(removed))

    unknown = sorted((extra | drop) - known)
    if unknown:
        console.print(f"[yellow]Not on your checklist, so ignored: "
                      f"{', '.join(unknown)}[/yellow]")

    # Removals win: the student correcting us is the most reliable signal there
    # is, and a failed course must go back into the plan as a retake.
    retakes = sorted(drop & known)
    taken = (taken | (extra & known)) - drop
    # Sat but not passed — everything marked done, plus the failures.
    attempted = taken | set(retakes)

    if retakes:
        console.print(f"  [yellow]↻[/yellow] {', '.join(retakes)} will be "
                      "planned again. Anything needing them passed stays "
                      "blocked; anything that only needs them [italic]taken[/italic] "
                      "(a soft requisite) is already cleared.")

    console.print(f"\n[dim]Planning with {len(taken)} passed course(s), "
                  f"{len(known) - len(taken)} to go.[/dim]")
    return taken, attempted


def _summarize(terms: set[int]) -> str:
    """'1,2,3,5' -> '1-3, 5' so the confirmation reads like the input."""
    out, run = [], []
    for term in sorted(terms):
        if run and term == run[-1] + 1:
            run.append(term)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
        run = [term]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
    return ", ".join(out)


# --- Plan rendering ------------------------------------------------------------

def _term_table(term, downstream: dict[str, int]) -> Table:
    # The cap is whatever governed THIS term — usually the checklist's own
    # prescribed load for it, which for an engineering program runs well above
    # the general 15 (§10.2 defers to the checklist).
    limit = f" of {term.cap:g}" if term.cap else ""
    source = (f"  [dim](checklist term {term.checklist_term})[/dim]"
              if term.checklist_term else "")
    table = Table(box=box.SIMPLE, title_justify="left",
                  title=f"[bold]{term.label}[/bold] — "
                        f"{term.units:g}{limit} units{source}")
    table.add_column("Code", style="cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Units", justify="right")
    table.add_column("Unlocks", justify="right", style="dim")
    if any(not c.credited for c in term.courses):
        table.caption = "(n) = required but not counted toward the unit load"
        table.caption_justify = "left"

    for course in term.courses:
        unlocks = downstream.get(course.code, 0)
        title = course.title + (" [dim](placeholder)[/dim]"
                               if course.placeholder else "")
        # Parenthesised for a non-credit course, exactly as the checklist writes
        # it — otherwise the column would not add up to the term total.
        units = (f"{course.units:g}" if course.credited
                 else f"[dim]({course.units:g})[/dim]")
        table.add_row(course.code, title, units,
                      str(unlocks) if unlocks else "")
    return table


def _simple_table(title: str, curriculum: Curriculum,
                  codes: list[str], style: str = "") -> Table:
    table = Table(box=box.SIMPLE, title_justify="left", title=title)
    table.add_column("Code", style=style or "cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Units", justify="right")
    for code in codes:
        course = curriculum.courses.get(code)
        if course is None:
            table.add_row(code, "", "")
        else:
            table.add_row(code, course.title, f"{course.units:g}")
    return table


def render_plan(console: Console, result: CoursePlan,
                artifact_path: Path | None = None) -> None:
    """Print the plan, the constraints behind it, and every caveat."""
    plan, curriculum = result.plan, result.curriculum

    if result.error:
        console.print(f"\n[red]{result.error}[/red]")
        return

    # Report against the codes the PLANNER treated as passed, not the artifact's
    # own flags: a checklist with no grade column carries none, and the student
    # supplies them at the prompt instead.
    passed_codes = result.taken or {c.code for c in curriculum.courses.values()
                                    if c.taken}
    passed = [c for c in curriculum.courses.values() if c.code in passed_codes]
    remaining = [c for c in curriculum.courses.values()
                 if c.code not in passed_codes]
    console.print()
    console.print(Panel.fit(
        f"[bold]{curriculum.program_name}[/bold]\n"
        f"{total_units(passed):g} of "
        f"{total_units(curriculum.courses.values()):g} units completed  ·  "
        f"{len(remaining)} course(s) remaining\n"
        f"prerequisite source: {curriculum.prereq_source.value}",
        border_style="blue"))

    banner = _SOURCE_BANNER.get(curriculum.prereq_source)
    if banner:
        console.print(f"\n[yellow]{banner}[/yellow]")

    if not plan.terms:
        console.print("\n[yellow]There is nothing left to schedule.[/yellow]")
    else:
        # "Unlocks" is recomputed here rather than threaded through StudyPlan:
        # it is presentation, and the planner's own copy is an internal detail.
        downstream = _downstream_for_display(curriculum, passed_codes)
        for term in plan.terms:
            console.print()
            console.print(_term_table(term, downstream))

    if plan.deferred:
        console.print()
        console.print(_simple_table(
            "[bold]Eligible now, but over the unit cap[/bold] — "
            "swap any of these into next term if you prefer",
            curriculum, plan.deferred))

    if plan.blocked:
        console.print()
        console.print(_simple_table("[bold]Still blocked[/bold]", curriculum,
                                    plan.blocked, style="yellow"))

    if plan.unreachable:
        console.print()
        console.print(_simple_table(
            "[bold]Cannot be scheduled[/bold] — see the caveats below",
            curriculum, plan.unreachable, style="red"))

    if result.policy:
        console.print("\n[dim]Constraints applied — and the provision each one "
                      "comes from:[/dim]")
        for rule in result.policy:
            console.print(f"  [dim]- {rule.statement}[/dim]")
            if rule.citation:
                console.print(f"    [dim]{rule.citation}[/dim]")
            else:
                console.print(f"    [yellow]{MISSING_CITATION_NOTE}[/yellow]")

    if plan.notes:
        console.print("\n[yellow]Caveats:[/yellow]")
        for note in plan.notes:
            console.print(f"  [yellow]- {note}[/yellow]")

    console.print("\n[dim]This is a schedule, not advice: it does not know what "
                  "is actually offered next term. Confirm with your adviser."
                  "[/dim]")

    if artifact_path is not None:
        console.print(f"\nPlan written to [bold]{artifact_path}[/bold]")
        console.print("[dim]Open it in any browser — it prints cleanly too."
                      "[/dim]")


def _downstream_for_display(curriculum: Curriculum,
                            passed: set[str] | frozenset[str]) -> dict[str, int]:
    """Transitive dependent counts over the untaken part of the checklist."""
    from src.curriculum.planner import downstream_counts

    edges = {
        code: {p for p in course.prereqs
               if p not in passed and p in curriculum.courses}
        for code, course in curriculum.courses.items() if code not in passed
    }
    try:
        return downstream_counts(edges)
    except Exception:            # a cycle here is already reported in the plan
        return {}


# --- The interactive flow ------------------------------------------------------

def _discover_checklists(settings: Settings) -> list[Path]:
    """Checklists already sitting in the checklist directory.

    Curriculum artifacts first: if one exists it is the corrected, authoritative
    version of its PDF (AD-8), and re-parsing the PDF would discard the user's
    edits. A PDF is only offered when nothing has been extracted from it yet.
    """
    directory = settings.planner.checklist_dir
    if not directory.exists():
        return []
    artifacts = sorted(directory.glob("*.curriculum.yaml"))
    extracted = {path.name[: -len(".curriculum.yaml")].lower()
                 for path in artifacts}
    pdfs = [path for path in sorted(directory.glob("*.pdf"))
            if path.stem.lower() not in extracted]
    return artifacts + pdfs


def _choose_checklist(console: Console, settings: Settings) -> Path | None:
    """Pick the checklist to plan from, asking only when there is a choice."""
    found = _discover_checklists(settings)
    if not found:
        console.print(
            f"\n[red]No checklist found in {settings.planner.checklist_dir}."
            "[/red]\nDrop your program checklist PDF there, then run "
            "[bold]scripts/inspect_checklist.py[/bold] on it (or just run "
            "/plan again).")
        return None
    if len(found) == 1:
        return found[0]

    console.print("\n[bold]Which checklist?[/bold]")
    for number, path in enumerate(found, start=1):
        console.print(f"  [cyan]{number}[/cyan]  {path.name}")
    try:
        raw = console.input("\n[bold cyan]Number (Enter for 1):[/bold cyan] ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None
    if raw.lower() in CANCEL_WORDS:
        return None
    if not raw:
        return found[0]
    if raw.isdigit() and 1 <= int(raw) <= len(found):
        return found[int(raw) - 1]
    console.print(f"[yellow]'{raw}' isn't one of those — using "
                  f"{found[0].name}.[/yellow]")
    return found[0]


def run_plan(console: Console, engine: ChatEngine, settings: Settings) -> None:
    """Show the program, ask what is already done, and plan the rest.

    Returns quietly on cancellation or any expected failure; the caller's REPL
    loop must survive whatever happens here.
    """
    path = _choose_checklist(console, settings)
    if path is None:
        return

    curriculum, yaml_path, columns = _load_or_extract(console, path, settings)
    if curriculum is None:
        return

    console.print()
    console.print(Panel.fit(
        f"[bold]{curriculum.program_name}[/bold]\n"
        f"{len(curriculum.courses)} courses · "
        f"{total_units(curriculum.courses.values()):g} units · "
        f"{len(curriculum.term_caps) or '?'} terms",
        border_style="blue"))
    if curriculum.warnings:
        console.print(f"[dim]{len(curriculum.warnings)} extraction warning(s) — "
                      f"see {yaml_path.name if yaml_path else 'the curriculum file'}"
                      "[/dim]")

    render_course_catalog(console, curriculum)

    picked = _pick_taken(console, curriculum)
    if picked is None:
        return
    taken, attempted = picked

    result = engine.plan_courses(curriculum, taken, attempted)
    artifact = _write_plan_page(console, result, settings)
    render_plan(console, result, artifact)

    if yaml_path is not None:
        console.print(f"[dim]Anything wrong above? Edit {yaml_path} and run "
                      "/plan again.[/dim]")


def _load_or_extract(console: Console, path: Path, settings: Settings):
    """Return (curriculum, yaml_path, columns) — extracting from a PDF if needed."""
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            return load_curriculum_yaml(path), path, None
        except CurriculumError as exc:
            console.print(f"\n[red]{exc}[/red]")
            return None, None, None

    if path.suffix.lower() != ".pdf":
        console.print(f"[red]I don't know how to read {path.suffix or 'that'} "
                      "files.[/red] Give me a .pdf checklist or a .yaml "
                      "curriculum file.")
        return None, None, None

    try:
        from src.curriculum.checklist_parser import parse_checklist
    except ImportError:
        console.print("[red]Checklist PDF parsing is not available in this "
                      "build.[/red] Hand-write a curriculum YAML instead — the "
                      "format is documented in docs/course_planner.md §4.")
        return None, None, None

    yaml_path = (settings.planner.checklist_dir /
                 f"{path.stem}.curriculum.yaml")
    if yaml_path.exists():
        # Never silently re-parse over corrections the user already made (AD-8).
        console.print(f"\n[dim]Using your existing {yaml_path.name} rather than "
                      "re-reading the PDF.[/dim]")
        console.print("[dim]Delete it, or run scripts/inspect_checklist.py "
                      "--force, to extract again.[/dim]")
        try:
            return load_curriculum_yaml(yaml_path), yaml_path, None
        except CurriculumError as exc:
            console.print(f"\n[red]{exc}[/red]")
            return None, None, None

    with console.status("[dim]Reading the checklist...[/dim]"):
        try:
            curriculum, report = parse_checklist(path)
        except Exception as exc:                    # a PDF we cannot read at all
            log.exception("Checklist parsing failed")
            console.print(f"\n[red]I could not read {path.name}: {exc}[/red]")
            return None, None, None

    columns = getattr(report, "columns", None)
    try:
        write_curriculum_yaml(curriculum, yaml_path, source_pdf=path,
                              tier=getattr(report, "tier", "unknown"),
                              columns=columns)
    except CurriculumError as exc:
        console.print(f"\n[yellow]{exc}[/yellow]")
    return curriculum, yaml_path, columns


def _write_plan_page(console: Console, result: CoursePlan,
                     settings: Settings) -> Path | None:
    """Write the printable HTML plan. Returns its path, or None on failure."""
    try:
        passed = result.taken or {c.code for c in result.curriculum.courses.values()
                                  if c.taken}
        text = render_plan_html(
            result.plan, result.curriculum, result.policy,
            include_taken=settings.planner.include_taken,
            taken=result.taken or None,
            downstream=_downstream_for_display(result.curriculum, passed))
        path = (settings.planner.plan_dir /
                f"{result.curriculum.program_id}-plan.html")
        return write_plan_html(text, path)
    except OSError as exc:
        console.print(f"\n[yellow]Could not write the plan page: {exc}[/yellow]")
        return None
