"""
html_report.py — Render a study plan as a self-contained HTML page.

Purpose:
    Turn a StudyPlan into one file the student can open in any browser and
    print: terms as columns, each course a card showing its units and what it
    requires, colour-coded by status, with the handbook citations and caveats
    that shaped the plan.

Inputs:
    A StudyPlan, the Curriculum it came from, and the PolicyRules applied.

Outputs:
    render_plan_html() returns the whole document as a string;
    write_plan_html() persists it under planner.plan_dir.

Dependencies:
    html (standard library), src.curriculum.model. No CSS or JS is fetched —
    the file works offline and survives being emailed.

Why this file exists:
    A node-and-arrow diagram was tried first and measured: a third-year
    student's plan is 111 courses and 68 prerequisite edges, which any
    auto-layout engine turns into unreadable spaghetti. The fix is to stop
    drawing the graph and lay out the thing the student actually reads — a
    term-by-term grid — putting each course's requirements on its own card as
    text. Same information, no crossing lines, and it prints.

    No JavaScript, deliberately: a plan is a document, not an app, and a
    static file is one less thing that can fail during a demo or on a printout.
"""

from __future__ import annotations

import html

from pathlib import Path

from src.curriculum.model import (
    Course,
    Curriculum,
    PlannedTerm,
    StudyPlan,
    credited_units,
    total_units,
)

# Status → (css class, human label). Order is the legend's order.
STATUSES = (
    ("taken", "Already passed"),
    ("ready", "Take next term"),
    ("later", "Scheduled later"),
    ("unknown", "No prerequisite info — verify"),
)

_CSS = """
:root{
  --ink:#1e2b1e; --muted:#5b6b5b; --line:#dfe5df; --bg:#ffffff;
  --taken-bg:#e8f5e9; --taken-br:#2e7d32;
  --ready-bg:#e3f2fd; --ready-br:#1565c0;
  --later-bg:#f5f5f5; --later-br:#9e9e9e;
  --unknown-bg:#fff8e1; --unknown-br:#f9a825;
}
*{box-sizing:border-box}
body{margin:0;padding:32px;background:var(--bg);color:var(--ink);
  font:15px/1.5 "Segoe UI",system-ui,-apple-system,sans-serif}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:17px;margin:32px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);font-size:14px;margin:0 0 20px}
.sub b{color:var(--ink)}

.legend{display:flex;flex-wrap:wrap;gap:14px;margin:0 0 24px;font-size:12.5px;color:var(--muted)}
.legend span{display:flex;align-items:center;gap:6px}
.swatch{width:13px;height:13px;border-radius:3px;border-left:3px solid}

.terms{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:16px}
.term{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fcfdfc;
  break-inside:avoid;page-break-inside:avoid}
.term > header{margin-bottom:10px}
.term h3{font-size:14px;margin:0;letter-spacing:.01em}
.term .meta{font-size:11.5px;color:var(--muted);margin-top:2px}

/* Already-passed courses are context, not work: rendering them as full cards
   cost two extra printed pages and pushed the actual plan off page one. They
   get dense chips instead, and the panel is allowed to break across pages. */
.term.done{grid-column:1/-1;break-inside:auto;page-break-inside:auto}
.chips{display:flex;flex-wrap:wrap;gap:5px}
.chip{background:var(--taken-bg);border-left:3px solid var(--taken-br);
  border-radius:4px;padding:3px 7px;font-size:11px;white-space:nowrap}
.chip b{font-size:11.5px;letter-spacing:.02em}
.chip span{color:var(--muted);margin-left:4px;font-variant-numeric:tabular-nums}

.card{border-radius:5px;padding:7px 9px;margin-bottom:7px;border-left:3px solid;
  background:var(--later-bg);border-color:var(--later-br)}
.card:last-child{margin-bottom:0}
.card.taken{background:var(--taken-bg);border-color:var(--taken-br)}
.card.ready{background:var(--ready-bg);border-color:var(--ready-br)}
.card.later{background:var(--later-bg);border-color:var(--later-br)}
.card.unknown{background:var(--unknown-bg);border-color:var(--unknown-br);
  border-left-style:dashed}
.code{font-weight:700;font-size:12.5px;letter-spacing:.02em}
.units{float:right;font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.title{font-size:11.5px;color:var(--muted);margin-top:1px}
.req{font-size:10.5px;color:var(--muted);margin-top:4px;padding-top:4px;
  border-top:1px dotted var(--line)}
.req b{font-weight:600;color:var(--ink)}
.unlocks{font-size:10.5px;color:var(--muted);margin-top:3px}

table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:5px 10px 5px 0;border-bottom:1px solid var(--line);
  vertical-align:top}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}

.rule{margin-bottom:12px}
.rule .stmt{font-size:13.5px}
.rule .cite{font-size:11.5px;color:var(--muted)}
.rule .excerpt{font-size:11.5px;color:var(--muted);font-style:italic;
  border-left:2px solid var(--line);padding-left:9px;margin-top:4px}
.missing{color:#a9600a}

ul.notes{margin:0;padding-left:18px;font-size:13px}
ul.notes li{margin-bottom:7px}
.disclaimer{margin-top:28px;padding-top:12px;border-top:1px solid var(--line);
  font-size:12px;color:var(--muted)}

@media print{
  body{padding:0;font-size:11pt}
  h2{margin-top:18px}
  .terms{grid-template-columns:repeat(3,1fr);gap:10px}
  .card{background:#fff !important}
}
"""


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def status_of(course: Course, term_index: int, passed: set[str]) -> str:
    """Which style a course card gets.

    Precedence is taken > unknown > ready > later. `unknown` outranks the
    scheduling statuses on purpose: the colour's job is to warn that the
    course's position was guessed, and a confident "take this next term" on a
    course with no known prerequisites would be a lie.
    """
    if term_index == 0 or course.code in passed:
        return "taken"
    if course.confidence.value == "unknown":
        return "unknown"
    return "ready" if term_index == 1 else "later"


def _requirement_line(course: Course, known: set[str]) -> str:
    """"needs / after / with" for one card — the arrows, written as text."""
    parts: list[str] = []
    hard = [c for c in course.prereqs if c in known]
    soft = [c for c in course.soft_prereqs if c in known]
    co = [c for c in course.coreqs if c in known]
    if hard:
        parts.append(f"<b>needs</b> {_esc(', '.join(hard))}")
    if soft:
        # Spelled out because "after" vs "needs" is the distinction that
        # decides whether a failed attempt still clears the requirement.
        parts.append(f"<b>after</b> {_esc(', '.join(soft))}")
    if co:
        parts.append(f"<b>with</b> {_esc(', '.join(co))}")
    return " · ".join(parts)


def _card(course: Course, status: str, known: set[str],
          downstream: dict[str, int] | None) -> str:
    units = (f"{course.units:g}u" if course.credited
             else f"({course.units:g}u)")
    out = [f'<div class="card {status}">',
           f'<span class="units">{_esc(units)}</span>',
           f'<div class="code">{_esc(course.code)}</div>',
           f'<div class="title">{_esc(course.title)}</div>']
    req = _requirement_line(course, known)
    if req:
        out.append(f'<div class="req">{req}</div>')
    unlocks = (downstream or {}).get(course.code, 0)
    if unlocks:
        out.append(f'<div class="unlocks">unlocks {unlocks} later course'
                   f'{"" if unlocks == 1 else "s"}</div>')
    out.append("</div>")
    return "".join(out)


def _term_panel(heading: str, meta: str, cards: str,
                extra_class: str = "") -> str:
    return (f'<section class="term{extra_class}"><header><h3>{heading}</h3>'
            f'<div class="meta">{meta}</div></header>{cards}</section>')


def _course_table(title: str, curriculum: Curriculum, codes: list[str]) -> str:
    if not codes:
        return ""
    rows = []
    for code in codes:
        course = curriculum.courses.get(code)
        if course is None:
            rows.append(f"<tr><td>{_esc(code)}</td><td></td><td class='num'></td></tr>")
            continue
        rows.append(f"<tr><td><b>{_esc(course.code)}</b></td>"
                    f"<td>{_esc(course.title)}</td>"
                    f"<td class='num'>{course.units:g}</td></tr>")
    return (f"<h2>{_esc(title)}</h2><table><thead><tr><th>Code</th>"
            f"<th>Title</th><th class='num'>Units</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def render_plan_html(plan: StudyPlan, curriculum: Curriculum,
                     policy: list | None = None, *,
                     include_taken: bool = True,
                     taken: set[str] | frozenset[str] | None = None,
                     downstream: dict[str, int] | None = None) -> str:
    """Render the whole plan as one self-contained HTML document.

    Args:
        policy: Rules applied, duck-typed on .statement/.citation/.excerpt.
        taken: Codes to treat as passed; None falls back to the curriculum's
            own `taken` flags (a checklist with no grade column carries none).
        downstream: Optional code → count of courses it unlocks, shown per card.
    """
    passed = ({str(c).upper() for c in taken} if taken is not None
              else {c.code for c in curriculum.courses.values() if c.taken})
    known = set(curriculum.courses)
    taken_courses = [c for c in curriculum.courses.values() if c.code in passed]
    remaining = [c for c in curriculum.courses.values() if c.code not in passed]

    panels: list[str] = []
    if include_taken and taken_courses:
        chips = "".join(
            f'<span class="chip"><b>{_esc(c.code)}</b>'
            f'<span>{c.units:g}u</span></span>' for c in taken_courses)
        panels.append(_term_panel(
            "Already passed",
            f"{len(taken_courses)} courses · {total_units(taken_courses):g} units",
            f'<div class="chips">{chips}</div>', extra_class=" done"))

    for term in plan.terms:
        cards = "".join(_card(c, status_of(c, term.index, passed), known, downstream)
                        for c in term.courses)
        cap = f" of {term.cap:g}" if term.cap else ""
        source = (f" · checklist term {term.checklist_term}"
                  if term.checklist_term else "")
        panels.append(_term_panel(
            _esc(term.label),
            f"{term.units:g}{cap} units{source}", cards))

    # Say so explicitly rather than just showing an absence: a student with
    # nothing left should read "you're done", not an empty grid.
    nothing_left = ""
    if not plan.terms:
        nothing_left = ('<p class="sub"><b>Nothing to plan</b> — no remaining '
                        'course could be scheduled. If that is unexpected, '
                        'check the caveats below.</p>')
        if not panels:
            panels.append(_term_panel("Nothing to plan",
                                      "every course is already passed", ""))

    legend = "".join(
        f'<span><i class="swatch" style="background:var(--{k}-bg);'
        f'border-color:var(--{k}-br)"></i>{_esc(label)}</span>'
        for k, label in STATUSES)

    head = (f"{total_units(taken_courses):g} of "
            f"{total_units(curriculum.courses.values()):g} units completed"
            f" · <b>{len(remaining)}</b> course"
            f"{'' if len(remaining) == 1 else 's'} remaining"
            f" · {len(plan.terms)} term"
            f"{'' if len(plan.terms) == 1 else 's'} planned")

    body = [
        f"<h1>{_esc(curriculum.program_name)}</h1>",
        f'<p class="sub">{head}</p>',
        f'<div class="legend">{legend}</div>',
        nothing_left,
        f'<div class="terms">{"".join(panels)}</div>',
        _course_table("Eligible now, but over the unit cap",
                      curriculum, plan.deferred),
        _course_table("Still blocked", curriculum, plan.blocked),
        _course_table("Cannot be scheduled", curriculum, plan.unreachable),
    ]

    if policy:
        rules = []
        for rule in policy:
            cite = getattr(rule, "citation", None)
            cite_html = (f'<div class="cite">{_esc(cite)}</div>' if cite else
                         '<div class="cite missing">handbook citation '
                         'unavailable — the index may be stale</div>')
            excerpt = getattr(rule, "excerpt", "")
            excerpt_html = (f'<div class="excerpt">{_esc(excerpt)}</div>'
                            if excerpt else "")
            rules.append(f'<div class="rule">'
                         f'<div class="stmt">{_esc(rule.statement)}</div>'
                         f'{cite_html}{excerpt_html}</div>')
        body.append("<h2>Constraints applied</h2>" + "".join(rules))

    if plan.notes:
        items = "".join(f"<li>{_esc(n)}</li>" for n in plan.notes)
        body.append(f'<h2>Caveats</h2><ul class="notes">{items}</ul>')

    body.append('<p class="disclaimer">This is a schedule, not advice. It does '
                'not know which courses are actually offered next term, whether '
                'sections conflict, or what your department recommends. Confirm '
                'with your adviser before enrolling.</p>')

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_esc(curriculum.program_name)} — study plan</title>"
        f"<style>{_CSS}</style></head><body>\n"
        + "\n".join(p for p in body if p)
        + "\n</body></html>\n"
    )


def write_plan_html(text: str, path: Path | str) -> Path:
    """Persist the document, creating parent directories. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
