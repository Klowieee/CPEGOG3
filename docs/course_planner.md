# Course Planner (Phase 15)
## DLSU Student Handbook RAG Chatbot

**Prepared by:** Curriculum Systems Engineer — v1.0, 26 July 2026

---

## 1. Problem and Scope

The chatbot answers questions *about* the handbook. It could not answer the
question students actually arrive with:

> *Given what I have already passed, what can I take next, and in what order?*

**In scope.** Read a program checklist, establish which courses are already
passed, and produce (a) the set of courses eligible next term, (b) a term-by-term
ordering of everything remaining that respects prerequisites and the unit cap,
and (c) a prerequisite flowchart. Every unit limit applied is shown with a real
handbook citation.

**Out of scope, deliberately.** Whether a course is actually *offered* next term;
section/schedule conflicts; approval workflows; picking an actual course for an
elective slot; academic standing under §10.17. See `limitations.md`.

## 2. Why the Checklist Is a Separate Data Source

The handbook states policy *about* checklists and flowcharts but contains no
curriculum. Verified against the built index (`data/processed/chunks.jsonl`):

| Provision | Chunk | What it gives us |
|---|---|---|
| §10.1 | `_0093` | "at least 12 academic units" — the full-time floor |
| §10.2 | `_0093` | "maximum academic load ... is 15 units, **or the number of units indicated on the program checklist**" |
| §10.10.1 | `_0095` | "The laboratory course is a co-requisite of the corresponding lecture course, both should be taken during the same term" |
| §10.19 | `_0099` | "required to complete the two (2) NSTP courses based on their flowchart" |

What the handbook does **not** contain: any course code (beyond `ENG501M`/
`ENG502M` as examples in §17), any curriculum, any prerequisite. Searching the
whole corpus for `GEE`/"General Education" returns nothing.

So the feature splits along that seam — **the checklist supplies the courses; the
handbook index supplies the rules and the citations.** This is why the planner is
a third pipeline (`architecture.md` §5) rather than a new prompt: no amount of
retrieval over the handbook can produce a curriculum it does not contain.

## 3. Extraction Strategy

A MyLaSalle checklist export's layout is outside our control and varies by
program and college. Three tiers are tried in order; the first that yields ≥10
code-bearing rows wins, and the winner is recorded in the artifact so the user
knows which path ran.

| Tier | Method | When it works |
|---|---|---|
| A `table_lines` | `page.extract_tables()`, lines strategy | the sheet has ruled cells |
| B `table_text` | same, text strategy | whitespace-aligned columns |
| C `words_x0` | `extract_words()`, bucket by `top` (tolerance 3.0, as `parser._reconstruct_lines`), split cells at x-gaps | no detectable grid at all |

Tier C deliberately reuses the geometric approach `src/ingestion/parser.py`
already proved on the handbook, rather than inventing a second one.

### 3.1 Course-code recognition: three signals, not a regex

A regex alone cannot distinguish `GEMATMW` from `REMARKS` — both are 7 uppercase
letters. So recognition mirrors `chunker._provision_matches_section`, where
pattern plus *context* beats pattern alone:

1. **Shape.** `^(?=[A-Z0-9]{5,9}$)[A-Z]{2,7}(?:[0-9]{1,4}[A-Z]?|[A-Z]{0,5}[0-9]?)$`
   — intentionally loose. It matches `GEMATMW`, `CSMATH2`, `LBYCPA1`, `ENG501M`,
   and also `STUDENT`. That is fine.
2. **Vocabulary stoplist.** `HEADER_WORDS` removes the sheet's own furniture
   (`COURSE`, `UNITS`, `GRADE`, `TOTAL`, `PREREQ`, `REMARKS`, `ELECTIVE`, …).
   `NSTP`, `CWTS`, `ROTC`, and `PE` stay *out* of the stoplist — they are real
   courses, and the units gate handles them.
3. **Row and column context.** A shape-passing, non-stopword token is this row's
   code only if the row also yields units-or-grade **and** the token sits in the
   elected code column. Shape-matching tokens elsewhere are treated as
   prerequisite references instead.

### 3.2 Column roles are decided once per table, by vote

The most important robustness decision. Per-cell guessing is what fails on real
sheets, because units and grades are both small decimals — a `3.0` could be
either. So each column's role is elected once by voting across all rows, and then
every row is read by index.

Voting order (first applicable wins):

1. **A header row overrides everything.** A row with ≥2 `HEADER_WORDS` and zero
   course codes *names* its columns (`PRE-REQUISITE` → prereq). Repeated headers
   on later pages are deduped by exact cell tuple.
2. **code** — most `is_course_code()` hits.
3. **units** — ≥70% of non-empty cells are integers 1–6; leftmost wins ties
   (units conventionally precede grades).
4. **grade** — ≥60% of **non-empty** cells are in the grade set. The threshold is
   over non-empty cells because a grade column is sparse by nature: only
   completed courses are filled.
5. **prereq** — highest fraction of cells holding a course code that is not the
   row's own, or a none-marker (`NONE`, `-`, `N/A`).
6. **title** — greatest mean length among what remains.
7. **year/term** — ≥50% parse as a term banner.

The tally is kept and printed by `scripts/inspect_checklist.py`, so the user sees
*why* a column was chosen and can override it in the artifact.

### 3.3 "Already taken" is a pass, not a filled cell

A course counts as taken **iff** its grade parses as passing. Pass:
`4.0 3.5 3.0 2.5 2.0 1.5 1.0`, `P`, `S`, `CR`, `✓`. Not taken: `0.0` (failed),
`9.9` (deferred), `INC`, `INP`, `W`, `WP`, `WF`.

This is the feature's sharpest correctness point. A naive "the grade cell is
non-empty, so it's done" would credit failures and withdrawals — and then tell a
student to enrol in a course they are actually blocked from. It has its own tests
(`test_curriculum_model.py::test_taken_codes_excludes_failed_courses`,
and the grade cases in `test_checklist_parser.py`).

Typed input is accepted in parallel and *unioned* with the grade-derived set,
with a second prompt to remove codes the grade column got wrong. Codes not on the
checklist are reported, never silently dropped.

### 3.4 Measured: the BS Computer Engineering (ID 122) checklist

The design above was written before seeing a real checklist. Measuring one
confirmed most of it and corrected two things outright.

**What the file is.** Gokongwei College of Engineering, *Bachelor of Science in
Computer Engineering*, "checklist for freshmen who started AY 2022-2023". Two
landscape pages (1008×612pt), fully text-based — no OCR needed. **103 courses,
209 units, 12 sequential terms** (`FIRST TERM` … `TWELFTH TERM`), which at three
terms per year maps to Y1T1–Y4T3.

**Tier A wins outright.** `extract_tables(lines)` finds 4 ruled tables across the
2 pages and yields all 139 rows. Tiers B and C were never needed.

**The multi-column risk did not materialize.** The pages *are* laid out with two
side-by-side term blocks (page 1: x0≈112 and x0≈509; page 2: x0≈62 and x0≈529),
which is exactly the layout §13 predicted would defeat a global x0 clustering.
But because the blocks are separately ruled, `find_tables` returns them as two
independent tables and the problem dissolves. Tables are read left-to-right so
terms arrive in sequence. The `page.crop()` mitigation was therefore not built —
it remains the fallback if a sheet ever needs tier C on a multi-column page.

**Column order:** `code=0, title=1, units=2, requirement-type=3, requirement-codes=4`.
No grade column exists at all (see below).

#### The correction: a requirement-TYPE column

The design assumed one prerequisite column. This sheet has **two**, and the
first holds a relationship marker that pairs *positionally* with the second:

| Row | type | codes | Meaning |
|---|---|---|---|
| `CALENG1` | `H` | `FNDMATH` | hard prerequisite |
| `LBYCPA2` | `C` | `DATSRAL` | **co-requisite** — the lab travels with its lecture |
| `LBYCPB3` | `H/H` | `FUNDLEC/LOGDSGN` | two hard prerequisites |
| `CPECOG1` | `H/C` | `EMBDSYS/THSCP4A` | one of each |
| `ENGPHYS` | `S / H` | `CALENG1 / BASPHYS` | a "soft" marker and a hard one |

Two consequences, both of which *improve* on the original design:

1. **`/` means AND here, not OR.** §13 of the pre-implementation plan worried
   that a slash might be a disjunction and that ANDing it would over-constrain.
   The parallel type list settles it: `H/H` beside two codes is two hard
   prerequisites. `pair_requirements()` therefore trusts the pairing only when
   the two lists are the same length, and warns when they are not.
2. **Corequisites are stated, not inferred.** 20 of 103 courses carry a `C`
   marker, which is precisely the §10.10.1 lab-with-lecture relationship. The
   `planner.pair_labs` heuristic is redundant for this sheet — the data says it
   outright. The heuristic stays for checklists that do not.

#### `S` is a soft requisite: sat, not necessarily passed

The three markers are not three flavours of the same thing. Confirmed against
the college's own usage:

| Marker | Meaning | Satisfied by |
|---|---|---|
| `H` | hard prerequisite — must have **passed** it | `taken` |
| `S` | soft prerequisite — must have **sat** it, pass or fail | `attempted` |
| `C` | corequisite — same term (§10.10.1) | scheduled together |

`ECNOMIC S CALENG1` means: sit Differential Calculus, fail it, and you may still
take Engineering Economics. This is why the planner tracks **two** sets rather
than one. `taken` is what you passed; `attempted` is what you sat, and the
second is a superset of the first. `_unmet_edges()` checks a hard requirement
against `taken` and a soft one against `attempted`; both still produce a graph
edge when unsatisfied, because either way the course has to come first — they
differ only in what clears them.

Getting this wrong is not cosmetic. Treating `S` as hard delayed `ECNOMIC` by a
full term for a student who had sat and failed `CALENG1`. Seven of the 103
courses carry soft requisites (`ENGPHYS`, `ENGDATA`, `DSIGPRO`, `CPECOG2`,
`CPECOG3`, `ECNOMIC`, `ENGMANA`), so the error compounded across a plan.

Where the two lists cannot be paired positionally, the fallback is **hard**:
over-constraining only delays a course, whereas mistaking a hard requirement for
a soft one would clear a student into something they cannot take.

#### The other correction: no grade column

This checklist has **no place to record grades**, so extraction can mark nothing
as passed and says so in a warning. "Already taken" comes entirely from what the
student types at the `/plan` prompt. The grade-parsing logic still exists and is
still tested, because other programs' checklists do carry the column — but the
first real file made the typed path the primary one, not the fallback.

That also surfaced a genuine bug: the plan header derived "units completed" from
the artifact's `taken:` flags, so a student two years in was told "0 of 209 units
completed". `CoursePlan` now carries the set the planner actually used, and
`test_header_counts_courses_the_planner_treated_as_passed` guards it.

#### The per-term unit limits, and the sheet's own arithmetic

The single most consequential measurement. Each term's `TOTAL` row prescribes a
load, and those loads are **16–19 units**, not 15. Summing the credited units of
the courses placed in each term reproduces the sheet's own totals on **10 of 12
terms exactly**:

| Term | Sheet's TOTAL | Derived | |
|---|---|---|---|
| 1 | `17` | 17 | ✓ |
| 2 | `18 (3)` | 18 (+3 non-credit: NSTPCW1) | ✓ |
| 3 | `19 (4)` | 19 (+4: NSTPCW2, LCLSONE) | ✓ |
| 6 | `17 (1)` | 17 (+1: LCLSTWO) | ✓ |
| 8 | `0` | **19** | sheet's cell is broken |
| 10 | `#REF!` | **18** | sheet's cell is broken |
| 11 | `3` | 3 | ✓ (the practicum term) |

Two conclusions. First, the parenthesised notation is a **non-credit marker**,
and the totals only reconcile if those units are excluded from the load — which
is why `Course.credited` exists. Second, **deriving the cap beats parsing the
TOTAL row**: it agrees wherever the sheet is intact and silently repairs the two
cells where the source spreadsheet is broken. `derive_term_caps()` does that, and
the result is written into the artifact as `term_units` so it stays editable.

#### Extraction quality

Judged on the checks that would expose a mis-read column:

- **Zero dangling references** — every one of the 45 prerequisite and 20
  corequisite codes names a course that exists on the same sheet. A column read
  wrongly produces dangling codes almost immediately, so this is the single most
  informative check, and it is asserted in `test_checklist_parser.py`.
- **Zero cycles** in the resulting graph.
- **All 103 courses placed** in a term; the units-per-term totals reconcile with
  the sheet's own `TOTAL` rows.
- Two warnings, both accurate and both informational (the `S` marker, and the
  missing grade column).

Thresholds needed no adjustment. Two parser bugs surfaced during testing rather
than on this file, and both are guarded now: `split_codes` uppercased tokens
before the shape test, which made every single-word course *title* look like a
course code and let the title column win the prerequisite vote; and `"N/A"` was
being split on its own slash into `N` and `A`.

## 4. The Curriculum Artifact

`data/checklists/<program-id>.curriculum.yaml` — see `src/curriculum/model.py`
for the writer and loader, and `README.md` for the workflow.

**Architectural Decision AD-8: the planner reads this file, never the PDF.** The
escape hatch is the design, not a fallback. Two consequences worth stating:
a mis-parsed prerequisite is a one-line fix rather than a dead end, and a plan is
reproducible from a file the user owns.

**YAML rather than the `.jsonl` used for ingestion intermediates.** Those hold
hundreds of machine-written records and are never hand-edited; this file exists
*to* be hand-edited. YAML carries the comments that tell the user what was
inferred and what to fix — which is the same reasoning that made
`tests/golden_set.yaml` a YAML file. JSONL was rejected: no comments, so
provenance and warnings could not travel with the data, and one-line records with
quoted lists are miserable to correct by hand.

Three guarantees the loader provides:

- **`write_curriculum_yaml` refuses to overwrite without `force=True`.**
  Clobbering hand corrections is the one unforgivable bug for this artifact.
- **Validation is loud and names the offender** — bad `schema_version`, duplicate
  code, missing `code`/`title`/`units`, implausible units — in the same spirit as
  `config._validate`. A malformed curriculum fails before any planning happens.
- **One deliberate exception:** a prerequisite naming a course *not on the
  checklist* is a **warning**, not an error. Checklists legitimately reference
  codes from other curriculum versions or shifted-in courses. Making that fatal
  would render a correct file unloadable; treating it as permanently blocking
  would hide a course the student can actually take. So the edge is dropped and
  recorded in `unresolved_prereqs`.

### 4.1 Degradation across the three cases

Extraction cannot know in advance whether a checklist states prerequisites. All
three outcomes are first-class:

| Case | `prereq_source` | Planner behavior | What the user is told |
|---|---|---|---|
| (a) prerequisite column | `column` | full dependency graph, real levelling | the flowchart, with edges |
| (b) only year/term grouping | `year_term` | **no synthesized edges**; a course's level *is* its checklist term index | *"Ordering follows your checklist's own year/term layout, not stated prerequisites."* |
| (c) neither | `none` | one unordered bucket; nothing is claimed to be blocked | *"No prerequisite or term information could be extracted. Fill in `prereqs:` or `year:`/`term:` and run /plan again."* |

**The important non-decision in case (b): do not synthesize prerequisite edges
from year/term.** "Every Y1T1 course is a prerequisite of every Y1T2 course"
produces a dense, wrong graph and a diagram with hundreds of meaningless arrows.
Using the term index directly as the level yields the same correct ordering with
zero fabricated claims. In case (c) the page still renders — one panel, no
requirement lines — so the extracted course set remains verifiable.

## 5. The Planner

`src/curriculum/planner.py`. All of it is deterministic; none of it calls a model
(AD-7).

### 5.1 Cycles: Tarjan, then break deterministically

Cycle detection uses **iterative Tarjan SCC**, reporting every component of size
> 1 plus every self-loop. Each cycle is canonicalized (rotated so its
lexicographically smallest code is first) and the list sorted, so output is
stable across runs. Cycles are then broken by sorting each cycle's members on
`(checklist_order, code)` and deleting prerequisite edges pointing *backwards* in
that order; `graphlib.TopologicalSorter` runs on the result.

Two rejected alternatives:

- **Refuse to plan when a cycle exists.** With an unknown input layout a
  spurious cycle is *likely* — one mis-assigned prerequisite cell does it — and
  refusing turns a cosmetic extraction bug into total failure. Instead every
  cycle is reported, broken, and flagged, telling the user which `prereqs:` field
  to fix.
- **`TopologicalSorter.prepare()` and catch `CycleError`.** Simpler (~10 lines),
  but `CycleError` reports one cycle at a time as a DFS *path*, not a canonical
  cycle, so the reported members look arbitrary and "delete the backward edge"
  loses its meaning. Listing every cycle at once is precisely the diagnostic
  value this feature needs. `TopologicalSorter` is still used — for the
  levelling, which is what it is good at.

### 5.2 Levelling and corequisite bundling

`level = 0` when a course has no unmet prerequisites, else
`1 + max(level(p) for p in unmet)`, computed over `static_order()`.
`downstream_counts()` (memoized DFS) gives the "Unlocks" number — how many
courses transitively depend on this one.

Courses that must share a term are grouped into **bundles** (connected components
of the corequisite graph), so the packer has exactly one kind of thing to place.
A bundle's level is its members' maximum, its units their sum, and it is eligible
only when *every* member's prerequisites are satisfied. Corequisites come from
the checklist's own column when present; otherwise, when `pair_labs` is on, a
`LBY*`/"… Laboratory" course is paired with the same-`(year, term)` course whose
title is the lab's minus the trailing "Laboratory". A non-unique match yields no
pairing plus a warning, and every inferred pair is labelled as inferred, citing
§10.10.1 as the authority. A bundle whose units alone exceed the cap becomes
`unreachable` with a note — never silently dropped, and never allowed to spin the
packing loop.

### 5.3 Term packing, and the determinism guarantee

Packing is greedy, level-respecting, and first-fit. Eligible bundles are ordered
by this key:

```python
(bundle.level,                                  # never violate a prerequisite
 min(courses[c].checklist_order for c in bundle.codes),   # the program's own sequence
 -max(downstream[c] for c in bundle.codes),     # among equals, unblock the most
 -bundle.units,                                 # then larger first
 bundle.codes[0])                               # total order: a tiebreak that always decides
```

`level` must come first for correctness. `checklist_order` next, so the
university's intended sequence wins whenever it is known. `-downstream_count`
next, because among otherwise-equal choices, taking what unblocks the most is
what makes a plan short. The last two components exist to make the key **total**:
shuffling the input dict cannot change one byte of output, and
`test_tie_break_is_stable_under_input_reordering` asserts exactly that.

#### The unit limit is per term, and the checklist sets it

§10.2 caps a regular term at 15 units **"or the number of units indicated on the
program checklist"**. For an engineering program those are not the same number:
the BS CpE checklist prescribes **16–19 credited units** per term. Treating 15 as
the cap would stretch a 12-term program out by years and contradict the sheet the
student is actually following, so the checklist's own numbers govern.

Precedence, highest first:

1. `program.max_units` in the curriculum artifact — an explicit hand override.
2. **The checklist's own per-term load** (`term_units` in the artifact), derived
   by summing the credited units of the courses the sheet places in each term.
3. `planner.max_units` from config — the general 15, used only when the
   curriculum states no term loads at all.

Which term's limit applies is decided by **where the student is resuming**: the
earliest checklist term still holding unfinished work. Someone who has cleared
six terms is entering the seventh, so the seventh's prescribed load applies to
their next term, the eighth's to the one after, and so on. Each `PlannedTerm`
records the `cap` that governed it and the `checklist_term` it came from, so the
UI can show *"Next term — 16 of 16 units (checklist term 7)"* rather than a
number the student cannot trace.

**Non-credit courses do not consume the limit.** A checklist writing units in
parentheses — `(3)` for NSTP, `(1)` for a Lasallian studies course — means the
course is required but sits outside the load. That is the sheet's own
arithmetic: a term total of `18 (3)` is 18 credited units plus a 3-unit
non-credit course. `Course.credited` carries this, and such courses are still
scheduled, just not counted.

Rules the packer follows:

- A bundle that does not fit is skipped and smaller ones are still tried
  (first-fit, not best-fit — simpler, and deterministic). It lands in `deferred`.
- A bundle heavier than *every* term limit in the program is `unreachable` with
  a note; it is never dropped, and never allowed to spin the loop.
- **The 12-unit floor is a warning, never a packing objective.** If a term ends
  below `min_units` because nothing else was *eligible*, a note cites §10.1. The
  planner never exceeds the cap to reach the floor.
- Packing stops at `max_terms`; anything left goes to `blocked`. This is the
  guard against a malformed graph running forever.

**`build_plan` never raises for a data reason.** An empty curriculum, an
all-cyclic graph, an over-cap bundle, everything-already-taken — each returns a
`StudyPlan` whose `notes`/`blocked`/`unreachable`/`cycles` carry the problem. That
mirrors `Answer(error=...)` in `src/chat/core.py`: the UI must always have
something honest to show.

`available_now` lists every course eligible next term *regardless of whether the
cap fitted it*, because a student with 10 eligible courses can only take 5 —
showing only the 5 would hide the choice they actually have.

## 6. Policy Grounding: config numbers, retrieved citations

`src/curriculum/policy.py` runs four fixed queries against the existing index and
returns a `PolicyRule` per constraint: our plain-English statement, the number
actually applied, the real `RetrievedChunk.citation`, and a ~200-character
excerpt of the handbook's own words. Four local retrievals; **zero LLM calls,
zero API tokens, sub-second.**

The numbers come from `config/settings.yaml`, not from the retrieved text. Four
reasons this beats extracting them at runtime:

1. **They are constants of a fixed handbook edition.** Extracting 15 and 12 every
   session to obtain values we already know is risk for no information gain.
2. **§10.2 is deliberately ambiguous** — "15 units, *or the number of units
   indicated on the program checklist*". An extractor must choose between two
   governing numbers, and the right answer depends on the student's checklist,
   not on the chunk. A config default plus a per-program `program.max_units`
   override models the provision *correctly*, which no extraction can.
3. **The repo already settled its position on small-model structured output.**
   `src/chat/rewriter.py` uses plain lines rather than JSON "because small models
   mangle JSON often enough that the parser would become the unreliable part". A
   number that gates a scheduling decision is strictly worse than a search query,
   where a bad parse degrades gracefully.
4. **The alternative's own fallback proves the point.** "LLM extraction with
   range validation and a config fallback" means the config value is the answer
   whenever the model is wrong — so config is already the source of truth, with
   an extra ~1,200-token call bolted on against a 6,000 TPM ceiling the Q&A path
   already consumes most of.

What that trade would otherwise leave missing is proof that the rule still
exists, so it is closed by a test: an `@pytest.mark.integration` case asserts the
cited provision is retrievable from the real index (asserting on part + section +
keyword, so re-chunking cannot make it brittle). Config holds the number, the
index proves the rule, the test proves the proof still works.

When the best similarity falls below the retriever's floor, `citation` is `None`
and the UI says *"handbook citation unavailable — the index may be stale"*. The
number still applies, because it is config; only the claim of grounding is
withdrawn. Printing the excerpt is a deliberate honesty measure: the user reads
§10.2's checklist caveat themselves, including the part the planner cannot
resolve for them.

## 7. Rendering

**HTML** (`src/curriculum/html_report.py`) — one panel per planned term, each
course a card carrying its code, units, and requirements; four colour states
(taken / ready / later / **unknown**, the last dashed and amber so uncertainty
is *visible* rather than hidden). Written to
`data/plans/<program-id>-plan.html`.

A node-and-arrow diagram was built first, then replaced — and the reason is
worth recording. Measured on the real BS CpE checklist, a third-year student's
plan is **111 nodes and 68 prerequisite edges**. No auto-layout engine turns
that into anything readable; the arrows cross so heavily they obscure the thing
the student actually wants, which is *what do I take, and when*. So the fix was
to stop drawing the graph and lay out the artifact instead: terms as panels,
and each course's requirements written on its own card as text —
`needs CALENG2` for a hard prerequisite, `after CALENG1` for a soft one,
`with LOGDSGN` for a corequisite. Identical information, no crossing lines, and
it prints.

Three properties the format has to keep, each with a test:

- **Self-contained.** No `http://`, `<script>`, `<img>`, or `@import` may appear
  in the output. The page must work offline, when emailed, and on paper — a
  study plan that needs a CDN is a study plan that breaks during a demo.
- **Escaped.** Course titles come from a PDF outside our control, so everything
  interpolated is `html.escape`d.
- **Printable.** Term panels avoid page breaks; the "Already passed" list
  renders as dense chips rather than full cards. That last one is not cosmetic:
  as cards it cost two extra printed pages and pushed the actual plan off page
  one. A full 103-course program prints in about six pages.

No JavaScript, deliberately: a plan is a document, not an application, and a
static file is one less thing that can fail.

**Terminal** (`src/chat/plan_view.py`) — one `rich` table per term with
`Code | Title | Units | Unlocks`; then deferred, blocked (with the blocking
prerequisite named), the constraints applied with citations, and caveats in
yellow, reusing `terminal._render`'s convention that anything less than fully
verified prints yellow. `render_extraction_summary` is the *same function*
`scripts/inspect_checklist.py` calls, so the script and the chatbot can never
disagree about what was extracted.

## 8. Determinism Guarantees

Stated explicitly because a study plan that changes between runs is unusable:

1. Bundles are sorted by `codes[0]`; sets are never iterated directly.
2. The packing tie-break key is **total**, so input order cannot affect output.
3. Cycles are canonicalized and sorted before reporting.
4. `Course` is a frozen dataclass with tuple fields — hashable, and impossible to
   mutate part-way through a plan.
5. No LLM, hence no sampling, hence no temperature.

## 9. What This Cannot Do

See `limitations.md` § *Course planning* for the full account. The short version:
it is a **schedule, not advice** — it knows what the prerequisite graph and a unit
cap permit, not what is offered, what conflicts, or what your department
recommends. Take it to your adviser.
