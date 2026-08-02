# Limitations

An honest account of what version 1 does not do well. Stating these plainly
is part of the project's methodology (and pre-empts the obvious defense
questions).

## Retrieval

- **Dense-only retrieval by default.** We match on meaning. A BM25 keyword
  channel is implemented (`src/retrieval/bm25_index.py`) and can be switched
  on in settings, but it ships **disabled**: measured on this corpus it did
  not improve retrieval and at full weight it made vague questions worse.
  See system_design.md §6 for the numbers.
- **No reranking.** The top-k are used as-is. A cross-encoder reranker would
  improve ordering, at some latency cost. The architecture leaves an
  insertion point for it.
- **The similarity floor cannot detect off-topic questions.** On a
  single-domain corpus every question is "about university rules" to a
  degree: measured, deliberately not-covered questions score 0.50–0.64
  against a 0.35 floor, so none of them are stopped there. The floor's real
  job is avoiding *false* refusals; deciding that the excerpts do not answer
  the question is Layer 2's (the model's). Raising the floor far enough to
  catch off-topic questions would refuse real ones first.
- **Vague questions cost more.** A casually-worded question that the model
  first answers `NOT_COVERED` triggers a rewrite call plus a second answer
  call. On Groq's free tier (6000 tokens/minute for this model) a burst of
  such questions can approach the rate limit — see the note in
  config/settings.yaml.

## Generation

- **Depends on an external API.** Answer generation calls a hosted model
  (Groq/Llama 3.1 8B by default). The retrieval stack is fully local, but
  generation needs a network and a key. A local SLM backend is designed for
  but not implemented (future_work.md).
- **Small model.** An 8B model is used deliberately (cost, speed, the "SLM"
  framing). It is generally faithful when handed the right excerpts, but is
  weaker than frontier models at nuanced multi-part questions.
- **Data leaves the machine at generation time.** The question and the
  retrieved handbook excerpts are sent to the API provider. The handbook is
  public, so exposure is minimal, but it is not zero.

## Document scope

- **One edition, English only.** The AY 2021–2025 Student Handbook, in
  English. Non-English questions are out of scope (the embedding model and
  corpus are English); they may be answered poorly rather than refused.
- **Parsing is tuned to this template.** Heading and provision detection use
  font sizes and column positions measured from this InDesign export. A
  differently-formatted future edition may need the thresholds in
  `src/ingestion/parser.py` re-checked (they are named constants for exactly
  this reason).
- **Tables and directories are approximate.** The appendix office/room
  directories are small-type tabular content; they are kept and chunked, but
  their line-by-line structure is flattened, so answers about them are less
  precise than answers about prose policy.

## Course planning (Phase 15)

**Extraction**

- **The planner is only as good as the checklist parse.** The extracted
  `data/checklists/<program>.curriculum.yaml` is the contract, not the PDF
  (AD-8). If a sheet's layout defeats all three extraction tiers, the honest
  outcome is a YAML with warnings and empty `prereqs:` fields — a form for you
  to fill in, not a plan. `scripts/inspect_checklist.py` exists to tell you
  which of those two you got, in its first three lines of output.
- **Tuned to one export, like the handbook parser.** Column-role voting, the
  x-gap threshold, and the course-code shape are measured from one file. They
  are named constants for the same reason `src/ingestion/parser.py`'s font
  sizes are.
- **Units and grades are hard to tell apart.** Both are small decimals.
  Deciding each column's role once by voting across all rows is far more
  reliable than guessing per cell, but a sheet with *two* unit columns
  (separate LEC and LAB units, common on engineering checklists) or "Credit"
  alongside "Units" can still be voted wrong. The `extraction.columns` block in
  the YAML is editable for exactly this.
- **"Or" prerequisites are not modelled.** `prereqs` is a flat AND-list, so a
  genuine *"A or B"* requirement would be recorded as both required. Note the
  measured CpE checklist does **not** have this problem: its slashes are ANDs,
  confirmed by the parallel `H/H` type column, and the parser only trusts that
  pairing when the two lists are the same length. Where a real disjunction does
  appear, over-constraining is the safe direction — the planner delays a course
  rather than telling you to enrol in one you are blocked from. Proper
  AND-of-ORs is a `schema_version: 2` change, deferred until a sheet needs it.
- **Soft requisites depend on knowing what you *attempted*, not just passed.**
  An `S` requirement is cleared by having sat the course, even if you failed it,
  so the planner tracks attempted separately from passed. It learns what you
  attempted from two places: a failing grade recorded in the curriculum file,
  and the courses you name at the *"anything in there you HAVEN'T passed?"*
  prompt — those were inside a term you marked complete, so you sat them. A
  course you dropped before it started, or one you never enrolled in at all, is
  not attempted, and the planner has no way to know about it unless you say so.
  If a soft requisite is holding something back that it should not, add the
  course to `soft_prereqs:` handling by recording a failing `grade:` on it.
- **Grades are read, not judged.** `0.0`, `9.9` (deferred), `INC`, and
  withdrawals count as *not taken*. That is correct for prerequisite purposes,
  but it says nothing about your standing under §10.17's accumulated-failure
  limits, which this feature does not model.
- **Many checklists have no grade column at all.** The measured CpE checklist
  does not, so nothing can be marked as passed automatically; you type the
  courses you have finished when `/plan` asks, or set `taken: true` in the
  curriculum file. Extraction says so explicitly rather than quietly reporting
  that you have completed nothing.
- **Lab–lecture pairing was not needed here.** The CpE checklist states its
  corequisites outright with a `C` marker (20 of 103 courses), so the
  `planner.pair_labs` heuristic never fires on it. That heuristic exists for
  checklists that leave the relationship implicit.

**Planning**

- **A schedule, not advice.** The plan is what the prerequisite graph and a unit
  cap permit. It does not know whether a course is actually offered next term
  (§10.20/§10.21 exist precisely because courses go unoffered), whether sections
  conflict, whether you need your Associate Dean's approval, or what your
  department recommends. Take it to your adviser.
- **Greedy packing, not optimal.** Term packing is greedy and level-respecting,
  with ties broken by the checklist's own sequence and then by how much a course
  unblocks. It reliably produces a *valid* plan; it does not prove the
  *shortest* one. Optimal packing is bin-packing — NP-hard — and the terms saved
  would be noise against the offering constraints above.
- **Cycles are repaired, not solved.** Two courses listing each other as
  prerequisites is an extraction error, not a curriculum. Every cycle is
  reported, broken using your checklist's term order, and flagged — because
  refusing to plan would turn a one-cell parsing mistake into total failure. If
  a cycle appears in the caveats, fix `prereqs:` in the YAML.
- **Lab–lecture pairing is a guess.** §10.10.1 requires a laboratory and its
  lecture in the same term; when your checklist states no corequisites, they are
  paired by code prefix and title, and the guess is always labelled as inferred.
  Turn it off with `planner.pair_labs: false`.
- **Placeholders stay placeholders.** `GE ELECTIVE 1` is scheduled as a 3-unit
  slot in the term your checklist puts it in. The planner cannot pick the actual
  course for you, and it does not know that elective's own prerequisites.
- **The unit limit comes from your checklist, not from a fixed number.** §10.2
  sets 15 units *"or the number of units indicated on the program checklist"*,
  and for engineering programs those differ — the CpE checklist prescribes 16–19
  credited units per term. The planner uses your checklist's own per-term loads,
  falling back to the configured 15 only when a curriculum states none, and
  `program.max_units` overrides both. Which term's limit applies depends on where
  you are resuming: clear six terms and the seventh's load governs your next one.
  It warns when a term falls below the 12-unit full-time floor from §10.1, and
  prints the graduating-overload provision as a note — it never decides an
  overload for you.
- **Courses with parenthesised units are not counted toward the load.** `(3)` on
  the sheet means required but non-credit (NSTP, the Lasallian series). They are
  still scheduled; they just do not consume the term limit, which is how the
  checklist's own `18 (3)` totals add up. If your program counts them, set
  `credited: true` on those courses.
- **No LLM is in the loop (AD-7).** `/plan` makes zero API calls and spends zero
  tokens. Every constraint it applies is printed with a real handbook citation
  retrieved from the local index, and the numbers come from
  `config/settings.yaml`, not from a model. That is a deliberate trade: the
  citation proves the rule exists, and the config value is what is actually
  applied — so if the index is stale, the plan still holds but the grounding
  claim is visibly withdrawn.

## Behavior

- **No conversation memory.** Follow-ups that rely on a previous question
  ("what about for graduate students?") are not understood in context; ask
  full questions.
- **No guarantee of completeness.** The bot answers from the top-k retrieved
  excerpts; if a rule is spread across many sections, it may cite only the
  most similar ones. It is an assistant for finding and explaining handbook
  content, not an authority — the handbook itself governs.
