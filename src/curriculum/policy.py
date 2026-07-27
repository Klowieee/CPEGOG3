"""
policy.py — The handbook provisions behind every planning constraint.

Purpose:
    Retrieve, for each of the four rules the planner enforces (the term unit
    cap, the full-time floor, lab/lecture co-requisite pairing, and NSTP
    completion), the handbook passage that governs it — so a study plan can
    print each constraint with a real citation and the handbook's own words
    instead of an unsourced number.

Inputs:
    Anything exposing .retrieve(query) and .similarity_floor (the configured
    src.retrieval.retriever.Retriever in production, a fake in tests), plus the
    PlannerSettings holding the numbers that are actually applied.

Outputs:
    list[PolicyRule] — one per POLICY_QUERIES entry, in insertion order: the
    rule as plain English, the number applied, and the citation + excerpt
    backing it, or citation=None when retrieval could not back it up.

Dependencies:
    src.utils.config (PlannerSettings); dataclasses/logging/typing (standard
    library). Deliberately NOT src.retrieval.retriever: the retriever is
    duck-typed (see RetrieverLike), so importing this module costs nothing and
    the repo's hand-rolled fakes satisfy it unchanged.

Why this file exists:
    Architectural Decision AD-7 keeps course planning out of the LLM, and this
    module is where that decision stays honest under pressure. Citing the
    handbook is exactly the kind of feature that invites a generation call;
    here it is four LOCAL retrievals, zero API tokens, and no model of any
    kind. Grounding is added to planning without making planning depend on a
    model being reachable, funded, or correct.

    The NUMBERS therefore come from config (config/settings.yaml -> planner),
    never from the retrieved text, for two reasons.

    First, they are constants of this handbook edition: 15 and 12 do not vary
    per question, so re-deriving them per run buys nothing and risks a
    different answer each time.

    Second, Undergraduate §10.2 is deliberately ambiguous — the maximum load is
    "15 units, or the number of units indicated on the program checklist" — so
    there is no single number in the text TO extract. A configured default
    (planner.max_units) plus a per-program override read from the checklist
    (Curriculum.max_units_override) models that provision exactly as written,
    which is also why every rule carries the handbook's verbatim excerpt: the
    caveat the planner cannot resolve for the student is the student's to read.

    Asking an 8B model to pull that number out of the text instead is the job
    src/chat/rewriter.py already refuses to give it. There, rewrites are parsed
    as plain lines rather than JSON because small models mangle structured
    output often enough that the parser becomes the unreliable part — and a
    rewrite failing merely costs a retrieval. A misread unit cap does not
    surface as a parse error at all; it silently produces a wrong schedule that
    looks exactly as confident as a right one.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Protocol

from src.utils.config import PlannerSettings

log = logging.getLogger(__name__)

# The four queries, fixed rather than generated: each was checked against the
# real index (see tests/test_policy.py's integration test) and a fixed query
# means the citation a student sees for a rule is the same one every run.
POLICY_QUERIES: dict[str, str] = {
    "max_units": "maximum academic load for undergraduate students per regular "
                 "term units",
    "min_units": "full-time undergraduate student minimum academic units per term",
    "lab_coreq": "laboratory course is a co-requisite of the corresponding "
                 "lecture course same term",
    "nstp": "students are required to complete the two NSTP courses flowchart",
}

# Long enough to carry a whole provision sentence (the handbook's median
# provision is ~32 words), short enough to sit under a plan's constraint line.
EXCERPT_CHARS = 200
ELLIPSIS = "…"

# What a caller prints in place of a citation when retrieval came up short.
# It lives here so the wording cannot drift from the condition that causes it.
MISSING_CITATION_NOTE = (
    "(handbook citation unavailable — the index may be stale; "
    "run scripts/run_ingestion.py)"
)


# --- The retriever contract ----------------------------------------------------

class RetrievedChunkLike(Protocol):
    """The three fields of a retrieved chunk this module reads."""

    text: str
    citation: str
    similarity: float


class RetrieverLike(Protocol):
    """The retriever surface this module uses — deliberately tiny.

    Typing the parameter structurally instead of as the concrete
    src.retrieval.retriever.Retriever keeps two things true: importing this
    module does not drag in chromadb and the embedding model, and every
    hand-rolled fake in tests/ already satisfies the contract. Note that
    meets_floor() is NOT part of it — this module compares against
    similarity_floor itself, because it decides per RULE (withdraw one
    citation) rather than per question (refuse to answer at all).
    """

    similarity_floor: float

    def retrieve(self, question: str,
                 k: int | None = None) -> list[RetrievedChunkLike]:
        ...


# --- Data model ----------------------------------------------------------------

@dataclass(frozen=True)
class PolicyRule:
    """One planning constraint, the number applied, and its handbook backing.

    Frozen because a rule is evidence: once loaded it is quoted in a plan and
    in the plan's Markdown export, and a mutable copy could disagree with the
    number the packer actually used.
    """

    key: str                  # "max_units" | "min_units" | "lab_coreq" | "nstp"
    statement: str            # our plain-English rendering of the rule
    value: float | None       # the number actually applied; None if non-numeric
    citation: str | None      # None when nothing cleared the similarity floor
    similarity: float         # best similarity seen, reported even when weak
    excerpt: str              # ~200 chars of the handbook's own words


# --- Rendering the rules -------------------------------------------------------

def _applied(planner: PlannerSettings) -> dict[str, tuple[float | None, str]]:
    """(value, statement) per rule key — the single place both are decided.

    Kept as one mapping so a statement can never quote a number the planner
    does not apply: both come from the same PlannerSettings field in the same
    expression.
    """
    return {
        "max_units": (
            planner.max_units,
            f"Terms are capped at {planner.max_units:g} units "
            "(maximum regular-term academic load).",
        ),
        "min_units": (
            planner.min_units,
            f"A full-time term carries at least {planner.min_units:g} units; "
            "a lighter term is flagged as a warning, never blocked.",
        ),
        "lab_coreq": (
            None,
            "A laboratory course is scheduled in the same term as the lecture "
            "course it belongs to.",
        ),
        "nstp": (
            None,
            "Both NSTP courses must be completed, and the second follows the "
            "first.",
        ),
    }


def _excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """Collapse whitespace and keep the first ~limit chars, marking a trim.

    Whitespace is collapsed because a chunk's text arrives with its breadcrumb
    newline and the PDF's line breaks intact; the trim stops at a word boundary
    so the quote does not end mid-word. The ellipsis is load-bearing: it tells
    the reader the provision continues, which for §10.2 is the difference
    between a quote and a misquote.
    """
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed

    head = collapsed[:limit]
    cut = head.rfind(" ")
    if cut > limit // 2:                 # a word boundary near enough the end
        head = head[:cut]
    return head.rstrip(" ,;:") + ELLIPSIS


# --- Public API ----------------------------------------------------------------

def load_policy(retriever: RetrieverLike, planner: PlannerSettings,
                k: int = 3, term_caps: dict[int, float] | None = None
                ) -> list[PolicyRule]:
    """Retrieve the handbook provision behind each planning constraint.

    Costs exactly len(POLICY_QUERIES) local retrievals and zero API tokens
    (AD-7). Never raises for retrieval reasons.

    Args:
        retriever: Duck-typed; only .retrieve() and .similarity_floor are used.
        planner: Supplies the numbers actually applied. They are NOT read out
            of the retrieved text — see this module's docstring.
        k: How many chunks to consider per query; the best by similarity wins.
        term_caps: The checklist's own per-term unit loads, when it states them.
            They replace the general maximum in the reported rule, because
            §10.2 defers to "the number of units indicated on the program
            checklist" and reporting 15 while packing to 19 would be a lie.

    Returns:
        One PolicyRule per POLICY_QUERIES entry, in that dict's insertion
        order, so a plan's constraint list is identical between runs.
    """
    applied = _applied(planner)
    if term_caps:
        low, high = min(term_caps.values()), max(term_caps.values())
        span = f"{low:g}" if low == high else f"{low:g}-{high:g}"
        applied["max_units"] = (
            high,
            f"Terms are capped at the load your checklist prescribes "
            f"({span} units), which §10.2 allows in place of the general "
            f"{planner.max_units:g}-unit maximum.",
        )
    floor = retriever.similarity_floor
    rules: list[PolicyRule] = []

    for key, query in POLICY_QUERIES.items():
        value, statement = applied[key]
        try:
            results = retriever.retrieve(query, k=k) or []
        except Exception as exc:
            # Broad on purpose: a missing collection, a model mismatch, and a
            # half-written index all surface differently here, and none of them
            # is a reason to refuse to plan. The rule still applies (it is
            # config); only its grounding is lost, and the log says so.
            log.warning("Policy retrieval for %r failed (%s); '%s' will be "
                        "shown without a citation.", query, exc, key)
            rules.append(PolicyRule(key, statement, value, None, 0.0, ""))
            continue

        # max() rather than results[0]: with hybrid retrieval the list is in
        # fused-rank order, so the first result is not necessarily the closest
        # one (the same reason retriever.py has best_similarity()).
        best = max(results, key=lambda chunk: chunk.similarity, default=None)
        similarity = best.similarity if best is not None else 0.0

        if best is None or similarity < floor:
            # Withdraw the CLAIM OF GROUNDING, not the constraint: the number
            # is configuration and still governs the plan. Callers print
            # MISSING_CITATION_NOTE beside it.
            log.info("No chunk cleared the %.2f floor for '%s' (best %.3f); "
                     "the rule applies but is uncited.", floor, key, similarity)
            rules.append(PolicyRule(key, statement, value, None, similarity, ""))
            continue

        rules.append(PolicyRule(key, statement, value, best.citation,
                                similarity, _excerpt(best.text)))

    return rules
