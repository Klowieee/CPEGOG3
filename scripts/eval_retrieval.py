"""
eval_retrieval.py — Measure retrieval quality without the LLM (Phase 7/11).

Purpose:
    Run the golden question set through the retriever and report hit@k and
    hit@1 for answerable questions, and the best-similarity distribution for
    not-covered questions (used to sanity-check the similarity floor). Because
    this never calls the generation API, it is fast, free, and deterministic —
    and if retrieval is wrong, no amount of prompting will fix the answer, so
    this is the first thing to trust.

Usage:
    uv run python scripts/eval_retrieval.py

Inputs:
    A built index (run scripts/run_ingestion.py first) and
    tests/golden_set.yaml.

Outputs:
    A printed report; the similarity-floor sweep hint for tuning settings.

Why this file exists:
    docs/testing.md §1 separates retrieval evaluation from answer evaluation
    precisely so retrieval can be judged on its own. This is that tool.
"""

import sys
from pathlib import Path

import yaml

# The report prints section signs and en-dashes; without this the Windows
# console encodes them to '?' replacement characters.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.retriever import best_similarity, build_retriever  # noqa: E402
from src.retrieval.vector_store import VectorStore      # noqa: E402
from src.utils.config import load_settings              # noqa: E402
from src.utils.logging_setup import setup_logging       # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden_set.yaml"


def main() -> None:
    setup_logging()
    settings = load_settings()

    store = VectorStore(settings.paths.vector_db_dir, settings.document.id)
    try:
        store.validate(settings.embedding.model)
    except Exception as exc:
        raise SystemExit(f"{exc}\nBuild the index first: "
                         "uv run python scripts/run_ingestion.py")

    retriever = build_retriever(settings, store=store)
    mode = "hybrid (BM25 + semantic)" if retriever.bm25_index else "semantic only"
    print(f"\nRetrieval mode: {mode}\n")

    golden = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))

    # --- Answerable: hit@k / hit@1 ---
    answerable = golden["answerable"]
    hits_k = hits_1 = 0
    print("ANSWERABLE (expected section must appear in top-k):\n")
    for item in answerable:
        results = retriever.retrieve(item["question"])
        sections = [r.section_number for r in results]
        in_k = item["expected_section"] in sections
        in_1 = sections[:1] == [item["expected_section"]]
        hits_k += in_k
        hits_1 += in_1
        mark = "OK " if in_k else "MISS"
        print(f"  [{mark}] §{item['expected_section']:>2} "
              f"(top={sections[0]}, sim={best_similarity(results):.2f})  "
              f"{item['question'][:52]}")
    n = len(answerable)
    print(f"\n  hit@{settings.retrieval.top_k}: {hits_k}/{n} ({hits_k/n:.0%})   "
          f"hit@1: {hits_1}/{n} ({hits_1/n:.0%})")

    # --- Not covered: best similarity should sit below the floor ---
    print(f"\nNOT COVERED (best similarity vs floor "
          f"{settings.retrieval.similarity_floor}):\n")
    floor = settings.retrieval.similarity_floor
    would_answer = 0
    for item in golden["not_covered"]:
        results = retriever.retrieve(item["question"])
        best = best_similarity(results)
        above = best >= floor
        would_answer += above
        flag = "would ANSWER" if above else "would refuse"
        print(f"  {best:.2f}  {flag}  {item['question'][:52]}")
    print(f"\n  {would_answer}/{len(golden['not_covered'])} not-covered "
          "questions clear the floor and reach the model.")
    if would_answer:
        print("  Expected on a single-domain corpus: every question is 'about "
              "university\n  rules' to a degree. Refusing these is Layer 2's "
              "job (the model reading\n  the excerpts) — raising the floor "
              "far enough to catch them refuses real\n  questions first.")

    # --- Vague phrasings: informational, retrieval only ---
    # These are the questions the rewrite rescue exists for, so a miss here
    # is not a failure — it shows how much work the rescue has to do. The
    # rescue itself needs an API key and is covered by test_golden_set.py.
    vague = golden.get("vague_answerable", [])
    if vague:
        margin = settings.rewrite.margin
        print(f"\nVAGUE (retrieval only, before any rewrite rescue; "
              f"rescue triggers below {floor + margin:.2f}):\n")
        vague_hits = 0
        for item in vague:
            results = retriever.retrieve(item["question"])
            sections = [r.section_number for r in results]
            in_k = item["expected_section"] in sections
            vague_hits += in_k
            best = best_similarity(results)
            mark = "OK  " if in_k else "MISS"
            rescue = "rescue" if best < floor + margin else "direct"
            print(f"  [{mark}] §{item['expected_section']:>2} "
                  f"(sim={best:.2f}, {rescue})  {item['question'][:52]}")
        print(f"\n  hit@{settings.retrieval.top_k}: {vague_hits}/{len(vague)} "
              "without rewriting.")

    print("\nKeep retrieval.similarity_floor low enough that every answerable "
          "and vague\nquestion stays above it — its job is to avoid false "
          "refusals, not to detect\noff-topic questions (docs/testing.md §4).")


if __name__ == "__main__":
    main()
