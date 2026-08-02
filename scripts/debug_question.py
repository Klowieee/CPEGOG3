"""debug_question.py — Trace one question through both refusal layers.

Prints the retrieval scores, the raw model reply, the citation markers the
parser resolved, and which layer decided the outcome. Use this first whenever
the bot refuses a question you believe the handbook covers: it distinguishes
"nothing retrieved" from "model refused" from "model answered but cited
nothing" — three very different faults that share one user-facing message.

Usage:
    uv run python scripts/debug_question.py "Is plagiarism a major offense?"
"""
import sys
from pathlib import Path

# Citations carry section signs and en-dashes; without this the Windows
# console encodes them to '?' replacement characters.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat.terminal import build_engine                      # noqa: E402
from src.llm.backend import LLMError                            # noqa: E402
from src.prompts.builder import build_prompt, parse_response    # noqa: E402
from src.retrieval.retriever import best_similarity             # noqa: E402
from src.utils.config import load_settings                      # noqa: E402

question = " ".join(sys.argv[1:]) or "Is plagiarism a major offense?"
settings = load_settings()
engine = build_engine(settings)

results = engine.retriever.retrieve(question)
mode = "hybrid (BM25 + semantic)" if engine.retriever.bm25_index else "semantic only"
print(f"\nQuestion: {question}")
print(f"Model: {settings.llm.model}  "
      f"(max_output_tokens={settings.llm.max_output_tokens}, "
      f"reasoning_effort={settings.llm.reasoning_effort})")
print(f"Retrieval: {mode}")
print(f"Similarity floor: {engine.retriever.similarity_floor} "
      f"(rescue below {engine.retriever.similarity_floor + engine.rescue_margin:.2f})\n")
for r in results:
    print(f"  {r.similarity:.3f}  {r.citation}")
print(f"\nBest similarity: {best_similarity(results):.3f}")

# Rewrite rescue: the same call the chat engine would make, traced.
if engine.rewriter is not None and engine.needs_rescue(results):
    print("\n--- REWRITE RESCUE (retrieval was weak) ---")
    queries = engine.rewriter.rewrite(question)
    if not queries:
        print("  Rewrite produced nothing usable; keeping original results.")
    else:
        for q in queries:
            print(f"  rewritten: {q}")
        results = engine.retrieve_merged(queries, results)
        print("\n  After rescue:")
        for r in results:
            print(f"    {r.similarity:.3f}  {r.citation}")
        print(f"\n  Best similarity: {best_similarity(results):.3f}")

passes = engine.retriever.meets_floor(results)
print(f"\nLayer 1 (floor) passes: {passes}")
if not passes:
    print("-> Refused by retrieval; no API call made.")
    raise SystemExit(0)

try:
    raw = engine.backend.generate(build_prompt(question, results))
except LLMError as exc:
    raise SystemExit(f"\nCould not reach the model:\n  {exc}")
print(f"\n--- RAW MODEL REPLY ({len(raw)} chars) ---\n{raw!r}")

parsed = parse_response(raw, results)
print("\n--- PARSED ---")
print(f"  refused:    {parsed.refused}   (model emitted the NOT_COVERED sentinel)")
print(f"  unverified: {parsed.unverified} (answered, but no [n] marker resolved)")
print(f"  citations:  {[(c.marker, c.citation) for c in parsed.citations]}")

if parsed.refused:
    print("\n-> Layer 2: the MODEL refused. Check whether the excerpts above "
          "really answer the question.")
    if engine.rewriter is not None:
        print("   In the chat app this refusal triggers the rewrite rescue: "
              "the question\n   is restated in handbook wording, retrieved "
              "again, and asked once more.")
elif parsed.unverified:
    print("\n-> Layer 2: answered without citations. The chat app retries "
          "once, then shows the answer with the retrieved sources flagged.")
else:
    print("\n-> Answered and grounded.")
