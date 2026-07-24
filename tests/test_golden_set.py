"""
test_golden_set.py — End-to-end evaluation against the golden set (Phase 11).

Two integration tests, both skipped unless their prerequisites exist:
  * Retrieval quality (needs a built index): asserts hit@k on answerable
    questions and that not-covered questions stay below the similarity floor.
    No LLM — fast and free once the index is built.
  * Full pipeline (needs index + API key): runs a couple of questions through
    the real ChatEngine and checks answerable -> answer with citations,
    not-covered -> refusal.

Both are marked 'integration' and skip cleanly in offline CI. The unit tests
in the other test_*.py files cover the same logic with fakes and always run.

Dependencies:
    pytest, pyyaml, the project stack. Real model/API only when present.
"""

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_settings              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = yaml.safe_load((ROOT / "tests" / "golden_set.yaml").read_text("utf-8"))


def _index_ready():
    try:
        from src.retrieval.vector_store import VectorStore
        s = load_settings()
        store = VectorStore(s.paths.vector_db_dir, s.document.id)
        store.validate(s.embedding.model)
        return True
    except Exception:
        return False


def _build_retriever():
    from src.retrieval.retriever import build_retriever
    s = load_settings()
    return build_retriever(s), s


@pytest.mark.integration
@pytest.mark.skipif(not _index_ready(),
                    reason="vector index not built (run scripts/run_ingestion.py)")
def test_retrieval_hit_at_k():
    retriever, _ = _build_retriever()
    answerable = GOLDEN["answerable"]
    hits = 0
    for item in answerable:
        results = retriever.retrieve(item["question"])
        if item["expected_section"] in [r.section_number for r in results]:
            hits += 1
    rate = hits / len(answerable)
    assert rate >= 0.8, f"retrieval hit@k too low: {rate:.0%}"


@pytest.mark.integration
@pytest.mark.skipif(not _index_ready(),
                    reason="vector index not built (run scripts/run_ingestion.py)")
def test_vague_questions_retrieve_their_section():
    """Casual phrasings must at least reach the right section.

    Retrieval alone gets most of the way; whether the excerpts actually state
    the rule is Layer 2's problem, and the rewrite rescue's. Kept at 0.8 to
    match the answerable threshold — a vague question is not allowed to be a
    second-class question at the retrieval stage.
    """
    retriever, _ = _build_retriever()
    vague = GOLDEN["vague_answerable"]
    hits = sum(
        item["expected_section"] in
        [r.section_number for r in retriever.retrieve(item["question"])]
        for item in vague
    )
    rate = hits / len(vague)
    assert rate >= 0.8, f"vague retrieval hit@k too low: {rate:.0%}"


@pytest.mark.integration
@pytest.mark.skipif(not _index_ready(), reason="vector index not built")
def test_similarity_floor_does_not_cause_false_refusals():
    """The floor's real job: never block a question the handbook covers.

    It is deliberately NOT asserted here that not-covered questions fall
    below the floor. On a single-domain corpus they do not — measured, they
    score 0.50-0.64 against a 0.35 floor, because every question is "about
    university rules" to a degree. Refusing them is Layer 2's job (the model
    reading the excerpts), which test_full_pipeline_answer_and_refuse covers.
    Raising the floor far enough to catch them would refuse real questions
    first.
    """
    retriever, s = _build_retriever()
    covered = GOLDEN["answerable"] + GOLDEN["vague_answerable"]
    for item in covered:
        results = retriever.retrieve(item["question"])
        assert retriever.meets_floor(results), (
            f"floor {s.retrieval.similarity_floor} wrongly refuses: "
            f"{item['question']}"
        )


@pytest.mark.integration
@pytest.mark.skipif(
    not (_index_ready() and os.environ.get("GROQ_API_KEY")),
    reason="needs built index AND GROQ_API_KEY for a live end-to-end run",
)
def test_full_pipeline_answer_and_refuse():
    from src.chat.terminal import build_engine
    engine = build_engine(load_settings())

    answered = engine.answer_question("Is plagiarism a major offense?")
    assert not answered.refused and answered.citations

    refused = engine.answer_question("What is the campus wifi password?")
    assert refused.refused


@pytest.mark.integration
@pytest.mark.skipif(
    not (_index_ready() and os.environ.get("GROQ_API_KEY")),
    reason="needs built index AND GROQ_API_KEY for a live end-to-end run",
)
def test_vague_question_is_answered_live():
    """One vague question, end to end, through the rewrite rescue if needed.

    Deliberately ONE question: a refused first pass costs a rewrite call plus
    a second full answer call, and Groq's free tier caps this model at 6000
    tokens per minute (config/settings.yaml). Looping the whole vague set
    here would rate-limit rather than test anything.
    """
    from src.chat.terminal import build_engine
    engine = build_engine(load_settings())

    answer = engine.answer_question("what happens if I copy someone's homework")
    assert not answer.refused, "vague question refused; check the rewrite rescue"
    assert answer.citations
