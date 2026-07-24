"""
test_prompts.py — Tests for src/prompts/builder.py (Phase 8). No LLM needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prompts.builder import (  # noqa: E402
    REFUSAL_SENTINEL,
    build_prompt,
    parse_response,
)
from src.retrieval.vector_store import RetrievedChunk  # noqa: E402


def chunk(cid, citation, text="body text", sim=0.8):
    return RetrievedChunk(cid, text, "Undergraduate", "10", "GRADING",
                          ["10.1"], [101], citation, sim)


CHUNKS = [
    chunk("d_1", "General Provisions, Section 5, p. 60", "Plagiarism is an offense."),
    chunk("d_2", "Undergraduate, Section 10, p. 101", "Grades are computed thus."),
]


def test_build_prompt_structure():
    msgs = build_prompt("Is plagiarism an offense?", CHUNKS)
    assert msgs[0]["role"] == "system" and "ONLY" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "[1] General Provisions, Section 5" in user
    assert "[2] Undergraduate, Section 10" in user
    assert user.strip().endswith("QUESTION: Is plagiarism an offense?")


def test_parse_refusal_sentinel():
    ans = parse_response(REFUSAL_SENTINEL, CHUNKS)
    assert ans.refused and ans.text == "" and ans.citations == []


def test_parse_maps_markers_to_citations():
    reply = "Yes, plagiarism is a major offense [1]. Grades follow a scale [2]."
    ans = parse_response(reply, CHUNKS)
    assert not ans.refused
    assert [c.marker for c in ans.citations] == [1, 2]
    assert ans.citations[0].citation == "General Provisions, Section 5, p. 60"
    assert ans.citations[0].chunk_id == "d_1"
    assert "[1]" in ans.text  # marker preserved in visible text


def test_parse_dedupes_repeated_markers():
    ans = parse_response("A [1]. B [1]. C [1].", CHUNKS)
    assert len(ans.citations) == 1 and ans.citations[0].marker == 1


def test_parse_strips_out_of_range_markers():
    ans = parse_response("Answer with a bad ref [9] and a good one [1].", CHUNKS)
    assert "[9]" not in ans.text
    assert [c.marker for c in ans.citations] == [1]


def test_parse_uncited_answer_is_unverified_not_refused():
    # A fluent answer with no resolvable citation is a FORMATTING failure, not
    # an out-of-scope question: flag it so the caller can retry (regression —
    # refusing here made covered questions report "not covered").
    ans = parse_response("Plagiarism is definitely not allowed at all.", CHUNKS)
    assert not ans.refused
    assert ans.unverified
    assert ans.text.startswith("Plagiarism")
    assert ans.citations == []


def test_parse_accepts_grouped_markers():
    # Models routinely write "[1, 2]" despite the prompt asking for "[1][2]".
    ans = parse_response("Both rules apply here [1, 2].", CHUNKS)
    assert not ans.refused and not ans.unverified
    assert [c.marker for c in ans.citations] == [1, 2]
    assert "[1][2]" in ans.text            # normalized in the visible text


def test_parse_accepts_semicolons_and_padding():
    ans = parse_response("A claim [ 2 ]. Another [1; 2].", CHUNKS)
    assert [c.marker for c in ans.citations] == [2, 1]


def test_parse_drops_out_of_range_inside_a_group():
    ans = parse_response("Mixed group [1, 9].", CHUNKS)
    assert [c.marker for c in ans.citations] == [1]
    assert "[9]" not in ans.text and "[1]" in ans.text


def test_parse_empty_reply_refuses():
    assert parse_response("", CHUNKS).refused


def test_parse_sentinel_mention_with_citation_is_not_a_refusal():
    # "NOT_COVERED" appearing at the start of a real, cited answer must not be
    # mistaken for the bare sentinel.
    reply = f"{REFUSAL_SENTINEL} does not apply: the handbook covers this [1]."
    ans = parse_response(reply, CHUNKS)
    assert not ans.refused
    assert [c.marker for c in ans.citations] == [1]


def test_parse_decorated_sentinel_still_refuses():
    assert parse_response(f"**{REFUSAL_SENTINEL}**", CHUNKS).refused
