"""
test_rewriter.py — Tests for src/chat/rewriter.py.

The rewriter's contract is narrow but load-bearing: clean queries out of
messy model output, and NEVER raise — it runs on a path that is already
heading for a refusal, so a failure there must degrade to the old behavior
rather than break answering.

Dependencies:
    pytest, src.chat.rewriter, src.llm.backend.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat.rewriter import (                       # noqa: E402
    QueryRewriter,
    build_rewrite_prompt,
    parse_rewrites,
)
from src.llm.backend import LLMBackend, LLMError      # noqa: E402


class FakeBackend(LLMBackend):
    """Returns a canned reply, or raises LLMError."""

    def __init__(self, reply: str = "", fail: bool = False):
        self.reply = reply
        self.fail = fail
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        if self.fail:
            raise LLMError("boom")
        return self.reply


def test_prompt_has_system_and_user_turns():
    messages = build_rewrite_prompt("can i get kicked out")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "can i get kicked out" in messages[1]["content"]


def test_parse_strips_bullets_and_numbering():
    raw = "- academic dishonesty\n2. major offense sanction\n* expulsion grounds"
    assert parse_rewrites(raw, "orig") == [
        "academic dishonesty", "major offense sanction", "expulsion grounds"]


def test_parse_strips_quotes_and_blank_lines():
    raw = '\n"academic dishonesty"\n\n   \n\'major offense\'\n'
    assert parse_rewrites(raw, "orig") == ["academic dishonesty", "major offense"]


def test_parse_drops_echo_of_the_original_question():
    raw = "Can I get kicked out\nacademic dishonesty sanction"
    assert parse_rewrites(raw, "can i get kicked out") == [
        "academic dishonesty sanction"]


def test_parse_deduplicates_case_insensitively():
    raw = "academic dishonesty\nAcademic Dishonesty\nmajor offense"
    assert parse_rewrites(raw, "orig") == ["academic dishonesty", "major offense"]


def test_parse_caps_at_max_queries():
    raw = "one\ntwo\nthree\nfour\nfive"
    assert parse_rewrites(raw, "orig", max_queries=2) == ["one", "two"]


def test_parse_handles_empty_reply():
    assert parse_rewrites("", "orig") == []
    assert parse_rewrites(None, "orig") == []


def test_rewrite_returns_parsed_queries():
    backend = FakeBackend("academic dishonesty\nmajor offense sanction")
    assert QueryRewriter(backend).rewrite("can i cheat") == [
        "academic dishonesty", "major offense sanction"]
    assert backend.calls == 1


def test_rewrite_returns_empty_on_backend_failure():
    """The whole point: a dead rewrite call must not break the answer path."""
    backend = FakeBackend(fail=True)
    assert QueryRewriter(backend).rewrite("can i cheat") == []


def test_rewrite_returns_empty_on_unusable_output():
    assert QueryRewriter(FakeBackend("   \n\n")).rewrite("can i cheat") == []


def test_rewrite_respects_max_queries():
    backend = FakeBackend("one\ntwo\nthree\nfour")
    assert len(QueryRewriter(backend, max_queries=2).rewrite("q")) == 2
