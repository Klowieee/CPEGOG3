"""
test_backend.py — Tests for src/llm/backend.py (Phase 9). No network needed.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.backend import APIBackend, LLMError, LLMBackend  # noqa: E402
from src.utils.config import LLMSettings                      # noqa: E402

SETTINGS = LLMSettings(
    base_url="https://example/v1", model="test-model", api_key_env="TEST_KEY",
    temperature=0.1, max_output_tokens=500, request_timeout_seconds=30,
    max_retries=2,
)


class _Msg:
    def __init__(self, content, finish):
        self.message = type("M", (), {"content": content})
        self.finish_reason = finish


class _Resp:
    def __init__(self, content, finish="stop"):
        self.choices = [_Msg(content, finish)]
        self.usage = None


class FakeClient:
    """Mimics the OpenAI client surface: client.chat.completions.create(...)."""

    def __init__(self, reply=None, fail_times=0, finish="stop"):
        self.reply = reply
        self.fail_times = fail_times
        self.finish = finish
        self.calls = 0
        self.last_kwargs = None
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.calls <= self.fail_times:
            raise RuntimeError("transient")
        return _Resp(self.reply, self.finish)


def test_generate_returns_content():
    client = FakeClient(reply="Hello from the model.")
    backend = APIBackend(SETTINGS, client=client)
    assert backend.generate([{"role": "user", "content": "hi"}]) == \
        "Hello from the model."
    assert client.calls == 1


def test_generate_retries_then_succeeds(monkeypatch):
    import src.llm.backend as mod
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)  # no real waiting
    client = FakeClient(reply="ok", fail_times=2)            # fail twice, then ok
    backend = APIBackend(SETTINGS, client=client)
    assert backend.generate([{"role": "user", "content": "hi"}]) == "ok"
    assert client.calls == 3


def test_generate_raises_after_exhausting_retries(monkeypatch):
    import src.llm.backend as mod
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    client = FakeClient(reply="never", fail_times=99)
    backend = APIBackend(SETTINGS, client=client)
    with pytest.raises(LLMError, match="Generation failed after 3 attempts"):
        backend.generate([{"role": "user", "content": "hi"}])


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    backend = APIBackend(SETTINGS)                 # no injected client
    with pytest.raises(LLMError, match="TEST_KEY is not set"):
        backend.generate([{"role": "user", "content": "hi"}])


def test_apibackend_is_an_llmbackend():
    assert isinstance(APIBackend(SETTINGS), LLMBackend)


def test_empty_content_raises_instead_of_returning_blank(monkeypatch):
    # Regression: a thinking model that spends max_tokens on reasoning returns
    # a SUCCESSFUL call with empty content. Returning "" made the chat layer
    # report the question as out-of-scope; it must be a loud error instead.
    import src.llm.backend as mod
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    client = FakeClient(reply="", finish="length")
    backend = APIBackend(SETTINGS, client=client)
    with pytest.raises(LLMError, match="no visible text"):
        backend.generate([{"role": "user", "content": "hi"}])
    assert client.calls == 3                      # retried before giving up


def test_reasoning_effort_is_sent_only_when_configured():
    client = FakeClient(reply="ok")
    APIBackend(SETTINGS, client=client).generate([{"role": "user", "content": "x"}])
    assert "reasoning_effort" not in client.last_kwargs

    thinking = replace(SETTINGS, reasoning_effort="low")
    client = FakeClient(reply="ok")
    APIBackend(thinking, client=client).generate([{"role": "user", "content": "x"}])
    assert client.last_kwargs["reasoning_effort"] == "low"
