"""
backend.py — Pluggable LLM generation backend (interface + API impl).

Purpose:
    Define the one seam through which answers are generated, so the provider
    is swappable and a local SLM backend can be added later (AC-1) without
    touching retrieval, prompting, or the chat loop. Ships one implementation:
    APIBackend, an OpenAI-compatible client pointed at the configured endpoint
    (Gemini's OpenAI-compatibility endpoint by default — see
    config/settings.yaml and docs/rag_pipeline.md §7).

Inputs:
    Chat messages (from src.prompts.builder.build_prompt) and LLMSettings.

Outputs:
    The model's raw text reply.

Dependencies:
    openai (lazy import), os, src.utils.config.

Why this file exists:
    Isolating the network/provider details behind LLMBackend keeps the rest
    of the system provider-agnostic and testable: the chat core is tested
    with a fake backend, and swapping Gemini for Groq/OpenAI is a config edit.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod

from src.utils.config import LLMSettings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised for unrecoverable generation failures (surfaced to the user)."""


class _EmptyReply(Exception):
    """Internal: the API returned a choice with no visible text.

    Thinking models (e.g. Gemini 3.x) bill reasoning tokens against
    max_tokens; when the budget runs out the call SUCCEEDS but content is
    empty and finish_reason is "length". Returning "" for that made the chat
    layer see an uncited answer and print the out-of-scope refusal, hiding a
    configuration problem behind a content message. Treated as retryable and
    converted into an explicit LLMError below.
    """


class LLMBackend(ABC):
    """A text-generation backend. Implementations must be interchangeable."""

    @abstractmethod
    def generate(self, messages: list[dict]) -> str:
        """Return the model's reply text for a list of chat messages."""


class APIBackend(LLMBackend):
    """OpenAI-compatible chat-completions backend (default: Groq)."""

    def __init__(self, settings: LLMSettings, client=None):
        """
        Args:
            settings: Provider base_url, model, temperature, limits, and the
                NAME of the env var holding the API key (never the key itself).
            client: Optional pre-built OpenAI-compatible client (injected in
                tests). If None, one is created lazily from settings + env key.
        """
        self.settings = settings
        self._client = client

    def _get_client(self):
        """Create the API client on first use; fail clearly if the key is unset."""
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self.settings.api_key_env)
        if not api_key:
            raise LLMError(
                f"Environment variable {self.settings.api_key_env} is not set. "
                "Export your API key before starting the chatbot, e.g.\n"
                f"  export {self.settings.api_key_env}=your-key-here"
            )
        try:
            from openai import OpenAI  # lazy import
        except ImportError as exc:
            raise LLMError("The 'openai' package is not installed. "
                           "Run: uv sync") from exc

        self._client = OpenAI(base_url=self.settings.base_url, api_key=api_key)
        return self._client

    def generate(self, messages: list[dict]) -> str:
        """Call the chat-completions endpoint with retries and clear errors.

        Retries transient failures (network/rate-limit) up to max_retries with
        linear backoff, then raises LLMError so the chat loop can show a
        graceful message instead of a traceback.
        """
        client = self._get_client()
        kwargs = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_completion_tokens":self.settings.max_output_tokens,
            "timeout": self.settings.request_timeout_seconds,
        }
        # Only sent when configured, so providers that reject the parameter
        # (and non-reasoning models) are unaffected.
        if self.settings.reasoning_effort:
            kwargs["reasoning_effort"] = self.settings.reasoning_effort

        last_error: Exception | None = None

        for attempt in range(self.settings.max_retries + 1):
            try:
                response = client.chat.completions.create(**kwargs)
                return self._extract_text(response)
            except Exception as exc:  # provider/network errors, empty replies
                last_error = exc
                if attempt < self.settings.max_retries:
                    wait = 1.5 * (attempt + 1)
                    log.warning("Generation attempt %d failed (%s); retrying in "
                                "%.1fs", attempt + 1, type(exc).__name__, wait)
                    time.sleep(wait)

        if isinstance(last_error, _EmptyReply):
            raise LLMError(
                f"The model returned no visible text ({last_error}). This "
                "usually means reasoning tokens consumed the whole output "
                "budget: raise llm.max_output_tokens or lower "
                "llm.reasoning_effort in config/settings.yaml."
            )
        raise LLMError(
            f"Generation failed after {self.settings.max_retries + 1} attempts: "
            f"{last_error}"
        )

    @staticmethod
    def _extract_text(response) -> str:
        """Return the reply text, or raise _EmptyReply with the diagnosis."""
        choice = response.choices[0]
        content = (getattr(choice.message, "content", None) or "").strip()
        finish = getattr(choice, "finish_reason", None)
        if not content:
            usage = getattr(response, "usage", None)
            log.warning("Empty completion (finish_reason=%s, usage=%s)",
                        finish, usage)
            raise _EmptyReply(f"finish_reason={finish!r}")
        log.debug("Completion received (%d chars, finish_reason=%s)",
                  len(content), finish)
        return content
