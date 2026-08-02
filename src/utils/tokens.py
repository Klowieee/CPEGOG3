"""
tokens.py — Token counting for chunk budgeting.

Purpose:
    Provide a single count_tokens() used by the chunker to enforce chunk
    size limits, measured in the embedding model's own tokens (what actually
    matters, since the 512-token input limit of bge-small is the hard
    constraint the limits protect).

Inputs:
    Text strings; optionally the embedding model name from settings.

Outputs:
    Integer token counts.

Dependencies:
    transformers (optional). If the tokenizer cannot be loaded — library not
    installed or no network to fetch it — a calibrated word-based heuristic
    is used instead (~1.35 wordpiece tokens per English word), which is
    accurate enough for budgeting because the chunking limits carry margin
    (max 500 vs the model's 512 hard limit).

Why this file exists:
    Chunk limits defined in tokens (docs/chunking_strategy.md §3) need one
    consistent, dependency-tolerant measuring stick shared by the chunker,
    its tests, and any future inspection tooling.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Wordpiece tokenizers emit roughly 1.3-1.4 tokens per English word; 1.35 is
# a safe middle estimate for budgeting purposes.
TOKENS_PER_WORD_ESTIMATE = 1.35


class TokenCounter:
    """Counts tokens with the real embedding tokenizer when available.

    Falls back transparently to a word-count heuristic so that chunking
    logic (and its tests) never depend on heavyweight ML libraries being
    installed. The mode in use is exposed via .using_real_tokenizer and
    logged once at load time.
    """

    def __init__(self, model_name: str | None = None):
        self._tokenizer = None
        self.using_real_tokenizer = False
        if model_name:
            try:
                from transformers import AutoTokenizer  # heavy import, optional

                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.using_real_tokenizer = True
                log.info("Token counting: using real tokenizer for %s", model_name)
            except Exception as exc:  # ImportError, network failure, etc.
                log.info(
                    "Token counting: real tokenizer unavailable (%s); "
                    "using word-based estimate.", type(exc).__name__,
                )

    def count(self, text: str) -> int:
        """Return the (possibly estimated) token count of `text`."""
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        words = len(text.split())
        return max(1, round(words * TOKENS_PER_WORD_ESTIMATE)) if words else 0
