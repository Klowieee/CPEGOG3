"""
test_config.py — Unit tests for src/utils/config.py (Phase 1).

Purpose:
    Verify that the real settings.yaml loads and validates, and that the
    loader fails loudly (ConfigError) on missing files, missing keys, and
    inconsistent values — startup-time failure is the whole point of the
    module.

Dependencies:
    pytest, src.utils.config.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import ConfigError, load_settings  # noqa: E402


def test_real_settings_file_loads():
    """The shipped config/settings.yaml must always be valid."""
    s = load_settings()
    assert s.document.id == "student-handbook-2021-2025"
    assert s.chunking.min_tokens < s.chunking.target_tokens <= s.chunking.max_tokens
    assert 1 <= s.retrieval.top_k <= 20
    assert s.llm.api_key_env  # non-empty
    assert "Handbook" in s.chat.refusal_message


def test_missing_file_raises():
    with pytest.raises(ConfigError, match="not found"):
        load_settings("does/not/exist.yaml")


def test_missing_key_raises(tmp_path):
    bad = tmp_path / "settings.yaml"
    bad.write_text("document:\n  id: x\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(bad)


SETTINGS = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def _variant(tmp_path, old: str, new: str) -> Path:
    """Write a copy of the shipped settings with one substitution applied.

    Asserts the substitution actually happened: tests that silently no-op
    when the shipped file changes stop testing anything.
    """
    text = SETTINGS.read_text(encoding="utf-8")
    assert old in text, f"settings.yaml no longer contains {old!r}"
    p = tmp_path / "settings.yaml"
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    return p


def test_reasoning_effort_is_optional():
    """The shipped file omits it (Groq rejects it), and must still load."""
    assert load_settings().llm.reasoning_effort is None


def test_invalid_reasoning_effort_raises(tmp_path):
    p = _variant(tmp_path, "llm:\n", "llm:\n  reasoning_effort: turbo\n")
    with pytest.raises(ConfigError, match="reasoning_effort"):
        load_settings(p)


def test_inconsistent_chunking_raises(tmp_path):
    """min_tokens > target_tokens must be rejected by _validate."""
    p = _variant(tmp_path, "min_tokens: 80", "min_tokens: 400")
    with pytest.raises(ConfigError, match="min_tokens"):
        load_settings(p)


# --- Optional hybrid / rewrite sections ------------------------------------

def test_hybrid_and_rewrite_default_off_when_absent(tmp_path):
    """Settings files predating these features keep their old behavior."""
    minimal = SETTINGS.read_text(encoding="utf-8")
    minimal = minimal[:minimal.index("  # Hybrid retrieval")] + """
llm:
  base_url: http://x
  model: m
  api_key_env: K
  temperature: 0.1
  max_output_tokens: 100
  request_timeout_seconds: 30
  max_retries: 1

chat:
  refusal_message: Not in the Handbook.
"""
    p = tmp_path / "settings.yaml"
    p.write_text(minimal, encoding="utf-8")

    s = load_settings(p)
    assert s.retrieval.hybrid.enabled is False
    assert s.retrieval.hybrid.rrf_k == 60
    assert s.retrieval.hybrid.keyword_weight == 0.5
    assert s.rewrite.enabled is False
    assert s.rewrite.margin == 0.05
    assert s.rewrite.max_queries == 3


def test_invalid_rrf_k_raises(tmp_path):
    p = _variant(tmp_path, "rrf_k: 60", "rrf_k: 0")
    with pytest.raises(ConfigError, match="rrf_k"):
        load_settings(p)


def test_invalid_keyword_weight_raises(tmp_path):
    p = _variant(tmp_path, "keyword_weight: 0.5", "keyword_weight: 2.0")
    with pytest.raises(ConfigError, match="keyword_weight"):
        load_settings(p)


def test_invalid_rewrite_margin_raises(tmp_path):
    p = _variant(tmp_path, "margin: 0.05", "margin: 0.9")
    with pytest.raises(ConfigError, match="margin"):
        load_settings(p)


def test_invalid_rewrite_max_output_tokens_raises(tmp_path):
    p = _variant(tmp_path, "max_output_tokens: 150", "max_output_tokens: 4")
    with pytest.raises(ConfigError, match="max_output_tokens"):
        load_settings(p)


def test_invalid_max_queries_raises(tmp_path):
    p = _variant(tmp_path, "max_queries: 3", "max_queries: 9")
    with pytest.raises(ConfigError, match="max_queries"):
        load_settings(p)


# --- Optional planner section (Phase 15) -----------------------------------

def test_real_settings_file_loads_the_planner_section():
    """The shipped file sets these; the planner must read the handbook's numbers."""
    p = load_settings().planner
    assert p.max_units == 15.0        # Undergraduate §10.2
    assert p.min_units == 12.0        # Undergraduate §10.1
    # Paths are resolved against the project root, like paths.processed_dir.
    assert p.checklist_dir.is_absolute()
    assert p.plan_dir.is_absolute()


def test_planner_defaults_when_section_absent(tmp_path):
    """Settings files predating Phase 15 keep working with the same numbers.

    Reuses the truncation trick above: the minimal file it builds stops before
    the planner block, so this asserts the optional-section default path.
    """
    minimal = SETTINGS.read_text(encoding="utf-8")
    minimal = minimal[:minimal.index("  # Hybrid retrieval")] + """
llm:
  base_url: http://x
  model: m
  api_key_env: K
  temperature: 0.1
  max_output_tokens: 100
  request_timeout_seconds: 30
  max_retries: 1

chat:
  refusal_message: Not in the Handbook.
"""
    p = tmp_path / "settings.yaml"
    p.write_text(minimal, encoding="utf-8")

    s = load_settings(p)
    assert s.planner.max_units == 15.0
    assert s.planner.min_units == 12.0
    assert s.planner.max_terms == 8
    assert s.planner.pair_labs is True
    assert s.planner.include_taken is True


def test_min_units_above_max_units_raises(tmp_path):
    p = _variant(tmp_path, "min_units: 12", "min_units: 20")
    with pytest.raises(ConfigError, match="min_units"):
        load_settings(p)


def test_invalid_max_units_raises(tmp_path):
    p = _variant(tmp_path, "max_units: 15", "max_units: 99")
    with pytest.raises(ConfigError, match="max_units"):
        load_settings(p)


def test_invalid_max_terms_raises(tmp_path):
    p = _variant(tmp_path, "max_terms: 8", "max_terms: 0")
    with pytest.raises(ConfigError, match="max_terms"):
        load_settings(p)


