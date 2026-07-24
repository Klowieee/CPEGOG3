"""
config.py — Load and validate config/settings.yaml into typed objects.

Purpose:
    Provide every other module with a single, validated `Settings` object
    instead of raw dictionary access. Catching a bad or missing setting
    HERE, at startup, is far easier to debug than a KeyError deep inside
    the pipeline.

Inputs:
    Path to a YAML settings file (defaults to config/settings.yaml,
    resolved relative to the project root).

Outputs:
    A `Settings` dataclass instance with nested sections mirroring the
    YAML structure.

Dependencies:
    pyyaml (external), dataclasses/pathlib (standard library).

Why this file exists:
    Architectural Decision AD-6 (docs/architecture.md) centralizes all
    tunable parameters in settings.yaml; this module is the sole reader
    of that file, so validation logic lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# The project root is two levels above this file: src/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# Values the OpenAI-compatible `reasoning_effort` parameter accepts.
REASONING_EFFORTS = {"minimal", "low", "medium", "high", "none"}


@dataclass(frozen=True)
class DocumentSettings:
    id: str
    title: str
    edition: str
    source_pdf: Path


@dataclass(frozen=True)
class PathSettings:
    processed_dir: Path
    vector_db_dir: Path


@dataclass(frozen=True)
class EmbeddingSettings:
    model: str
    query_prefix: str


@dataclass(frozen=True)
class ChunkingSettings:
    target_tokens: int
    max_tokens: int
    min_tokens: int
    overlap_tokens: int


@dataclass(frozen=True)
class HybridSettings:
    """Keyword+semantic retrieval. Off by default: older settings files that
    predate this section keep their original semantic-only behavior."""

    enabled: bool = False
    rrf_k: int = 60               # reciprocal rank fusion constant
    keyword_weight: float = 0.5   # keyword rank weight relative to semantic


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int
    similarity_floor: float
    hybrid: HybridSettings = HybridSettings()


@dataclass(frozen=True)
class RewriteSettings:
    """Fallback LLM query rewriting, used only when retrieval looks weak."""

    enabled: bool = False
    # Rescue when the best similarity is below similarity_floor + margin, so
    # questions that barely scrape past the floor get a second chance too.
    margin: float = 0.05
    # Small on purpose: providers bill the reserved budget against rate
    # limits, and a rewrite is three short lines.
    max_output_tokens: int = 150
    max_queries: int = 3


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    model: str
    api_key_env: str
    temperature: float
    max_output_tokens: int
    request_timeout_seconds: int
    max_retries: int
    # Optional; forwarded only when set. Reasoning models bill thinking tokens
    # against max_output_tokens, so capping the effort protects the answer.
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class ChatSettings:
    refusal_message: str


@dataclass(frozen=True)
class Settings:
    document: DocumentSettings
    paths: PathSettings
    embedding: EmbeddingSettings
    chunking: ChunkingSettings
    retrieval: RetrievalSettings
    llm: LLMSettings
    chat: ChatSettings
    rewrite: RewriteSettings = RewriteSettings()


class ConfigError(Exception):
    """Raised when settings.yaml is missing, malformed, or invalid."""


def _require(section: dict, key: str, section_name: str):
    """Return section[key] or raise a ConfigError naming the missing key."""
    if key not in section:
        raise ConfigError(
            f"Missing required setting '{section_name}.{key}' in settings.yaml"
        )
    return section[key]


def load_settings(path: Path | str = DEFAULT_SETTINGS_PATH) -> Settings:
    """Load, validate, and return the project settings.

    Args:
        path: Location of the YAML settings file.

    Returns:
        A fully populated, immutable Settings object. Relative paths in
        the file are resolved against the project root.

    Raises:
        ConfigError: If the file is missing, unparseable, or fails
            validation (missing keys or nonsensical values).
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Settings file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # malformed YAML
        raise ConfigError(f"Could not parse {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} did not contain a mapping at the top level")

    def section(name: str) -> dict:
        value = raw.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"Missing or invalid section '{name}' in {path}")
        return value

    doc = section("document")
    paths = section("paths")
    emb = section("embedding")
    chk = section("chunking")
    ret = section("retrieval")
    llm = section("llm")
    chat = section("chat")
    # Optional sections: absent means "feature off", so settings files written
    # before these features stay valid.
    hyb = ret.get("hybrid") or {}
    rw = raw.get("rewrite") or {}

    settings = Settings(
        document=DocumentSettings(
            id=_require(doc, "id", "document"),
            title=_require(doc, "title", "document"),
            edition=_require(doc, "edition", "document"),
            source_pdf=PROJECT_ROOT / _require(doc, "source_pdf", "document"),
        ),
        paths=PathSettings(
            processed_dir=PROJECT_ROOT / _require(paths, "processed_dir", "paths"),
            vector_db_dir=PROJECT_ROOT / _require(paths, "vector_db_dir", "paths"),
        ),
        embedding=EmbeddingSettings(
            model=_require(emb, "model", "embedding"),
            query_prefix=_require(emb, "query_prefix", "embedding"),
        ),
        chunking=ChunkingSettings(
            target_tokens=int(_require(chk, "target_tokens", "chunking")),
            max_tokens=int(_require(chk, "max_tokens", "chunking")),
            min_tokens=int(_require(chk, "min_tokens", "chunking")),
            overlap_tokens=int(_require(chk, "overlap_tokens", "chunking")),
        ),
        retrieval=RetrievalSettings(
            top_k=int(_require(ret, "top_k", "retrieval")),
            similarity_floor=float(_require(ret, "similarity_floor", "retrieval")),
            hybrid=HybridSettings(
                enabled=bool(hyb.get("enabled", False)),
                rrf_k=int(hyb.get("rrf_k", 60)),
                keyword_weight=float(hyb.get("keyword_weight", 0.5)),
            ),
        ),
        llm=LLMSettings(
            base_url=_require(llm, "base_url", "llm"),
            model=_require(llm, "model", "llm"),
            api_key_env=_require(llm, "api_key_env", "llm"),
            temperature=float(_require(llm, "temperature", "llm")),
            max_output_tokens=int(_require(llm, "max_output_tokens", "llm")),
            request_timeout_seconds=int(
                _require(llm, "request_timeout_seconds", "llm")
            ),
            max_retries=int(_require(llm, "max_retries", "llm")),
            # Optional: absent in older settings files, which stay valid.
            reasoning_effort=(str(llm["reasoning_effort"]).strip().lower()
                              if llm.get("reasoning_effort") else None),
        ),
        chat=ChatSettings(
            refusal_message=str(_require(chat, "refusal_message", "chat")).strip(),
        ),
        rewrite=RewriteSettings(
            enabled=bool(rw.get("enabled", False)),
            margin=float(rw.get("margin", 0.05)),
            max_output_tokens=int(rw.get("max_output_tokens", 150)),
            max_queries=int(rw.get("max_queries", 3)),
        ),
    )

    _validate(settings)
    return settings


def _validate(s: Settings) -> None:
    """Sanity-check relationships between settings values."""
    c = s.chunking
    if not (0 < c.min_tokens < c.target_tokens <= c.max_tokens):
        raise ConfigError(
            "chunking values must satisfy 0 < min_tokens < target_tokens "
            f"<= max_tokens (got min={c.min_tokens}, target={c.target_tokens}, "
            f"max={c.max_tokens})"
        )
    if c.overlap_tokens >= c.target_tokens:
        raise ConfigError("chunking.overlap_tokens must be smaller than target_tokens")
    if s.retrieval.top_k < 1:
        raise ConfigError("retrieval.top_k must be at least 1")
    if not (0.0 <= s.retrieval.similarity_floor <= 1.0):
        raise ConfigError("retrieval.similarity_floor must be between 0 and 1")
    if s.retrieval.hybrid.rrf_k < 1:
        raise ConfigError("retrieval.hybrid.rrf_k must be at least 1")
    if not (0.0 <= s.retrieval.hybrid.keyword_weight <= 1.0):
        raise ConfigError("retrieval.hybrid.keyword_weight must be between 0 and 1")
    if not (0.0 <= s.rewrite.margin <= 0.5):
        raise ConfigError("rewrite.margin must be between 0 and 0.5")
    if s.rewrite.max_output_tokens < 16:
        raise ConfigError("rewrite.max_output_tokens must be at least 16")
    if not (1 <= s.rewrite.max_queries <= 5):
        raise ConfigError("rewrite.max_queries must be between 1 and 5")
    if not (0.0 <= s.llm.temperature <= 2.0):
        raise ConfigError("llm.temperature must be between 0 and 2")
    if s.llm.reasoning_effort is not None and \
            s.llm.reasoning_effort not in REASONING_EFFORTS:
        raise ConfigError(
            "llm.reasoning_effort must be one of "
            f"{sorted(REASONING_EFFORTS)} (got '{s.llm.reasoning_effort}')"
        )
