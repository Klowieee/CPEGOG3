"""
handbook_retriever.py — Whole-handbook semantic retrieval.

This is the third tier of the pipeline, sitting between the FAQ shortcut
and raw generation:

  1. DomainGuard.check()      -> reject off-topic questions
  2. DomainGuard.top_match()  -> close FAQ match? return its real answer
  3. HandbookRetriever.search() -> THIS FILE. No close FAQ match, but the
     question is in-domain — search all 339 pages of the handbook itself
     (not just the 32 hand-picked FAQ pairs) for the closest real passage,
     and return that instead of letting GPT-2 free-generate.
  4. Only if nothing above found anything usable does it fall to raw
     generation.

Approach: chunk the raw text along its own numbered-clause structure,
embed every chunk once with a small sentence-transformers model, and
cache the result to disk so startup after the first run is instant.

Usage:
    retriever = HandbookRetriever(text_path="../data/handbook_full_text.txt")
    label, passage, score = retriever.top_match("what is the maximum failure units in gcoe")
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# A new clause starts a new chunk boundary — matches lines like
# "10.17.1", "4.2", "13.6" at the start of a (stripped) line.
_CLAUSE_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\s+\S")
_SECTION_RE = re.compile(r"^\s*Section\s+\d+[:.]?\s*(.+)$", re.IGNORECASE)
_APPENDIX_RE = re.compile(r"^\s*APPENDIX\s+([A-Z])\b(.*)$", re.IGNORECASE)
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")

MIN_CHUNK_CHARS = 220   # merge short clauses together up to roughly this size
MAX_CHUNK_CHARS = 900   # hard cap so one chunk doesn't swallow a whole section


def _clean_lines(raw_text: str):
    """Strip form-feed page breaks, bare page-number lines, and normalize
    line endings, without disturbing the actual policy text."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    lines = []
    for line in text.split("\n"):
        if _PAGE_NUM_RE.match(line):
            continue
        lines.append(line.rstrip())
    return lines


def chunk_handbook(raw_text: str):
    """Splits the handbook into clause-aware chunks.

    Returns a list of {"label": str, "text": str} dicts. `label` is the
    nearest Section/Appendix heading, used for the source chip in the UI.
    """
    lines = _clean_lines(raw_text)

    chunks = []
    current_label = "General"
    buf = []

    def flush():
        text = "\n".join(buf).strip()
        text = re.sub(r"\n{2,}", "\n", text)
        if text:
            chunks.append({"label": current_label, "text": text})
        buf.clear()

    for line in lines:
        stripped = line.strip()

        section_m = _SECTION_RE.match(stripped)
        appendix_m = _APPENDIX_RE.match(stripped)
        if section_m or appendix_m:
            flush()
            current_label = stripped[:80]
            continue

        if not stripped:
            # Blank line: safe place to close a chunk if it's already
            # grown past the minimum size.
            if sum(len(b) for b in buf) >= MIN_CHUNK_CHARS:
                flush()
            continue

        if _CLAUSE_RE.match(stripped) and sum(len(b) for b in buf) >= MIN_CHUNK_CHARS:
            flush()

        buf.append(stripped)
        if sum(len(b) for b in buf) >= MAX_CHUNK_CHARS:
            flush()

    flush()

    # Drop near-empty or boilerplate chunks (headers, TOC noise, etc.)
    chunks = [c for c in chunks if len(c["text"]) >= 40]
    return chunks


class HandbookRetriever:
    def __init__(self, text_path: str, cache_dir: str = None,
                 model_name: str = DEFAULT_MODEL_NAME):
        self.text_path = Path(text_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.text_path.parent / "processed"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self._model = None  # lazy-loaded — only downloaded/loaded if cache misses
        self.chunks = []
        self.embeddings = None  # np.ndarray, shape (n_chunks, dim)

        self._load_or_build()

    # ── Cache handling ──────────────────────────────────────────────
    def _source_hash(self) -> str:
        raw = self.text_path.read_bytes()
        return hashlib.sha256(raw).hexdigest()[:16]

    def _cache_paths(self, source_hash: str):
        base = f"handbook_index_{source_hash}"
        return (self.cache_dir / f"{base}_chunks.json",
                self.cache_dir / f"{base}_embeddings.npy")

    def _load_or_build(self):
        source_hash = self._source_hash()
        chunks_path, emb_path = self._cache_paths(source_hash)

        if chunks_path.exists() and emb_path.exists():
            self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            self.embeddings = np.load(emb_path)
            return

        print("[HandbookRetriever] No cache found — chunking and embedding "
              "the full handbook text (one-time cost, cached afterward)...")
        raw_text = self.text_path.read_text(encoding="utf-8", errors="ignore")
        self.chunks = chunk_handbook(raw_text)

        model = self._get_model()
        texts = [c["text"] for c in self.chunks]
        self.embeddings = model.encode(
            texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True,
        )
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)

        chunks_path.write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
        np.save(emb_path, self.embeddings)
        print(f"[HandbookRetriever] Indexed {len(self.chunks)} chunks -> {chunks_path.name}")

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ── Search ──────────────────────────────────────────────────────
    def search(self, query: str, k: int = 3):
        """Returns up to k (label, text, score) tuples, best first."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        model = self._get_model()
        q_vec = model.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q_vec  # cosine sim, since both are normalized
        top_idx = np.argsort(sims)[::-1][:k]
        return [(self.chunks[i]["label"], self.chunks[i]["text"], float(sims[i]))
                for i in top_idx]

    def top_match(self, query: str):
        """Returns (label, text, score) for the single closest chunk."""
        results = self.search(query, k=1)
        if not results:
            return None, None, 0.0
        return results[0]


if __name__ == "__main__":
    # quick manual test
    retriever = HandbookRetriever(text_path="../data/handbook_full_text.txt")
    for q in ["maximum units gcoe", "unit failure", "student attire policy",
              "can I bring my laptop to class"]:
        label, text, score = retriever.top_match(q)
        print(f"\n[{score:.3f}] {q!r} -> {label}")
        print(f"   {text[:200]}...")
