"""
run_ingestion.py — Entry point for the offline ingestion pipeline.

Purpose:
    Execute the ingestion stages in sequence. Stages are added phase by
    phase; each new phase extends this script by one step so the pipeline
    can always be run and inspected up to the latest implemented stage.

    Implemented so far:
      Stage 1 (Phase 2) — parse the PDF into role-classified lines.
      Stage 2 (Phase 3) — clean out non-content; write cleaned.jsonl.
      Stage 3 (Phase 4) — chunk into citable units; write chunks.jsonl.
      Stage 4 (Phases 5-6) — embed all chunks locally and store them in a
        persistent ChromaDB collection (drops and recreates: idempotent).

Inputs:
    The PDF at document.source_pdf and parameters in config/settings.yaml.

Outputs:
    data/processed/cleaned.jsonl and data/processed/chunks.jsonl
    (inspectable intermediates) and a populated ChromaDB collection under
    data/vector_db/ (the chatbot's index).

Dependencies:
    src.ingestion, src.utils.

Why this file exists:
    Architectural Decision AD-1: ingestion is a separate one-time pipeline
    so the chat application starts instantly against a prebuilt index.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.chunker import chunk, write_chunks_jsonl    # noqa: E402
from src.ingestion.cleaner import clean, write_cleaned_jsonl  # noqa: E402
from src.embedding.embedder import Embedder                   # noqa: E402
from src.ingestion.parser import parse_pdf                     # noqa: E402
from src.retrieval.vector_store import VectorStore             # noqa: E402
from src.utils.config import load_settings                     # noqa: E402
from src.utils.logging_setup import setup_logging              # noqa: E402
from src.utils.tokens import TokenCounter                      # noqa: E402

log = logging.getLogger("ingestion")


def main() -> None:
    setup_logging()
    settings = load_settings()

    pdf_path = settings.document.source_pdf
    if not pdf_path.exists():
        raise SystemExit(
            f"Handbook PDF not found at {pdf_path}. "
            "Place it there or update config/settings.yaml."
        )

    # Stage 1 — parse.
    log.info("Stage 1/4: parsing %s ...", pdf_path.name)
    pages = parse_pdf(str(pdf_path))
    log.info("Parsed %d pages.", len(pages))

    # Stage 2 — clean and persist.
    log.info("Stage 2/4: cleaning ...")
    cleaned = clean(pages)
    kept_lines = sum(len(p.lines) for p in cleaned)
    out_path = settings.paths.processed_dir / "cleaned.jsonl"
    written = write_cleaned_jsonl(cleaned, out_path)
    log.info(
        "Kept %d content lines across %d pages; wrote %d records to %s",
        kept_lines, len(cleaned), written, out_path,
    )

    # Stage 3 — chunk and persist.
    log.info("Stage 3/4: chunking ...")
    counter = TokenCounter(settings.embedding.model)
    chunks = chunk(cleaned, settings.chunking, counter, settings.document.id)
    chunks_path = settings.paths.processed_dir / "chunks.jsonl"
    write_chunks_jsonl(chunks, chunks_path)
    sizes = sorted(c.token_count for c in chunks)
    log.info(
        "Built %d chunks (min=%d, median=%d, max=%d tokens); wrote %s",
        len(chunks), sizes[0], sizes[len(sizes) // 2], sizes[-1], chunks_path,
    )

    # Stage 4 — embed and store (atomic: an embedded-but-unstored corpus
    # is useless, so these always happen together).
    log.info("Stage 4/4: embedding %d chunks with %s (first run downloads "
             "the model) ...", len(chunks), settings.embedding.model)
    embedder = Embedder(settings.embedding.model, settings.embedding.query_prefix)
    vectors = embedder.embed_texts([c.text for c in chunks])
    store = VectorStore(settings.paths.vector_db_dir, settings.document.id)
    stored = store.rebuild(chunks, vectors, settings.embedding.model)
    log.info("Stored %d chunks (dim=%d) in %s",
             stored, vectors.shape[1], settings.paths.vector_db_dir)

    print(
        f"\nIngestion complete: {stored} chunks parsed, cleaned, chunked, "
        "embedded, and stored. The index is ready — next phases build the "
        "retriever and chat interface on top of it."
    )


if __name__ == "__main__":
    main()
