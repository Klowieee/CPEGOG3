"""
logging_setup.py — One-line logging configuration for scripts.

Purpose:
    Give every script and module a consistent, readable log format and a
    single switch for verbosity. Modules obtain loggers with
    `logging.getLogger(__name__)`; only entry-point scripts call
    `setup_logging()`.

Inputs:
    Optional verbosity flag.

Outputs:
    Configures the root logger (side effect); returns nothing.

Dependencies:
    Standard library only.

Why this file exists:
    Ingestion is a multi-stage offline pipeline; when a stage misbehaves
    (e.g., the parser detects too few headings) the log trail is the
    primary debugging tool, so it should look the same everywhere.
"""

from __future__ import annotations

import logging


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging for a script run.

    Args:
        verbose: If True, show DEBUG messages; otherwise INFO and above.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Third-party libraries can be chatty at INFO; keep them at WARNING.
    for noisy in ("httpx", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
