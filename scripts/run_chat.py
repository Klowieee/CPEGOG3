"""
run_chat.py — Entry point for the terminal chatbot.

Purpose:
    Launch the interactive question-answer loop against the prebuilt index.

Inputs:
    A populated data/vector_db/ (from run_ingestion.py), settings.yaml, and
    the API key in the environment variable named by llm.api_key_env.

Outputs:
    Interactive terminal session.

Dependencies:
    src.chat.terminal, src.utils.

Why this file exists:
    Thin entry point per AC-4; all logic lives in src/ so a future GUI can
    reuse the ChatEngine core unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat import terminal                       # noqa: E402
from src.utils.config import load_settings          # noqa: E402
from src.utils.logging_setup import setup_logging   # noqa: E402


def main() -> None:
    setup_logging()
    settings = load_settings()
    terminal.run(settings)


if __name__ == "__main__":
    main()
