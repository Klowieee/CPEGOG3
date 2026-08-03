"""
web/api.py — HTTP server exposing the ChatEngine (src.chat.core) to a browser.

This is a thin sibling to scripts/run_chat.py: it reuses build_engine()
completely unchanged (per AC-4 — "a future GUI reuses this unchanged") rather
than re-implementing any retrieval/prompting/refusal logic. All it adds is a
POST /chat endpoint and a served frontend.

Usage:
    export GROQ_API_KEY="..."
    uv run python web/api.py
    # or: python web/api.py   (inside the project's venv)

Then open http://127.0.0.1:5000/ in a browser.

Requires the same prerequisites as scripts/run_chat.py: a populated
data/vector_db/ (run scripts/run_ingestion.py first) and GROQ_API_KEY set.
"""
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chat.terminal import build_engine          # noqa: E402
from src.retrieval.vector_store import StoreError    # noqa: E402
from src.utils.config import load_settings           # noqa: E402
from src.utils.logging_setup import setup_logging    # noqa: E402

WEB_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)

STATE = {"engine": None, "settings": None, "setup_error": None}


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "frontend2.html")


@app.route("/<path:filename>")
def serve_asset(filename):
    # static_folder=None disables Flask's default static handling above —
    # this catches everything else the page references by relative path
    # (logo.gif, future icons, etc.) so they don't silently 404.
    return send_from_directory(WEB_DIR, filename)


@app.route("/config")
def config_info():
    """Real pipeline settings for the frontend's info drawer — read from
    settings.yaml at startup, not hardcoded, so the UI never claims a model
    or threshold the backend isn't actually using."""
    if STATE["setup_error"]:
        return jsonify({"error": STATE["setup_error"]}), 503
    s = STATE["settings"]
    return jsonify({
        "document_title": s.document.title,
        "edition": s.document.edition,
        "embedding_model": s.embedding.model,
        "llm_model": s.llm.model,
        "llm_provider": "Groq",
        "top_k": s.retrieval.top_k,
        "similarity_floor": s.retrieval.similarity_floor,
        "hybrid_enabled": s.retrieval.hybrid.enabled,
        "rewrite_enabled": s.rewrite.enabled,
    })


@app.route("/chat", methods=["POST"])
def chat():
    if STATE["setup_error"]:
        return jsonify({
            "response": STATE["setup_error"],
            "citations": [],
            "sources": [{"type": "sql", "label": "setup required"}],
            "elapsed": 0,
        }), 503

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"response": "Please type a question.", "citations": [],
                         "sources": [], "elapsed": 0}), 400

    engine = STATE["engine"]
    start = time.time()
    answer = engine.answer_question(message)
    elapsed = round(time.time() - start, 2)

    # answer.citations are Citation(marker, citation, chunk_id) objects with
    # the [n] numbers the model itself wrote inline in answer.text — pass
    # them through as-is and let the frontend resolve [1]/[2]/... markers
    # against this list rather than re-parsing bracket text out of the answer.
    citations = [{"marker": c.marker, "citation": c.citation} for c in answer.citations]

    if answer.refused:
        sources = [{"type": "sql", "label": "not covered by the handbook"}]
    elif answer.unverified:
        sources = [{"type": "unverified", "label": "sources not confirmed by the model",
                    "citation": c.citation} for c in answer.citations]
    else:
        sources = [{"label": c.citation} for c in answer.citations]

    return jsonify({
        "response": answer.text,
        "citations": citations,
        "sources": sources,
        "elapsed": elapsed,
        "refused": answer.refused,
        "unverified": answer.unverified,
    })


def main():
    setup_logging()
    try:
        settings = load_settings()
        STATE["settings"] = settings
        print(f"Loading {settings.document.title} ({settings.document.edition}) engine...")
        STATE["engine"] = build_engine(settings)
        print("Engine ready.")
    except StoreError as exc:
        # Same failure terminal.py handles — missing/empty/mismatched index.
        # Keep the server up so the browser gets a clear message instead of
        # a connection refused, but every /chat call short-circuits until
        # ingestion is run and the process is restarted.
        STATE["setup_error"] = (
            f"Setup incomplete: {exc} Run `python scripts/run_ingestion.py` "
            "first, then restart this server."
        )
        print(f"[api] {STATE['setup_error']}")
    except Exception as exc:  # e.g. missing GROQ_API_KEY
        STATE["setup_error"] = (
            f"Could not start the chat engine: {exc}. Check that "
            "GROQ_API_KEY is set in your environment."
        )
        print(f"[api] {STATE['setup_error']}")

    print("\nOpen http://127.0.0.1:5000/ in your browser.\n")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
