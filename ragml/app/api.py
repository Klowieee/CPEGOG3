"""
api.py — HTTP server that connects frontend2.html to the RAGML pipeline.

This wires up the same pieces cli.py uses (DomainGuard, prompt_template,
the HF model) behind a POST /chat endpoint, and serves frontend2.html as
the root page, so opening one URL gives you the working chat UI.

Usage:
    python api.py --model gpt2
    python api.py --model ../model/ragml-gpt2-lora   # after fine-tuning

Then open http://127.0.0.1:5000/ in a browser.
"""
import argparse
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

sys.path.append(str(Path(__file__).resolve().parent.parent / "model"))
from domain_guard import DomainGuard          # noqa: E402
from prompt_template import build_prompt       # noqa: E402
from handbook_retriever import HandbookRetriever  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)

# If a question matches a known FAQ entry closely enough, return that
# entry's real, human-written answer directly instead of letting GPT-2
# generate — removes generation (and hallucination risk) entirely for
# anything the FAQ set already covers. Overridable via STATE at startup.
DIRECT_ANSWER_THRESHOLD = 0.30

# Below that, but before falling to raw generation: search the full
# handbook text itself (not just the FAQ pairs) for the closest real
# passage. Different similarity scale than the FAQ's TF-IDF score, so
# tuned separately.
HANDBOOK_MATCH_THRESHOLD = 0.45

# Populated in main() once the model/guard/retriever are loaded, so a
# single worker process holds them in memory instead of reloading per-request.
STATE = {"guard": None, "generate": None, "retriever": None,
         "direct_threshold": DIRECT_ANSWER_THRESHOLD,
         "handbook_threshold": HANDBOOK_MATCH_THRESHOLD}


def load_generator(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
            )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        answer = full_text[len(prompt_text):].strip()
        # Untuned GPT-2 doesn't know to stop after one answer — it often
        # keeps pattern-completing the template into a fake follow-up
        # "### Question:" turn. Cut off anything after the first such marker.
        for marker in ("###", "Question:", "Instruction:"):
            idx = answer.find(marker)
            if idx != -1:
                answer = answer[:idx].strip()
        return answer

    return generate


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "frontend2.html")


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"response": "Please type a question.", "citations": [],
                         "sources": [], "elapsed": 0}), 400

    guard = STATE["guard"]
    generate = STATE["generate"]
    start = time.time()

    in_domain, fallback = guard.check(message)
    if not in_domain:
        elapsed = round(time.time() - start, 2)
        return jsonify({
            "response": fallback,
            "citations": [],
            "sources": [{"type": "sql", "label": "out of scope"}],
            "elapsed": elapsed,
        })

    matched_item, score = guard.top_match(message)

    if score >= STATE["direct_threshold"]:
        # Close FAQ match — return the real, human-written answer directly.
        # No generation call at all, so no hallucination risk here.
        elapsed = round(time.time() - start, 2)
        sources = [{
            "id": matched_item.get("question", ""),
            "label": matched_item.get("category", "handbook"),
            "preview": matched_item.get("answer", "")[:160],
        }]
        return jsonify({
            "response": matched_item.get("answer", ""),
            "citations": [],
            "sources": sources,
            "elapsed": elapsed,
        })

    # No close FAQ match — try the full handbook text before generating.
    retriever = STATE["retriever"]
    label, passage, hb_score = retriever.top_match(message)

    if passage and hb_score >= STATE["handbook_threshold"]:
        elapsed = round(time.time() - start, 2)
        sources = [{
            "id": label,
            "label": label,
            "preview": passage[:160],
        }]
        return jsonify({
            "response": passage,
            "citations": [],
            "sources": sources,
            "elapsed": elapsed,
        })

    # No close FAQ or handbook match either — fall back to generation,
    # flagged as such via the source chip's low-confidence label.
    prompt = build_prompt(message)
    answer = generate(prompt)
    if not answer:
        answer = ("I don't have a confident answer for that. The closest handbook "
                   f"topic I found was \"{matched_item.get('question', '')}\" — try "
                   "rephrasing closer to that, or check with your adviser directly.")

    sources = [{
        "id": matched_item.get("question", ""),
        "label": f"{matched_item.get('category', 'handbook')} (generated, low match)",
        "preview": matched_item.get("answer", "")[:160],
    }]

    elapsed = round(time.time() - start, 2)
    return jsonify({
        "response": answer,
        "citations": [],
        "sources": sources,
        "elapsed": elapsed,
    })


DEFAULT_FAQ_PATH = str(Path(__file__).resolve().parent.parent / "data" / "handbook_faq_real.json")
DEFAULT_HANDBOOK_PATH = str(Path(__file__).resolve().parent.parent / "data" / "handbook_full_text.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2",
                         help="Hugging Face model name or path to fine-tuned checkpoint")
    parser.add_argument("--faq", default=DEFAULT_FAQ_PATH)
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override the domain-guard similarity threshold")
    parser.add_argument("--direct-threshold", type=float, default=DIRECT_ANSWER_THRESHOLD,
                         help="Similarity above which a matched FAQ answer is "
                              "returned directly instead of generated")
    parser.add_argument("--handbook", default=DEFAULT_HANDBOOK_PATH,
                         help="Path to the full handbook text for whole-document retrieval")
    parser.add_argument("--handbook-threshold", type=float, default=HANDBOOK_MATCH_THRESHOLD,
                         help="Similarity above which a matched handbook passage is "
                              "returned directly instead of generated")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print("Loading domain guard...")
    guard_kwargs = {"faq_path": args.faq}
    if args.threshold is not None:
        guard_kwargs["threshold"] = args.threshold
    STATE["guard"] = DomainGuard(**guard_kwargs)
    STATE["direct_threshold"] = args.direct_threshold
    STATE["handbook_threshold"] = args.handbook_threshold

    print("Loading handbook retriever (first run builds+caches the index)...")
    STATE["retriever"] = HandbookRetriever(text_path=args.handbook)

    print(f"Loading model: {args.model} (this may take a moment)")
    STATE["generate"] = load_generator(args.model)

    print(f"\nRAGML API ready — open http://{args.host}:{args.port}/ in your browser.\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
