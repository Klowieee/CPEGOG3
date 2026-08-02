"""
Week 4 — Terminal chat interface for RAGML.

Pipeline per turn:
  1. DomainGuard checks if the question is about the handbook.
     - If not: return the fixed fallback message (no generation call at all).
  2. If in-domain: wrap the question in the system prompt template.
  3. Generate a response with the (fine-tuned, if available) GPT-2 model.

Usage:
    python cli.py --model gpt2
    python cli.py --model ../model/ragml-gpt2-lora   # after fine-tuning
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "model"))
from domain_guard import DomainGuard          # noqa: E402
from prompt_template import build_prompt       # noqa: E402
from handbook_retriever import HandbookRetriever  # noqa: E402


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


DEFAULT_FAQ_PATH = str(Path(__file__).resolve().parent.parent / "data" / "handbook_faq_real.json")
DEFAULT_HANDBOOK_PATH = str(Path(__file__).resolve().parent.parent / "data" / "handbook_full_text.txt")

# If a question matches a known FAQ entry closely enough, return that
# entry's real, human-written answer directly instead of letting GPT-2
# generate — this is the single biggest accuracy win available without
# fine-tuning, since it removes generation (and hallucination risk)
# entirely for anything the FAQ set already covers.
DIRECT_ANSWER_THRESHOLD = 0.30

# Below that, but before falling to raw generation: search the full
# handbook text itself (not just the 32 FAQ pairs) for the closest real
# passage. Cosine similarity on sentence embeddings runs on a different
# scale than the FAQ's TF-IDF score, so this threshold is tuned separately.
HANDBOOK_MATCH_THRESHOLD = 0.45


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
    args = parser.parse_args()

    print("Loading domain guard...")
    guard_kwargs = {"faq_path": args.faq}
    if args.threshold is not None:
        guard_kwargs["threshold"] = args.threshold
    guard = DomainGuard(**guard_kwargs)

    print("Loading handbook retriever (first run builds+caches the index)...")
    retriever = HandbookRetriever(text_path=args.handbook)

    print(f"Loading model: {args.model} (this may take a moment)")
    generate = load_generator(args.model)

    print("\nRAGML ready. Ask a student handbook question, or type 'exit' to quit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not query:
            continue

        in_domain, fallback = guard.check(query)
        if not in_domain:
            print(f"RAGML: {fallback}\n")
            continue

        matched_item, score = guard.top_match(query)
        if score >= args.direct_threshold:
            print(f"RAGML: {matched_item['answer']}")
            print(f"  (matched FAQ, similarity {score:.2f}: \"{matched_item['question']}\")\n")
            continue

        label, passage, hb_score = retriever.top_match(query)
        if passage and hb_score >= args.handbook_threshold:
            print(f"RAGML: {passage}")
            print(f"  (matched handbook passage, similarity {hb_score:.2f}: {label})\n")
            continue

        prompt = build_prompt(query)
        answer = generate(prompt)
        if not answer:
            answer = ("I don't have a confident answer for that. The closest handbook "
                       f"topic I found was \"{matched_item['question']}\" — try rephrasing "
                       "closer to that, or check with your adviser directly.")
        print(f"RAGML: {answer}")
        print(f"  (generated — no close FAQ or handbook-passage match, closest was "
              f"FAQ={score:.2f} handbook={hb_score:.2f})\n")


if __name__ == "__main__":
    main()
