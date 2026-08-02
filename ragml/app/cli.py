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
                pad_token_id=tokenizer.eos_token_id,
            )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # strip the prompt back off so we only show the new answer
        return full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

    return generate


DEFAULT_FAQ_PATH = str(Path(__file__).resolve().parent.parent / "data" / "handbook_faq_real.json")

# If a question matches a known FAQ entry closely enough, return that
# entry's real, human-written answer directly instead of letting GPT-2
# generate — this is the single biggest accuracy win available without
# fine-tuning, since it removes generation (and hallucination risk)
# entirely for anything the FAQ set already covers.
DIRECT_ANSWER_THRESHOLD = 0.35


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
    args = parser.parse_args()

    print("Loading domain guard...")
    guard_kwargs = {"faq_path": args.faq}
    if args.threshold is not None:
        guard_kwargs["threshold"] = args.threshold
    guard = DomainGuard(**guard_kwargs)

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

        prompt = build_prompt(query)
        answer = generate(prompt)
        print(f"RAGML: {answer}")
        print(f"  (generated — closest FAQ match was only {score:.2f}, below "
              f"direct-answer threshold)\n")


if __name__ == "__main__":
    main()
