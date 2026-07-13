"""
Week 1 utility: turn a raw FAQ JSON (see handbook_faq_sample.json) into
JSONL training examples for GPT-2 style causal LM fine-tuning, plus a
held-out split for BLEU evaluation later.

Usage:
    python prepare_dataset.py --input handbook_faq_sample.json --outdir ./processed
"""
import json
import argparse
import random
from pathlib import Path

PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "You are a helpful assistant that only answers questions about the "
    "student handbook. Answer the question below using handbook policy.\n\n"
    "### Question:\n{question}\n\n"
    "### Answer:\n{answer}"
)


def load_faq(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_training_examples(faq_items):
    examples = []
    for item in faq_items:
        text = PROMPT_TEMPLATE.format(
            question=item["question"].strip(),
            answer=item["answer"].strip(),
        )
        examples.append({"text": text, "category": item.get("category", "general")})
    return examples


def split_train_eval(examples, eval_ratio=0.2, seed=42):
    random.Random(seed).shuffle(examples)
    n_eval = max(1, int(len(examples) * eval_ratio))
    return examples[n_eval:], examples[:n_eval]


def write_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="handbook_faq_sample.json")
    parser.add_argument("--outdir", default="./processed")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    faq_items = load_faq(args.input)
    examples = to_training_examples(faq_items)
    train, eval_ = split_train_eval(examples, args.eval_ratio)

    write_jsonl(train, Path(args.outdir) / "train.jsonl")
    write_jsonl(eval_, Path(args.outdir) / "eval.jsonl")

    print(f"Wrote {len(train)} training examples and {len(eval_)} eval examples to {args.outdir}")
    print("NOTE: with only a handful of FAQ pairs, fine-tuning will overfit fast.")
    print("Aim for at least 100-300 Q&A pairs pulled from the real handbook before training.")


if __name__ == "__main__":
    main()
