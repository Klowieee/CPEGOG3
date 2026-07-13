"""
Week 3 — Lightweight fine-tuning with LoRA (via peft) on top of GPT-2.

LoRA only trains a small set of adapter weights instead of the whole model,
which makes this realistic to run on a single free-tier GPU (e.g. Colab T4)
or even CPU for a quick smoke test with --dry-run.

Usage:
    # Quick check that everything loads correctly, no real training:
    python finetune_gpt2.py --dry-run

    # Real training run:
    python finetune_gpt2.py \
        --train-file ../data/processed/train.jsonl \
        --eval-file ../data/processed/eval.jsonl \
        --model-name gpt2 \
        --output-dir ./ragml-gpt2-lora \
        --epochs 3
"""
import argparse
import json


def load_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--train-file", default="../data/processed/train.jsonl")
    parser.add_argument("--eval-file", default="../data/processed/eval.jsonl")
    parser.add_argument("--output-dir", default="./ragml-gpt2-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dry-run", action="store_true",
                         help="Only checks that libraries and model load; skips training.")
    args = parser.parse_args()

    # Imports are deferred so --dry-run still gives a useful error message
    # if a package is missing, rather than failing on file loading first.
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: {}. Run `pip install -r ../requirements.txt` first.".format(e)
        )

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],  # GPT-2's attention projection layer name
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.dry_run:
        print("Dry run complete: model + tokenizer + LoRA config all loaded successfully.")
        return

    train_examples = load_jsonl(args.train_file)
    eval_examples = load_jsonl(args.eval_file)

    train_ds = Dataset.from_list(train_examples)
    eval_ds = Dataset.from_list(eval_examples)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256, padding="max_length")

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=eval_ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
