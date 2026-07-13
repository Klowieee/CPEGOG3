# Week 2 — Model Selection Notes

## Candidates

| Model | Params | Pros | Cons |
|---|---|---|---|
| `distilgpt2` | 82M | Fastest to fine-tune, smallest download, runs fine on CPU | Weakest fluency/coherence, more prone to repetition |
| `gpt2` | 124M | Good balance of quality vs. size, huge community docs, easy Hugging Face support | Not built for dialogue turns, needs prompt formatting to behave like a chatbot |
| `microsoft/DialoGPT-small` | 117M | Pre-trained on conversational (Reddit) data, more "chat-like" by default | Conversational tone can drift off from formal handbook register; needs stronger fine-tuning to sound official |

## Recommendation
Start with **`gpt2`** (124M) for the first working prototype — it's the
best-documented, most predictable option, and the prompt-engineering
approach (Instruction/Question/Answer format in `prompt_template.py`) works
well with it. If generation quality is too weak after fine-tuning, try
`DialoGPT-small` as a second pass and compare BLEU scores + human ratings
between the two.

## What to record once you actually test (fill this in during Week 2/3)
- Load time on your hardware (CPU vs GPU)
- Fine-tuning time per epoch on your dataset size
- Qualitative output samples for 5 fixed test questions, per model
- BLEU score on your eval split, per model

## Why not just use a much larger model?
The brief specifically calls for small, easily fine-tunable models runnable
on modest hardware (Colab free tier or a laptop GPU). Bigger instruction-
tuned models (e.g. Llama, Mistral) would give better answers out of the box
but defeat the point of the exercise — demonstrating you can *adapt* a small
base model to a narrow domain via prompting + light fine-tuning.
