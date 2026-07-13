"""
Week 3 — Prompt engineering layer.

Even after fine-tuning, wrapping every user query in a consistent system
prompt keeps the model "on task" and makes the fine-tuned formatting
(Instruction/Question/Answer) reliable at inference time.
"""

SYSTEM_PROMPT = (
    "You are RAGML, the official student handbook assistant. "
    "You only answer questions about university policies, enrollment, "
    "grading, attendance, discipline, scholarships, and student "
    "organizations, based strictly on the student handbook. "
    "If a question is outside these topics, say you can only help with "
    "student handbook questions."
)

GENERATION_TEMPLATE = (
    "### Instruction:\n{system_prompt}\n\n"
    "### Question:\n{question}\n\n"
    "### Answer:\n"
)


def build_prompt(question: str) -> str:
    return GENERATION_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question.strip())
