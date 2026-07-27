"""
terminal.py — Interactive terminal UI and engine assembly for the chatbot.

Purpose:
    Build a ChatEngine from settings (embedder + store + retriever + API
    backend) and run the read-eval-print loop: banner, prompt, answer with
    citations, repeat, until the user exits.

Inputs:
    Settings; user keyboard input.

Outputs:
    Terminal session.

Dependencies:
    rich (formatting), src.chat.core, src.retrieval.*, src.embedding.*,
    src.llm.backend, src.utils.config.

Why this file exists:
    Separates all input/output and wiring from the reusable ChatEngine core,
    so the core can be tested and reused by a future GUI (AC-4).
"""

from __future__ import annotations

import dataclasses
import logging

from rich.console import Console
from rich.panel import Panel

from src.chat import plan_view
from src.chat.core import Answer, ChatEngine
from src.chat.rewriter import QueryRewriter
from src.llm.backend import APIBackend
from src.retrieval.retriever import build_retriever
from src.retrieval.vector_store import StoreError, VectorStore
from src.utils.config import Settings

EXIT_COMMANDS = {"exit", "quit", ":q"}
# Reserved input, dispatched before the question path. Explicit commands rather
# than intent classification: a classifier would cost an LLM call per question
# and would misroute "what are the prerequisites for CSOPESY?", which is a
# handbook question, not a planning request.
PLAN_COMMANDS = {"/plan", "/flowchart", "plan"}
HELP_COMMANDS = {"/help", "help", "?"}

log = logging.getLogger(__name__)


def build_engine(settings: Settings) -> ChatEngine:
    """Wire the full stack into a ChatEngine, validating the index first.

    Raises:
        StoreError: If the index is missing/empty/model-mismatched — surfaced
            to the user with instructions to run ingestion.
    """
    store = VectorStore(settings.paths.vector_db_dir, settings.document.id)
    store.validate(settings.embedding.model)   # loud failure before the loop
    retriever = build_retriever(settings, store=store)
    backend = APIBackend(settings.llm)

    rewriter = None
    if settings.rewrite.enabled:
        # Its own backend: same provider and model, but a much smaller output
        # budget, since providers bill the reservation against rate limits and
        # a rewrite is three short lines.
        rewrite_llm = dataclasses.replace(
            settings.llm, max_output_tokens=settings.rewrite.max_output_tokens)
        rewriter = QueryRewriter(APIBackend(rewrite_llm),
                                 max_queries=settings.rewrite.max_queries)

    return ChatEngine(retriever, backend, settings.chat.refusal_message,
                      rewriter=rewriter,
                      rescue_margin=settings.rewrite.margin,
                      planner=settings.planner)


def _render(console: Console, answer: Answer) -> None:
    """Print an answer (or refusal) with its citations."""
    console.print()
    if answer.error:
        console.print(f"[yellow]{answer.text}[/yellow]")
        return
    if answer.refused:
        console.print(f"[yellow]{answer.text}[/yellow]")
        return

    console.print(answer.text)
    if not answer.citations:
        return

    if answer.unverified:
        # Be explicit that these are the retrieved sections, not the model's
        # own attribution — citation accuracy is a non-negotiable claim
        # (docs/testing.md §4), so a weaker guarantee must be visible.
        console.print("\n[yellow]The model did not cite its sources. This "
                      "answer was drawn from:[/yellow]")
    else:
        console.print("\n[dim]Sources:[/dim]")
    for c in answer.citations:
        console.print(f"  [dim][{c.marker}] {c.citation}[/dim]")


def _render_help(console: Console) -> None:
    """What this thing can do, and what each mode is grounded in."""
    console.print()
    console.print(Panel.fit(
        "[bold]Ask a question[/bold] — anything covered by the Student "
        "Handbook. Answers quote the sections they came from, and it refuses "
        "rather than guessing when the handbook does not cover something.\n\n"
        "[bold]/plan[/bold] (or [bold]/flowchart[/bold]) — read your program "
        "checklist and work out what to take next, and in what order. Produces "
        "a term-by-term plan plus a Mermaid flowchart file. The ordering is "
        "computed, not generated: no AI model is involved, and the unit limits "
        "it applies are shown with the handbook provision each comes from.\n\n"
        "[bold]/help[/bold] — this. [bold]exit[/bold] — quit.",
        title="Commands", border_style="blue"))


def run(settings: Settings) -> None:
    """Start the interactive chatbot loop."""
    console = Console()

    try:
        engine = build_engine(settings)
    except StoreError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    console.print(Panel.fit(
        f"[bold]{settings.document.title} Assistant[/bold]\n"
        f"Edition: {settings.document.edition}\n"
        "Ask a question about the handbook, or type '/plan' to plan your\n"
        "courses from a program checklist. '/help' for more, 'exit' to quit.",
        border_style="blue",
    ))

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break

        command = question.lower()
        if command in EXIT_COMMANDS:
            console.print("Goodbye.")
            break
        if not question:
            continue
        if command in HELP_COMMANDS:
            _render_help(console)
            continue
        if command in PLAN_COMMANDS:
            try:
                plan_view.run_plan(console, engine, settings)
            except Exception as exc:      # the loop must outlive any one command
                log.exception("Course planning failed")
                console.print(f"\n[red]Course planning failed: {exc}[/red]")
            continue

        with console.status("[dim]Searching the handbook...[/dim]"):
            answer = engine.answer_question(question)
        _render(console, answer)
