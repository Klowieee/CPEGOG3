"""
inspect_parse.py — Human inspection of the parser's output (Phase 2 tool).

Purpose:
    Let a person SEE what parse_pdf() produced — role counts for the whole
    document and a line-by-line, role-tagged view of any page range. This is
    how you sanity-check parsing without reading code, and how you would
    demonstrate the parser during a presentation.

Usage:
    uv run python scripts/inspect_parse.py                # summary only
    uv run python scripts/inspect_parse.py --pages 40 41  # show pages 40-41

Inputs:
    The handbook PDF at the path in config/settings.yaml; optional page range.

Outputs:
    Printed summary and (optionally) a role-tagged line view to the terminal.

Why this file exists:
    The project values inspectable intermediate artifacts (AD-5). Parsing is
    the foundation of every later phase, so a quick way to eyeball its output
    prevents silent errors from propagating downstream.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.parser import parse_pdf, LineRole  # noqa: E402
from src.utils.config import load_settings            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect the handbook parser output.")
    ap.add_argument("--pages", nargs=2, type=int, metavar=("START", "END"),
                    help="1-indexed inclusive page range to print in detail.")
    args = ap.parse_args()

    settings = load_settings()
    pdf_path = settings.document.source_pdf
    if not pdf_path.exists():
        raise SystemExit(f"Handbook PDF not found at {pdf_path}. "
                         "Place it there or update config/settings.yaml.")

    print(f"Parsing {pdf_path.name} ...")
    pages = parse_pdf(str(pdf_path))
    print(f"Parsed {len(pages)} pages.\n")

    roles = Counter(l.role.value for p in pages for l in p.lines)
    print("Line role counts:")
    for role, n in roles.most_common():
        print(f"  {role:16} {n}")

    if args.pages:
        start, end = args.pages
        print(f"\nDetailed view of pages {start}-{end}:")
        for page in pages:
            if start <= page.page <= end:
                print(f"\n--- page {page.page} ---")
                for line in page.lines:
                    tag = line.role.value.upper()
                    print(f"  [{tag:15}] {line.text}")


if __name__ == "__main__":
    main()
