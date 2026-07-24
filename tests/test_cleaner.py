"""
test_cleaner.py — Tests for src/ingestion/cleaner.py (Phase 3).

Fast synthetic unit tests for each cleaning rule, plus one integration test
that cleans the real parsed handbook (skipped if the PDF is absent) and
asserts the invariants we care about: front matter is gone, page numbers are
gone, all 21 section headings survive, and no core policy is lost.

Dependencies:
    pytest, src.ingestion.parser, src.ingestion.cleaner.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.cleaner import clean, write_cleaned_jsonl  # noqa: E402
from src.ingestion.parser import (  # noqa: E402
    Line,
    LineRole,
    PageContent,
    parse_pdf,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDBOOK = PROJECT_ROOT / "data" / "handbooks" / "student-handbook.pdf"


def L(text, role=LineRole.BODY, page=20, x0=202.0, size=10.0, **kw) -> Line:
    return Line(text=text, page=page, font_size=size, font_name="F", x0=x0,
                role=role, **kw)


def make_pages(*page_specs) -> list[PageContent]:
    """page_specs: (page_number, [lines]) tuples."""
    return [PageContent(page=n, lines=list(lines)) for n, lines in page_specs]


# --- Front-matter removal ------------------------------------------------------

def test_front_matter_before_lasallian_values_is_dropped():
    pages = make_pages(
        (8, [L("Message from", role=LineRole.CHAPTER_TITLE, size=28)]),
        (16, [L("Lasallian Values", role=LineRole.CHAPTER_TITLE, size=28)]),
        (18, [L("General Provisions", role=LineRole.CHAPTER_TITLE, size=28)]),
        (18, [L("1.1 Students must comply.", role=LineRole.PROVISION, x0=90,
                provision_number="1.1")]),
    )
    cleaned = clean(pages)
    kept_pages = {p.page for p in cleaned}
    assert 8 not in kept_pages          # President's message dropped
    assert 16 in kept_pages             # Lasallian Values kept
    assert 18 in kept_pages             # main content kept


def test_fallback_to_general_provisions_when_no_lasallian_divider():
    pages = make_pages(
        (8, [L("Message from", role=LineRole.CHAPTER_TITLE, size=28)]),
        (18, [L("General Provisions", role=LineRole.CHAPTER_TITLE, size=28)]),
    )
    cleaned = clean(pages)
    assert {p.page for p in cleaned} == {18}


# --- Per-line filtering --------------------------------------------------------

def test_page_numbers_removed():
    pages = make_pages(
        (18, [L("General Provisions", role=LineRole.CHAPTER_TITLE, size=28),
              L("Real body text here.", role=LineRole.BODY),
              L("60", role=LineRole.PAGE_NUMBER, x0=39)]),
    )
    cleaned = clean(pages)
    texts = [l.text for p in cleaned for l in p.lines]
    assert "60" not in texts
    assert "Real body text here." in texts


def test_numeric_chapter_banner_removed_but_named_title_kept():
    pages = make_pages(
        (18, [L("1", role=LineRole.CHAPTER_TITLE, size=37, x0=60),
              L("General Provisions", role=LineRole.CHAPTER_TITLE, size=28, x0=60)]),
    )
    cleaned = clean(pages)
    titles = [l.text for p in cleaned for l in p.lines
              if l.role is LineRole.CHAPTER_TITLE]
    assert "1" not in titles
    assert "General Provisions" in titles


def test_substantial_other_kept_as_body_short_other_dropped():
    pages = make_pages(
        (18, [L("General Provisions", role=LineRole.CHAPTER_TITLE, size=28)]),
        (200, [
            L("The Business Doctoral Society of DLSU is a student organization.",
              role=LineRole.OTHER, size=8),          # 10 words -> kept as BODY
            L("Policy.", role=LineRole.OTHER, size=8),  # 1 word  -> dropped
        ]),
    )
    cleaned = clean(pages)
    lines = [l for p in cleaned for l in p.lines]
    kept_texts = [l.text for l in lines]
    assert any("Business Doctoral Society" in t for t in kept_texts)
    assert "Policy." not in kept_texts
    # The kept small-type line is relabeled to BODY.
    org_line = next(l for l in lines if "Business Doctoral" in l.text)
    assert org_line.role is LineRole.BODY


# --- Persistence ---------------------------------------------------------------

def test_write_cleaned_jsonl(tmp_path):
    import json
    pages = make_pages(
        (18, [L("General Provisions", role=LineRole.CHAPTER_TITLE, size=28),
              L("5.1 A rule.", role=LineRole.PROVISION, x0=90,
                provision_number="5.1")]),
    )
    cleaned = clean(pages)
    out = tmp_path / "cleaned.jsonl"
    n = write_cleaned_jsonl(cleaned, out)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert n == len(records) == 2
    prov = next(r for r in records if r["role"] == "provision")
    assert prov["provision_number"] == "5.1"


# --- Integration on the real handbook ------------------------------------------

@pytest.mark.skipif(not HANDBOOK.exists(), reason="handbook PDF not present")
def test_clean_real_handbook():
    pages = parse_pdf(str(HANDBOOK))
    cleaned = clean(pages)

    kept_page_numbers = {p.page for p in cleaned}
    # Front matter (cover p5, President p8, founder p10) must be gone.
    assert kept_page_numbers.isdisjoint({5, 8, 10, 12, 14})
    # Content pages present.
    assert 18 in kept_page_numbers and 101 in kept_page_numbers

    # All 21 section headings survive cleaning.
    sections = [l for p in cleaned for l in p.lines
                if l.role is LineRole.SECTION_HEADING]
    assert len(sections) == 21

    # No page-number lines remain.
    assert not any(l.role is LineRole.PAGE_NUMBER
                   for p in cleaned for l in p.lines)

    # Core policy is preserved: a known plagiarism provision is still present.
    all_text = " ".join(l.text for p in cleaned for l in p.lines)
    assert "Plagiarism" in all_text
