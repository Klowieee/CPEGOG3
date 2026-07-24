"""
test_parser.py — Tests for src/ingestion/parser.py (Phase 2).

Two layers:
  * Fast unit tests exercise the classification and line-reconstruction
    logic with synthetic inputs — no PDF, instant, deterministic.
  * One integration test parses the real handbook (skipped automatically if
    the PDF is absent) and asserts the structural invariants we verified
    during design: 21 section headings, the five chapter dividers, and a
    large number of positionally-detected provisions.

Dependencies:
    pytest, src.ingestion.parser.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.parser import (  # noqa: E402
    Line,
    LineRole,
    _classify,
    _reconstruct_lines,
    parse_pdf,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDBOOK = PROJECT_ROOT / "data" / "handbooks" / "student-handbook.pdf"


def make_line(text, size=10.0, font="Degular-Regular", x0=202.0) -> Line:
    """Build a Line with handbook-typical defaults for classification tests."""
    return Line(text=text, page=1, font_size=size, font_name=font, x0=x0)


# --- Classification unit tests -------------------------------------------------

def test_section_heading_detected():
    line = make_line("SECTION 5: LASALLIAN COMMUNITY STANDARDS", size=14.0,
                     font="Degular-Bold", x0=60.0)
    _classify(line)
    assert line.role is LineRole.SECTION_HEADING
    assert line.section_number == "5"
    assert line.section_title == "LASALLIAN COMMUNITY STANDARDS"


def test_chapter_title_detected():
    line = make_line("Undergraduate", size=28.0, font="DegularText-Bold", x0=60.0)
    _classify(line)
    assert line.role is LineRole.CHAPTER_TITLE


def test_provision_detected_by_pattern_and_position():
    # Number in the left column (x0 well below the body column) → provision.
    line = make_line("5.3.1.2 Vandalism or the deliberate destruction of property",
                     x0=111.0)
    _classify(line)
    assert line.role is LineRole.PROVISION
    assert line.provision_number == "5.3.1.2"


def test_deep_provision_number_detected():
    line = make_line("5.3.1.37.2 Computer related offenses", x0=160.0)
    _classify(line)
    assert line.role is LineRole.PROVISION
    assert line.provision_number == "5.3.1.37.2"


def test_number_in_body_column_is_not_a_provision():
    # A numeric-looking token that starts in the BODY column is body text,
    # not a provision label — this guards against false positives.
    line = make_line("1.5 million pesos were allocated", x0=202.0)
    _classify(line)
    assert line.role is LineRole.BODY


def test_standalone_page_number():
    line = make_line("60", size=12.0, font="Degular-Light", x0=350.0)
    _classify(line)
    assert line.role is LineRole.PAGE_NUMBER


def test_plain_body_text():
    line = make_line("Students are required to comply with the provisions.")
    _classify(line)
    assert line.role is LineRole.BODY


def test_fine_print_is_other():
    line = make_line("footnote reference text", size=7.0)
    _classify(line)
    assert line.role is LineRole.OTHER


# --- Line reconstruction unit test ---------------------------------------------

def test_reconstruct_joins_words_on_same_baseline():
    # A provision number (left column) and body text share a baseline (top);
    # reconstruction must join them into one left-to-right ordered line.
    words = [
        {"text": "Plagiarism;", "x0": 202.9, "top": 57.9, "size": 10.0,
         "fontname": "ABC+Degular-Regular"},
        {"text": "5.3.1.1.6", "x0": 160.1, "top": 57.9, "size": 10.0,
         "fontname": "ABC+Degular-Regular"},
        {"text": "and", "x0": 249.9, "top": 57.9, "size": 10.0,
         "fontname": "ABC+Degular-Regular"},
    ]
    lines = _reconstruct_lines(words, page_number=60)
    assert len(lines) == 1
    assert lines[0].text == "5.3.1.1.6 Plagiarism; and"
    assert lines[0].x0 == pytest.approx(160.1)


def test_reconstruct_empty_page():
    assert _reconstruct_lines([], page_number=1) == []


# --- Integration test on the real handbook -------------------------------------

@pytest.mark.skipif(not HANDBOOK.exists(),
                    reason="handbook PDF not present (place it in data/handbooks/)")
def test_parse_real_handbook_structure():
    pages = parse_pdf(str(HANDBOOK))
    assert len(pages) == 339

    sections = [l for p in pages for l in p.lines
                if l.role is LineRole.SECTION_HEADING]
    # The handbook has 21 top-level numbered sections.
    numbers = [s.section_number for s in sections]
    assert numbers == [str(n) for n in range(1, 22)]

    chapter_titles = {l.text for p in pages for l in p.lines
                      if l.role is LineRole.CHAPTER_TITLE}
    for expected in ("General Provisions", "Undergraduate", "Graduate",
                     "Student Activities", "Appendices"):
        assert expected in chapter_titles

    provisions = [l for p in pages for l in p.lines
                  if l.role is LineRole.PROVISION]
    # Positional detection finds far more than plain-text extraction (~200).
    assert len(provisions) > 500
