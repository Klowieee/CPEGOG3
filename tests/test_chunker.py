"""
test_chunker.py — Tests for src/chunking/chunker.py and src/utils/tokens.py.

Fast synthetic tests exercise each pass (segmentation, section-boundary
grouping, merge, line-based split with overlap) plus citation rendering.
The integration test runs the full parse→clean→chunk pipeline on the real
handbook (skipped without the PDF) and asserts the invariants promised in
docs/chunking_strategy.md §5: chunk count in the estimated band, size cap
respected, all 21 sections represented, complete page coverage, and correct
part attribution for a known provision.

Dependencies:
    pytest, src.chunking.chunker, src.ingestion, src.utils.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.chunker import Chunk, chunk, write_chunks_jsonl  # noqa: E402
from src.ingestion.cleaner import clean                             # noqa: E402
from src.ingestion.parser import (                                  # noqa: E402
    Line,
    LineRole,
    PageContent,
    parse_pdf,
)
from src.utils.config import ChunkingSettings, load_settings        # noqa: E402
from src.utils.tokens import TokenCounter                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDBOOK = PROJECT_ROOT / "data" / "handbooks" / "student-handbook.pdf"

SETTINGS = ChunkingSettings(
    target_tokens=350, max_tokens=500, min_tokens=80, overlap_tokens=50
)
COUNTER = TokenCounter()  # heuristic mode: deterministic, dependency-free


def L(text, role=LineRole.BODY, page=20, **kw) -> Line:
    return Line(text=text, page=page, font_size=10.0, font_name="F",
                x0=202.0, role=role, **kw)


def pages_of(*lines) -> list[PageContent]:
    return [PageContent(page=lines[0].page, lines=list(lines))]


def make_doc(chapter, section_num, section_title, provisions) -> list[PageContent]:
    """One page: chapter title, section heading, then (label, text) provisions."""
    lines = [
        L(chapter, role=LineRole.CHAPTER_TITLE),
        L(f"SECTION {section_num}: {section_title}", role=LineRole.SECTION_HEADING,
          section_number=section_num, section_title=section_title),
    ]
    for label, text in provisions:
        lines.append(L(f"{label} {text}", role=LineRole.PROVISION,
                       provision_number=label))
    return pages_of(*lines)


# --- Merge behavior ------------------------------------------------------------

def test_small_provisions_merge_into_one_chunk():
    doc = make_doc("Undergraduate", "10", "GRADING",
                   [("10.1", "Rule one text."), ("10.2", "Rule two text."),
                    ("10.3", "Rule three text.")])
    chunks = chunk(doc, SETTINGS, COUNTER, "doc")
    assert len(chunks) == 1
    c = chunks[0]
    assert c.provisions == ["10.1", "10.2", "10.3"]
    assert c.text.startswith("Undergraduate › Section 10: GRADING\n")
    assert "Rule two text." in c.text


def test_chunks_never_cross_section_boundary():
    lines = [
        L("Undergraduate", role=LineRole.CHAPTER_TITLE),
        L("SECTION 9: EXAMS", role=LineRole.SECTION_HEADING,
          section_number="9", section_title="EXAMS"),
        L("9.1 Short exam rule.", role=LineRole.PROVISION, provision_number="9.1"),
        L("SECTION 10: GRADING", role=LineRole.SECTION_HEADING,
          section_number="10", section_title="GRADING"),
        L("10.1 Short grading rule.", role=LineRole.PROVISION,
          provision_number="10.1"),
    ]
    chunks = chunk(pages_of(*lines), SETTINGS, COUNTER, "doc")
    # Both provisions are tiny, but they must land in separate chunks.
    assert len(chunks) == 2
    assert chunks[0].section_number == "9" and chunks[1].section_number == "10"


def test_part_reset_on_new_chapter():
    lines = [
        L("Undergraduate", role=LineRole.CHAPTER_TITLE),
        L("SECTION 13: DISCONTINUANCE", role=LineRole.SECTION_HEADING,
          section_number="13", section_title="DISCONTINUANCE"),
        L("13.1 A rule.", role=LineRole.PROVISION, provision_number="13.1"),
        L("Graduate", role=LineRole.CHAPTER_TITLE),
        L("Preamble text under the Graduate part before any section."),
    ]
    chunks = chunk(pages_of(*lines), SETTINGS, COUNTER, "doc")
    grad = [c for c in chunks if c.part == "Graduate"]
    assert grad and grad[0].section_number is None


def test_wrapped_chapter_title_joined():
    lines = [
        L("Lasallian Values", role=LineRole.CHAPTER_TITLE),
        L("and Lasallian Prayers", role=LineRole.CHAPTER_TITLE),
        L("Spirit of Faith body text."),
    ]
    chunks = chunk(pages_of(*lines), SETTINGS, COUNTER, "doc")
    assert chunks[0].part == "Lasallian Values and Lasallian Prayers"


def test_table_values_not_treated_as_provisions():
    # Regression: the grading-scale table ("4.0 Excellent ... 9.9 Deferred")
    # sits in the left column and matches the provision pattern, but inside
    # Section 10 only labels starting with "10." are real provisions. Table
    # values must be kept as body text, never as citation labels.
    lines = [
        L("Undergraduate", role=LineRole.CHAPTER_TITLE),
        L("SECTION 10: GRADING", role=LineRole.SECTION_HEADING,
          section_number="10", section_title="GRADING"),
        L("10.3 The grading scale is as follows.", role=LineRole.PROVISION,
          provision_number="10.3"),
        L("4.0 Excellent", role=LineRole.PROVISION, provision_number="4.0"),
        L("9.9 Deferred", role=LineRole.PROVISION, provision_number="9.9"),
    ]
    chunks = chunk(pages_of(*lines), SETTINGS, COUNTER, "doc")
    assert len(chunks) == 1
    assert chunks[0].provisions == ["10.3"]
    assert "4.0 Excellent" in chunks[0].text and "9.9 Deferred" in chunks[0].text


# --- Split behavior ------------------------------------------------------------

def test_oversized_segment_split_with_page_metadata_preserved():
    # One provision followed by 120 body lines of ~10 words spread across
    # pages 100-119 -> far over max_tokens -> must split, and each piece must
    # carry the true pages of its own lines (the Phase 4 bug this guards).
    lines = [
        L("Undergraduate", role=LineRole.CHAPTER_TITLE, page=100),
        L("SECTION 7: FEES", role=LineRole.SECTION_HEADING, page=100,
          section_number="7", section_title="FEES"),
        L("7.1 The long fee schedule begins here.", role=LineRole.PROVISION,
          page=100, provision_number="7.1"),
    ]
    for i in range(120):
        lines.append(L(f"fee item number {i} costs some amount of money units",
                       page=100 + i // 6))
    chunks = chunk([PageContent(page=100, lines=lines)], SETTINGS, COUNTER, "doc")

    assert len(chunks) > 1                                   # actually split
    assert all(c.token_count <= SETTINGS.max_tokens + 30 for c in chunks)
    pages_seen = {p for c in chunks for p in c.pages}
    assert len(pages_seen) > 5, "split pieces must keep their own true pages"
    # Overlap: the last line of piece N reappears at the start of piece N+1.
    first_body = chunks[0].text.split("\n", 1)[1]
    second_body = chunks[1].text.split("\n", 1)[1]
    tail = " ".join(first_body.split()[-8:])
    assert tail in second_body


# --- Citation & persistence ----------------------------------------------------

def test_citation_string_formats():
    c = Chunk("d_0001", "t", "d", "Undergraduate", "10", "GRADING",
              ["10.1", "10.3"], [101, 102], 300)
    assert c.citation() == ("Undergraduate, Section 10: GRADING, "
                            "prov. 10.1–10.3, pp. 101–102")
    c2 = Chunk("d_0002", "t", "d", "Appendices", None, None, [], [260], 100)
    assert c2.citation() == "Appendices, p. 260"


def test_write_chunks_jsonl(tmp_path):
    import json
    doc = make_doc("Undergraduate", "10", "GRADING", [("10.1", "Rule.")])
    chunks = chunk(doc, SETTINGS, COUNTER, "doc")
    out = tmp_path / "chunks.jsonl"
    n = write_chunks_jsonl(chunks, out)
    recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert n == len(recs) == len(chunks)
    assert recs[0]["citation"].startswith("Undergraduate, Section 10")


# --- Token counter -------------------------------------------------------------

def test_token_counter_heuristic_scales_with_words():
    counter = TokenCounter()          # no model name -> heuristic
    assert counter.count("") == 0
    ten = counter.count("word " * 10)
    twenty = counter.count("word " * 20)
    assert 10 <= ten <= 16 and 2 * ten - 2 <= twenty <= 2 * ten + 2


# --- Integration on the real handbook ------------------------------------------

@pytest.mark.skipif(not HANDBOOK.exists(), reason="handbook PDF not present")
def test_chunk_real_handbook_invariants():
    settings = load_settings()
    cleaned = clean(parse_pdf(str(HANDBOOK)))
    chunks = chunk(cleaned, settings.chunking, COUNTER, settings.document.id)

    # Count within the design's estimated band (docs/chunking_strategy.md §4).
    assert 300 <= len(chunks) <= 500

    # Size cap respected (small slack for the breadcrumb prefix).
    assert all(c.token_count <= settings.chunking.max_tokens + 30 for c in chunks)

    # All 21 sections represented.
    sections = {c.section_number for c in chunks if c.section_number}
    assert sections == {str(n) for n in range(1, 22)}

    # Complete page coverage: every cleaned content page appears in a chunk.
    content_pages = {p.page for p in cleaned}
    chunk_pages = {pg for c in chunks for pg in c.pages}
    assert content_pages == chunk_pages

    # Document-wide: every provision label in a sectioned chunk begins with
    # that chunk's section number (guards the table-value false positive).
    for c in chunks:
        if c.section_number:
            assert all(p.split(".", 1)[0] == c.section_number
                       for p in c.provisions), c.chunk_id

    # Known provision lands in the right place with the right context.
    plag = next(c for c in chunks if "5.3.1.1.6" in c.provisions)
    assert plag.part == "General Provisions"
    assert plag.section_number == "5"
    assert "Plagiarism" in plag.text

    # Section 14's wrapped title is complete in chunk metadata.
    s14 = next(c for c in chunks if c.section_number == "14")
    assert s14.section_title == "FEES, SCHOLARSHIPS, AND PAYMENTS"
