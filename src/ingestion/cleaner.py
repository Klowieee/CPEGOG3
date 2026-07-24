"""
cleaner.py — Remove non-content from parsed pages, yielding chunk-ready lines.

Purpose:
    Strip everything that should not become searchable content — front
    matter, running page numbers, decorative chapter-number banners, and
    tiny fine-print fragments — while preserving all policy text and the
    structural markers (section headings, provisions, part titles) the
    chunker relies on.

Inputs:
    list[PageContent] from src.ingestion.parser.parse_pdf.

Outputs:
    list[PageContent] containing only kept content, in reading order.
    Also writes an inspectable data/processed/cleaned.jsonl artifact via
    write_cleaned_jsonl().

Dependencies:
    src.ingestion.parser (data model); json/pathlib (standard library).

Why this file exists:
    Design analysis (docs/rag_pipeline.md §2) requires removing ceremonial
    front matter and page furniture that would otherwise pollute retrieval.
    Measurement of the actual handbook (Phase 3) showed the core policy
    sections (pages 18-139) contain almost no fine print — so the only
    judgement call is the small-type directory/organization content in the
    back, which we KEEP as body text (it is real, queryable handbook
    content) except for sub-4-word fragments.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.parser import Line, LineRole, PageContent

# --- Cleaning policy constants -------------------------------------------------

# Content begins at the "Lasallian Values and Lasallian Prayers" divider: this
# keeps that (queryable) section plus all policy, while dropping the cover,
# acknowledgement form, table of contents, President's message, founder
# biography, and institutional introductions that precede it.
CONTENT_START_TITLE = "lasallian value"      # matched case-insensitively
CONTENT_START_FALLBACK = "general provisions"  # used if the primary is absent

# OTHER (small-type) lines with at least this many words are kept as body
# content; shorter ones are treated as fragments/footnote markers and dropped.
MIN_OTHER_CONTENT_WORDS = 4


def clean(pages: list[PageContent]) -> list[PageContent]:
    """Return a cleaned copy of the parsed pages.

    Removes front matter, page numbers, decorative numeric chapter banners,
    and short fine-print fragments. Reclassifies substantial small-type
    lines (OTHER with >= MIN_OTHER_CONTENT_WORDS words) as BODY so the
    chunker treats them as ordinary content.

    Args:
        pages: Parsed pages from parse_pdf().

    Returns:
        Cleaned pages (front-matter pages omitted entirely). Each retained
        Line keeps its original page number for citation purposes.
    """
    start_page = _find_content_start_page(pages)

    cleaned: list[PageContent] = []
    for page in pages:
        if page.page < start_page:
            continue
        kept: list[Line] = []
        for line in page.lines:
            new_line = _filter_line(line)
            if new_line is not None:
                kept.append(new_line)
        if kept:
            cleaned.append(PageContent(page=page.page, lines=kept))
    return cleaned


def _find_content_start_page(pages: list[PageContent]) -> int:
    """Find the first page of real content (the Lasallian Values divider).

    Returns:
        The 1-indexed page where kept content begins. Falls back to the
        "General Provisions" divider, then to 1 (keep everything) if neither
        marker is found — failing open rather than silently dropping content.
    """
    fallback_page: int | None = None
    for page in pages:
        for line in page.lines:
            if line.role is LineRole.CHAPTER_TITLE:
                text = line.text.lower()
                if CONTENT_START_TITLE in text:
                    return page.page
                if CONTENT_START_FALLBACK in text and fallback_page is None:
                    fallback_page = page.page
    if fallback_page is not None:
        return fallback_page
    return 1


def _filter_line(line: Line) -> Line | None:
    """Decide whether to keep a line, returning it (possibly relabeled) or None.

    Rules:
      * Drop running page numbers.
      * Drop decorative chapter banners whose text is purely a number.
      * Keep substantial OTHER lines by relabeling them BODY; drop short ones.
      * Keep SECTION_HEADING, PROVISION, BODY, and named CHAPTER_TITLE lines.
    """
    if line.role is LineRole.PAGE_NUMBER:
        return None

    if line.role is LineRole.CHAPTER_TITLE and line.text.strip().isdigit():
        # Decorative giant chapter number (e.g. "1"); the adjacent title line
        # carries the actual part name, so this one is noise.
        return None

    if line.role is LineRole.OTHER:
        if len(line.text.split()) >= MIN_OTHER_CONTENT_WORDS:
            # Substantial small-type content (student-org / office listings) —
            # keep it as body text so it can be chunked and retrieved.
            return _relabel(line, LineRole.BODY)
        return None

    return line


def _relabel(line: Line, role: LineRole) -> Line:
    """Return a copy of a line with a different role (leaves the original intact)."""
    return Line(
        text=line.text,
        page=line.page,
        font_size=line.font_size,
        font_name=line.font_name,
        x0=line.x0,
        role=role,
        section_number=line.section_number,
        section_title=line.section_title,
        provision_number=line.provision_number,
    )


def write_cleaned_jsonl(pages: list[PageContent], path: Path | str) -> int:
    """Persist cleaned lines to a JSONL file for inspection and debugging.

    Args:
        pages: Cleaned pages from clean().
        path: Output file path (parent directories are created).

    Returns:
        The number of line records written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for page in pages:
            for line in page.lines:
                record = {
                    "page": line.page,
                    "role": line.role.value,
                    "text": line.text,
                    "section_number": line.section_number,
                    "section_title": line.section_title,
                    "provision_number": line.provision_number,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count
