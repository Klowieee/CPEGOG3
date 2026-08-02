"""
parser.py — Parse the handbook PDF into structured, classified lines.

Purpose:
    Turn the raw PDF into a faithful, inspectable representation: for every
    page, a list of text lines, each tagged with its font size, font name,
    left-edge position, and a detected structural ROLE (chapter title,
    section heading, numbered provision, body text, or page number).
    Downstream phases (cleaning, chunking) consume this structure instead
    of re-deriving font/geometry information.

Inputs:
    Path to the handbook PDF.

Outputs:
    list[PageContent] — one entry per PDF page, in page order.

Dependencies:
    pdfplumber (external); dataclasses/enum/re (standard library).

Why this file exists:
    Design analysis of the actual handbook (docs/chunking_strategy.md §1)
    showed that structure is encoded two ways: major headings by FONT SIZE
    (SECTION headings at 14pt bold; chapter titles at 26pt+), and numbered
    provisions by POSITION (their number sits in a left column at x0≈111-160
    while body text starts at x0≈202) rather than by font. Plain text
    extraction loses both signals, so we parse with pdfplumber, which
    exposes per-word font size and coordinates, and classify each line here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import pdfplumber

# --- Classification thresholds -------------------------------------------------
# All derived from measurements of the AY 2021-2025 handbook (see module
# docstring). Kept as named constants so they are easy to inspect and adjust
# if a future handbook edition uses a different template.

SECTION_HEADING_MIN_SIZE = 13.5   # "SECTION N:" headings are 14pt; body is 10pt
CHAPTER_TITLE_MIN_SIZE = 24.0     # Chapter/part titles are 26-28pt
BODY_COLUMN_X0 = 200.0            # Body text begins at x0≈202; provision
                                  # numbers sit clearly left of this.
LEFT_COLUMN_MAX_X0 = 195.0        # A provision number's left edge is below this.
MIN_BODY_FONT_SIZE = 9.0          # Below this is fine print / footnotes.

# A provision label: two or more dot-separated numbers, e.g. "5.3", "5.3.1.1.6",
# "21.5.14.2". Requiring at least one dot avoids matching a stray leading digit.
PROVISION_RE = re.compile(r"^\d{1,2}(?:\.\d{1,3}){1,6}$")

# A "SECTION N: Title" heading (matched only on large-font lines).
SECTION_RE = re.compile(r"^SECTION\s+(\d+)\s*:?\s*(.*)$", re.IGNORECASE)


class LineRole(str, Enum):
    """The structural role a reconstructed text line plays in the handbook."""

    CHAPTER_TITLE = "chapter_title"     # e.g. "Student Activities" (part divider)
    SECTION_HEADING = "section_heading"  # e.g. "SECTION 5: LASALLIAN COMMUNITY STANDARDS"
    PROVISION = "provision"             # a numbered rule, e.g. "5.3.1.2 Vandalism ..."
    BODY = "body"                       # ordinary paragraph text
    PAGE_NUMBER = "page_number"         # standalone running page number
    OTHER = "other"                     # fine print, captions, footnotes


@dataclass
class Line:
    """One reconstructed line of text with its typographic and layout metadata."""

    text: str
    page: int                 # 1-indexed PDF page number
    font_size: float          # dominant (most common) font size on the line
    font_name: str            # dominant font name on the line
    x0: float                 # left edge of the leftmost word on the line
    role: LineRole = LineRole.BODY
    # Populated only when role is SECTION_HEADING or PROVISION:
    section_number: str | None = None
    section_title: str | None = None
    provision_number: str | None = None


@dataclass
class PageContent:
    """All reconstructed lines for a single PDF page, in reading order."""

    page: int                 # 1-indexed
    lines: list[Line] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.lines) == 0


def parse_pdf(pdf_path: str) -> list[PageContent]:
    """Parse a PDF into structured, role-classified pages.

    Args:
        pdf_path: Filesystem path to the handbook PDF.

    Returns:
        A list of PageContent objects in page order. Pages with no
        extractable text (e.g. image-only cover pages) yield an empty
        PageContent, preserving page numbering.
    """
    pages: list[PageContent] = []
    with pdfplumber.open(pdf_path) as pdf:
        for index, page in enumerate(pdf.pages):
            page_number = index + 1
            words = page.extract_words(extra_attrs=["fontname", "size"])
            lines = _reconstruct_lines(words, page_number)
            for line in lines:
                _classify(line)
            lines = _merge_heading_continuations(lines)
            pages.append(PageContent(page=page_number, lines=lines))
    return pages


def _merge_heading_continuations(lines: list[Line]) -> list[Line]:
    """Fold wrapped section-heading lines into the heading they continue.

    A long section title can wrap onto a second line in the PDF (verified:
    "SECTION 14: FEES, SCHOLARSHIPS, AND" / "PAYMENTS" on page 113). The
    continuation line is large-font but does not match the SECTION pattern,
    so per-line classification labels it OTHER — and it would later be
    discarded as a fragment, silently truncating the section title used in
    breadcrumbs and citations. This pass appends any large-font OTHER line
    that immediately follows a SECTION_HEADING to that heading's text and
    title, and removes the continuation line.
    """
    merged: list[Line] = []
    for line in lines:
        if (
            merged
            and merged[-1].role is LineRole.SECTION_HEADING
            and line.role is LineRole.OTHER
            and line.font_size >= SECTION_HEADING_MIN_SIZE
        ):
            heading = merged[-1]
            heading.text = f"{heading.text} {line.text}".strip()
            continuation = line.text.strip()
            heading.section_title = (
                f"{heading.section_title} {continuation}".strip()
                if heading.section_title
                else continuation
            )
            continue
        merged.append(line)
    return merged


def _reconstruct_lines(words: list[dict], page_number: int) -> list[Line]:
    """Group pdfplumber words into lines by vertical position.

    pdfplumber returns individual words; a "line" is the set of words sharing
    the same baseline. We bucket words by their rounded 'top' coordinate (with
    a small tolerance) and order each bucket left-to-right. This correctly
    reunites a provision number in the left column with the body text to its
    right when they share a baseline.

    Args:
        words: Word dicts from page.extract_words(extra_attrs=[...]).
        page_number: 1-indexed page number to stamp on each line.

    Returns:
        Lines for the page, ordered top-to-bottom.
    """
    if not words:
        return []

    # Bucket words whose tops fall within TOLERANCE points of each other.
    TOLERANCE = 3.0
    buckets: list[tuple[float, list[dict]]] = []
    for word in sorted(words, key=lambda w: w["top"]):
        placed = False
        for i, (top, bucket) in enumerate(buckets):
            if abs(word["top"] - top) <= TOLERANCE:
                bucket.append(word)
                placed = True
                break
        if not placed:
            buckets.append((word["top"], [word]))

    lines: list[Line] = []
    for _, bucket in buckets:
        ordered = sorted(bucket, key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in ordered).strip()
        if not text:
            continue
        dominant_size, dominant_font = _dominant_font(ordered)
        lines.append(
            Line(
                text=text,
                page=page_number,
                font_size=dominant_size,
                font_name=dominant_font,
                x0=ordered[0]["x0"],
            )
        )
    return lines


def _dominant_font(words: list[dict]) -> tuple[float, str]:
    """Return the most common (size, fontname) among a line's words.

    Using the most common rather than the first word's font avoids being
    misled by a single stray glyph (e.g. a bullet or footnote marker).
    """
    tally: dict[tuple[float, str], int] = {}
    for w in words:
        key = (round(float(w["size"]), 1), str(w["fontname"]).split("+")[-1])
        tally[key] = tally.get(key, 0) + 1
    (size, font), _ = max(tally.items(), key=lambda kv: kv[1])
    return size, font


def _classify(line: Line) -> None:
    """Assign a LineRole (and any extracted numbers) to a line in place.

    Rules, in priority order (see module docstring for the evidence behind
    each threshold):
      1. Chapter title  — font size >= CHAPTER_TITLE_MIN_SIZE.
      2. Section heading — large font AND matches "SECTION N: ...".
      3. Page number    — the whole line is 1-4 digits (running page number).
      4. Provision      — the first token matches the provision pattern AND
                          sits in the left column (x0 < LEFT_COLUMN_MAX_X0).
      5. Other          — fine print below MIN_BODY_FONT_SIZE.
      6. Body           — everything else.
    """
    # 1. Chapter / part title.
    if line.font_size >= CHAPTER_TITLE_MIN_SIZE:
        line.role = LineRole.CHAPTER_TITLE
        return

    # 2. Section heading (only trust the SECTION pattern on large-font lines).
    if line.font_size >= SECTION_HEADING_MIN_SIZE:
        match = SECTION_RE.match(line.text)
        if match:
            line.role = LineRole.SECTION_HEADING
            line.section_number = match.group(1)
            line.section_title = match.group(2).strip() or None
            return
        # Large font but not a SECTION line (e.g. decorative part banner).
        line.role = LineRole.OTHER
        return

    # 3. Standalone running page number.
    if re.fullmatch(r"\d{1,4}", line.text):
        line.role = LineRole.PAGE_NUMBER
        return

    # 4. Numbered provision, identified by pattern + left-column position.
    first_token = line.text.split(" ", 1)[0]
    if PROVISION_RE.match(first_token) and line.x0 < LEFT_COLUMN_MAX_X0:
        line.role = LineRole.PROVISION
        line.provision_number = first_token
        return

    # 5. Fine print / footnote.
    if line.font_size < MIN_BODY_FONT_SIZE:
        line.role = LineRole.OTHER
        return

    # 6. Ordinary body text.
    line.role = LineRole.BODY
