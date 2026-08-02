"""
chunker.py — Section-aware chunking of the cleaned handbook line stream.

Purpose:
    Convert cleaned lines into retrieval-ready chunks: coherent spans of
    ~target_tokens that never cross a section boundary, each prefixed with
    a breadcrumb (part › section) and tagged with full citation metadata
    (part, section number/title, provision labels, page range).

Inputs:
    list[PageContent] from src.ingestion.cleaner.clean, plus ChunkingSettings
    and a TokenCounter.

Outputs:
    list[Chunk]; write_chunks_jsonl() persists the inspectable
    data/processed/chunks.jsonl artifact.

Dependencies:
    src.ingestion.parser (line model), src.utils.tokens.

Why this file exists:
    The chunking strategy (docs/chunking_strategy.md) is the project's key
    retrieval-quality decision. Three passes implement it:
      1. SEGMENTATION — walk the lines tracking context (part, section) and
         cut a segment at every provision or heading; a segment is the
         smallest citable unit.
      2. MERGE — greedily pack consecutive segments of the same section into
         chunks up to target_tokens, so tiny provisions (median ~32 words)
         travel with their neighbors and parent context instead of becoming
         context-free fragments. Chunks never span sections.
      3. SPLIT — any single segment larger than max_tokens is split at
         sentence boundaries into ~target_tokens pieces with overlap_tokens
         of carried-over context, so a rule straddling a cut remains
         retrievable.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.parser import Line, LineRole, PageContent
from src.utils.config import ChunkingSettings
from src.utils.tokens import TokenCounter


# --- Data model ----------------------------------------------------------------

@dataclass
class Chunk:
    """A retrieval unit with its citation metadata."""

    chunk_id: str
    text: str                     # breadcrumb + content (what gets embedded)
    document: str
    part: str                     # e.g. "Undergraduate"
    section_number: str | None    # e.g. "10" (None outside numbered sections)
    section_title: str | None     # e.g. "CREDIT, GRADING AND RETENTION"
    provisions: list[str]         # provision labels contained, e.g. ["10.1","10.2"]
    pages: list[int]              # sorted page numbers the content came from
    token_count: int

    def citation(self) -> str:
        """Human-readable citation string for this chunk."""
        bits = [self.part]
        if self.section_number:
            title = f": {self.section_title}" if self.section_title else ""
            bits.append(f"Section {self.section_number}{title}")
        if self.provisions:
            first, last = self.provisions[0], self.provisions[-1]
            bits.append(f"prov. {first}" + (f"–{last}" if last != first else ""))
        if self.pages:
            bits.append(f"p. {self.pages[0]}" if len(self.pages) == 1
                        else f"pp. {self.pages[0]}–{self.pages[-1]}")
        return ", ".join(bits)


@dataclass
class _Segment:
    """Smallest citable unit produced by segmentation (internal)."""

    part: str
    section_number: str | None
    section_title: str | None
    provision: str | None         # label if the segment starts a provision
    lines: list[Line] = field(default_factory=list)

    def text(self) -> str:
        return " ".join(l.text for l in self.lines).strip()

    def pages(self) -> set[int]:
        return {l.page for l in self.lines}


# --- Public API ----------------------------------------------------------------

def chunk(
    pages: list[PageContent],
    settings: ChunkingSettings,
    counter: TokenCounter,
    document_id: str,
) -> list[Chunk]:
    """Run segmentation → merge → split over cleaned pages.

    Args:
        pages: Cleaned pages from cleaner.clean().
        settings: target/max/min/overlap token parameters.
        counter: Token counter (real tokenizer or estimate).
        document_id: Stable id stamped into chunk ids and metadata.

    Returns:
        Ordered list of Chunk objects covering all cleaned content.
    """
    segments = _segment(pages)
    chunks: list[Chunk] = []
    sequence = 0

    for group in _group_by_section(segments):
        packed = _merge_segments(group, settings, counter)
        for seg_list in packed:
            for piece_lines, piece_provisions in _split_if_needed(
                seg_list, settings, counter
            ):
                sequence += 1
                first = seg_list[0]
                text = _compose_text(first, piece_lines)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}_{sequence:04d}",
                        text=text,
                        document=document_id,
                        part=first.part,
                        section_number=first.section_number,
                        section_title=first.section_title,
                        provisions=piece_provisions,
                        pages=sorted({l.page for l in piece_lines}),
                        token_count=counter.count(text),
                    )
                )
    return chunks


# --- Pass 1: segmentation ------------------------------------------------------

def _segment(pages: list[PageContent]) -> list[_Segment]:
    """Walk cleaned lines tracking structural context; cut segments at every
    heading or provision line.

    Consecutive CHAPTER_TITLE lines merge into one part name (titles wrap
    across lines, e.g. "Lasallian Values" / "and Lasallian Prayers").
    Entering a new part or section resets the finer context below it.
    """
    segments: list[_Segment] = []
    part = "Front"                      # replaced by the first chapter title
    section_number: str | None = None
    section_title: str | None = None
    current: _Segment | None = None
    previous_role: LineRole | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and current.lines:
            segments.append(current)
        current = None

    for page in pages:
        for line in page.lines:
            if line.role is LineRole.CHAPTER_TITLE:
                flush()
                if previous_role is LineRole.CHAPTER_TITLE:
                    part = f"{part} {line.text}".strip()   # wrapped title
                else:
                    part = line.text.strip()
                section_number = None
                section_title = None
            elif line.role is LineRole.SECTION_HEADING:
                flush()
                section_number = line.section_number
                section_title = line.section_title
            elif line.role is LineRole.PROVISION:
                if not _provision_matches_section(line.provision_number,
                                                  section_number):
                    # Positional detection has one verified false-positive
                    # class: numeric table values in the left column (e.g.
                    # the grading scale "4.0 Excellent ... 9.9 Deferred" on
                    # p102). Inside Section N every true provision label
                    # starts with "N.", so a mismatched label is table data —
                    # keep its text as body content, not as a citation label.
                    if current is None:
                        current = _Segment(part, section_number, section_title,
                                           None, [])
                    current.lines.append(line)
                else:
                    flush()
                    current = _Segment(part, section_number, section_title,
                                       line.provision_number, [line])
            else:  # BODY
                if current is None:
                    # Preamble text under a section (or part) before any
                    # provision — a citable segment of its own.
                    current = _Segment(part, section_number, section_title,
                                       None, [])
                current.lines.append(line)
            previous_role = line.role
    flush()
    return segments


def _provision_matches_section(label: str | None,
                               section_number: str | None) -> bool:
    """True if a provision label is plausible in the current section context.

    Inside a numbered section, real provision labels share the section's
    number as their first component (10.2 under Section 10; 5.3.1.1.6 under
    Section 5 — verified across the whole handbook). Outside any numbered
    section (the appendices, which carry their own numbering), any label is
    accepted. This context check is what filters left-column numeric table
    values that are positionally indistinguishable from provision labels.
    """
    if section_number is None:
        return True
    if not label:
        return False
    return label.split(".", 1)[0] == section_number


def _group_by_section(segments: list[_Segment]):
    """Yield runs of consecutive segments sharing part + section.

    Chunks must never span a section boundary (a chunk mixing Section 9 and
    Section 10 text would produce a wrong citation), so merging only ever
    happens inside one of these groups.
    """
    group: list[_Segment] = []
    for seg in segments:
        if group and (seg.part != group[-1].part
                      or seg.section_number != group[-1].section_number):
            yield group
            group = []
        group.append(seg)
    if group:
        yield group


# --- Pass 2: merge -------------------------------------------------------------

def _merge_segments(
    group: list[_Segment], settings: ChunkingSettings, counter: TokenCounter
) -> list[list[_Segment]]:
    """Greedily pack consecutive segments into chunk-sized lists.

    A segment is added to the open pack while the running total stays within
    target_tokens; oversized single segments pass through alone (the split
    pass handles them). Greedy packing subsumes the design's "merge small
    provisions" rule: tiny segments (median ~32 words) accumulate with their
    neighbors until the pack reaches a useful size.
    """
    packs: list[list[_Segment]] = []
    open_pack: list[_Segment] = []
    open_tokens = 0

    for seg in group:
        seg_tokens = counter.count(seg.text())
        if open_pack and open_tokens + seg_tokens > settings.target_tokens:
            packs.append(open_pack)
            open_pack, open_tokens = [], 0
        open_pack.append(seg)
        open_tokens += seg_tokens
    if open_pack:
        packs.append(open_pack)
    return packs


# --- Pass 3: split -------------------------------------------------------------

def _split_if_needed(
    pack: list[_Segment], settings: ChunkingSettings, counter: TokenCounter
):
    """Yield (lines, provisions) pieces, splitting oversized packs.

    Packs within max_tokens pass through whole. An oversized pack (a long
    unnumbered passage — e.g. the appendix room-directory tables span many
    pages with no provision labels) is split at LINE boundaries into
    ~target_tokens pieces, with the last few lines of each piece repeated at
    the start of the next as overlap so content straddling a cut stays
    retrievable.

    Splitting at line granularity (rather than re-joining to text and
    splitting on sentences) matters for two verified reasons: (1) every
    line carries its true page number, so per-chunk page metadata and
    citations stay accurate across multi-page passages; and (2) table-like
    appendix content contains almost no sentence punctuation, which starved
    a sentence-based splitter of cut points and produced over-limit chunks.
    """
    all_lines = [l for seg in pack for l in seg.lines]
    provisions = [seg.provision for seg in pack if seg.provision]
    total_text = " ".join(l.text for l in all_lines).strip()

    if counter.count(total_text) <= settings.max_tokens:
        yield all_lines, provisions
        return

    piece: list[Line] = []
    piece_tokens = 0
    for line in all_lines:
        line_tokens = counter.count(line.text)
        if piece and piece_tokens + line_tokens > settings.target_tokens:
            yield piece, provisions
            overlap: list[Line] = []
            overlap_tokens = 0
            for prev in reversed(piece):
                prev_tokens = counter.count(prev.text)
                if overlap_tokens + prev_tokens > settings.overlap_tokens:
                    break
                overlap.insert(0, prev)
                overlap_tokens += prev_tokens
            piece = list(overlap)
            piece_tokens = overlap_tokens
        piece.append(line)
        piece_tokens += line_tokens
    if piece:
        yield piece, provisions


# --- Text composition & persistence --------------------------------------------

def _compose_text(context: _Segment, lines: list[Line]) -> str:
    """Prefix content with its breadcrumb so embeddings encode topical context.

    Example: "Undergraduate › Section 10: CREDIT, GRADING AND RETENTION\\n..."
    Costs ~10 tokens; measurably helps dense retrieval of short provisions
    (docs/chunking_strategy.md §2).
    """
    crumb = context.part
    if context.section_number:
        title = f": {context.section_title}" if context.section_title else ""
        crumb += f" › Section {context.section_number}{title}"
    body = " ".join(l.text for l in lines).strip()
    return f"{crumb}\n{body}"


def write_chunks_jsonl(chunks: list[Chunk], path: Path | str) -> int:
    """Persist chunks to JSONL for inspection; returns record count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for c in chunks:
            handle.write(json.dumps({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "document": c.document,
                "part": c.part,
                "section_number": c.section_number,
                "section_title": c.section_title,
                "provisions": c.provisions,
                "pages": c.pages,
                "token_count": c.token_count,
                "citation": c.citation(),
            }, ensure_ascii=False) + "\n")
    return len(chunks)
