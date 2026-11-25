"""Turn a document file (PDF, DOCX, Markdown, TXT) into a list of text blocks.

What: `parse_document(path)` returns the document's blocks in reading order.
A block is a paragraph or a heading, with the page it comes from.

  PDF   PyMuPDF gives text blocks with their font size, boldness and position.
        Blocks are kept in the order the PDF stores them, which is the
        reading order for most PDFs, two-column layouts included (sorting
        by position would mix the two columns line by line).
        A short block is a heading when its font is clearly larger than the
        body text, or when it is bold, and it has at least 3 letters besides
        chapter numbers ("III" alone is decoration). Heading levels come from font sizes:
        the largest heading size is level 1, the next one level 2, then 3
        (sizes less than 1 point apart count as the same level).
        Page numbers are the physical page numbers (1 = first page of the
        file), the same numbers used to render a page as an image.
  DOCX  python-docx gives paragraphs and tables in order. Headings come from
        Word's outline levels (what builds Word's table of contents), so
        custom heading styles count too. A DOCX file has no fixed pages: Word
        computes them each time it displays or prints the file. Word does save
        page-break markers, but on the French procurement regulations they
        are incomplete (145 markers for 151 pages), so the page numbers
        drifted by up to 8 pages. DOCX blocks therefore have no page
        (page = None) and DOCX citations show the section instead.
  MD    "# Title" lines are headings, blank lines separate paragraphs. No pages.
  TXT   blank lines separate paragraphs. No pages.

Why: chunking needs structure (headings, paragraphs) and citations need page
numbers, so we keep both from the start. Cleaning (headers, footers,
hyphenation) happens next, in app.cleaning.
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import docx
import pymupdf
from docx.table import Table
from docx.text.paragraph import Paragraph

MAX_HEADING_CHARS = 150  # longer blocks are never headings
MAX_HEADING_LINES = 3
MAX_HEADING_LEVEL = 3
SAME_LEVEL_POINTS = 1.0  # heading font sizes closer than this share a level
MERGE_GAP_POINTS = 15.0  # a heading this close below another one continues it
NUMBER_TOKEN = re.compile(r"\b(?:[IVX]{1,5}|\d+)\b")  # roman numerals (up to XVIII) and numbers


@dataclass
class Block:
    text: str
    page: int | None  # 1-based page number; None when the format has no pages
    is_heading: bool = False
    level: int = 0  # heading level 1..3 (0 for normal paragraphs)
    # PDF only: position on the page, used to find repeated headers and footers.
    y0: float | None = None
    y1: float | None = None
    page_height: float | None = None


@dataclass
class ParsedDocument:
    blocks: list[Block]
    page_count: int | None  # None for formats without pages


def parse_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix in (".md", ".txt"):
        return parse_plain_text(path, markdown=suffix == ".md")
    raise ValueError(f"unsupported file type: {path.suffix}")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def parse_pdf(path: Path) -> ParsedDocument:
    # Some PDFs trigger harmless MuPDF messages (e.g. about optional layers).
    pymupdf.TOOLS.mupdf_display_errors(False)

    # Pass 1: read every text block with its main font size and boldness.
    raw = []  # (block, font_size, is_bold, n_lines)
    size_chars: Counter = Counter()  # characters per font size, to find the body size
    with pymupdf.open(path) as pdf:
        page_count = len(pdf)
        for page_number, page in enumerate(pdf, start=1):
            height = page.rect.height
            # "dict": blocks -> lines -> spans (a span is a run of text with one font and size).
            for b in page.get_text("dict")["blocks"]:
                if b["type"] != 0:  # 1 = image block
                    continue
                lines, block_sizes, bold_chars, n_chars = [], Counter(), 0, 0
                for line in b["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                        n = len(span["text"].strip())
                        block_sizes[round(span["size"], 1)] += n
                        n_chars += n
                        if span["flags"] & 16:  # bit 4 = bold
                            bold_chars += n
                    if line_text.strip():
                        lines.append(line_text.strip())
                if not lines:
                    continue
                size = block_sizes.most_common(1)[0][0]
                size_chars.update(block_sizes)
                block = Block(text="\n".join(lines), page=page_number,
                              y0=b["bbox"][1], y1=b["bbox"][3], page_height=height)
                raw.append((block, size, bold_chars > n_chars / 2, len(lines)))

    if not raw:
        return ParsedDocument(blocks=[], page_count=page_count)
    body_size = size_chars.most_common(1)[0][0]

    # Pass 2: decide which blocks are headings, then give them levels by font size.
    headings = []
    for block, size, bold, n_lines in raw:
        # Letters of real words: a chapter number alone ("III", "IV 9") is decoration, not a heading.
        letters = sum(ch.isalpha() for ch in NUMBER_TOKEN.sub("", block.text))
        short = len(block.text) <= MAX_HEADING_CHARS and n_lines <= MAX_HEADING_LINES
        if short and letters >= 3 and (size >= body_size + 1.0 or (bold and size >= body_size - 0.1)):
            block.is_heading = True
            headings.append((block, size))
    levels = heading_levels([size for _, size in headings])
    for block, size in headings:
        block.level = levels[size]

    blocks = merge_consecutive_headings([block for block, *_ in raw])
    return ParsedDocument(blocks=blocks, page_count=page_count)


def heading_levels(sizes: list[float]) -> dict[float, int]:
    """Map each heading font size to a level: biggest = 1. Sizes < 1 pt apart share a level."""
    levels: dict[float, int] = {}
    level, level_size = 0, None
    for size in sorted(set(sizes), reverse=True):
        if level_size is None or level_size - size >= SAME_LEVEL_POINTS:
            level, level_size = level + 1, size
        levels[size] = min(level, MAX_HEADING_LEVEL)
    return levels


def merge_consecutive_headings(blocks: list[Block]) -> list[Block]:
    """Join a heading split over two blocks ("III) LA BANQUE MONDIALE ET LES" + "SANCTIONS").

    Only when the second block is just below the first one, at the same level.
    """
    merged: list[Block] = []
    for block in blocks:
        prev = merged[-1] if merged else None
        if (prev and prev.is_heading and block.is_heading
                and prev.level == block.level and prev.page == block.page
                and prev.y1 is not None and 0 <= block.y0 - prev.y1 < MERGE_GAP_POINTS):
            prev.text = f"{prev.text} {block.text}"
            prev.y1 = block.y1
        else:
            merged.append(block)
    return merged


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------
HEADING_STYLE = re.compile(r"^(heading|titre)\s*(\d)", re.IGNORECASE)


def docx_outline_level(paragraph: Paragraph) -> int | None:
    """Word's outline level of a paragraph: 0 = level 1, 1 = level 2... (None = body text).

    The outline level is what Word uses to build its navigation pane and its
    table of contents. It is set on the paragraph itself or on its style, and
    a style inherits it from its base style: "REGs H1" is based on "Heading 1"
    (outline level 0). python-docx has no property for it, so we read the
    XML element <w:outlineLvl w:val="..."> directly.
    """
    values = paragraph._p.xpath("./w:pPr/w:outlineLvl/@w:val")
    style = paragraph.style
    while not values and style is not None:
        values = style.element.xpath("./w:pPr/w:outlineLvl/@w:val")
        style = style.base_style
    if values and int(values[0]) < 9:  # 9 means "body text"
        return int(values[0])
    return None


def docx_heading_level(paragraph: Paragraph) -> int:
    """Heading level of a Word paragraph (1 = top level), or 0 for body text.

    First the outline level (covers custom heading styles), then the style
    name ("Heading 2", or "Titre 2" in French Word), for files written by
    tools that do not set outline levels. Long paragraphs are never headings.
    """
    if len(paragraph.text.strip()) > MAX_HEADING_CHARS:
        return 0
    outline = docx_outline_level(paragraph)
    if outline is not None:
        return min(outline + 1, MAX_HEADING_LEVEL)
    style = paragraph.style.name if paragraph.style is not None else ""
    m = HEADING_STYLE.match(style or "")
    if m:
        return min(int(m.group(2)), MAX_HEADING_LEVEL)
    if (style or "").lower() == "title":
        return 1
    return 0


def parse_docx(path: Path) -> ParsedDocument:
    """Paragraphs and table rows in order, without pages (see the module docstring)."""
    document = docx.Document(str(path))
    blocks: list[Block] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if text:
                level = docx_heading_level(item)
                blocks.append(Block(text=text, page=None, is_heading=level > 0, level=level))
        elif isinstance(item, Table):
            for row in item.rows:
                cells = []
                for cell in row.cells:
                    text = cell.text.strip()
                    if text and text not in cells:  # merged cells repeat their text
                        cells.append(text)
                if cells:
                    blocks.append(Block(text=" | ".join(cells), page=None))
    return ParsedDocument(blocks=blocks, page_count=None)


# ---------------------------------------------------------------------------
# Markdown and plain text
# ---------------------------------------------------------------------------
MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_plain_text(path: Path, markdown: bool) -> ParsedDocument:
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(Block(text="\n".join(paragraph), page=None))
            paragraph.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        m = MD_HEADING.match(line) if markdown else None
        if m:
            flush()
            level = min(len(m.group(1)), MAX_HEADING_LEVEL)
            blocks.append(Block(text=m.group(2).strip(), page=None, is_heading=True, level=level))
        elif line.strip():
            paragraph.append(line.strip())
        else:
            flush()
    flush()
    return ParsedDocument(blocks=blocks, page_count=None)
