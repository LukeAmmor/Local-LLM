"""Clean the blocks produced by app.parsing before chunking.

Steps, in order (`clean_blocks`):
  1. Remove repeated headers and footers (PDF only): a block near the top or
     bottom of the page whose text (digits ignored) appears on at least 3
     pages, e.g. "Procurement Regulations for IPF Borrowers" or "Page 3 of 11".
     (Not "half of the pages": long documents change their running header
     with each chapter.) Lone page numbers near the top or bottom are removed too.
  2. Normalise Unicode (NFKC): typographic ligatures such as "ﬁ" become "fi",
     non-breaking spaces become normal spaces. Invisible characters are
     removed: soft hyphens, zero-width characters, bidirectional control marks.
  3. Join words split by a hyphen at the end of a line ("infor-\\nmation" ->
     "information") when the next line starts with a lowercase letter. When
     it starts with a capital or a digit ("Jean-\\nPierre", "COVID-\\n19"),
     the hyphen is real: keep it and only remove the line break.
  4. Join the lines of a paragraph into one line (PDF lines are wrapped),
     except before a list item (•, -, 1., a) ...).
  5. Collapse dot leaders ("Value Engineering . . . . . . . 17" in a table of
     contents) into " … ", so they do not fill chunks with dots.
  6. Drop blocks left without any letter or digit, and blocks that are only a
     chapter number in roman numerals ("IV" printed in large type on a
     chapter page): alone, they mean nothing to a reader of the chunk.

Known limit of step 3: a French compound word split exactly at its own hyphen
("peut-\\nêtre") becomes "peutêtre". It is rare; a mid-line "peut-être" is
never touched.
"""

import re
import unicodedata

from app.parsing import Block

HEADER_FOOTER_ZONE = 0.08  # top and bottom 8% of the page height
MIN_REPEAT_PAGES = 3  # a margin text on at least this many pages is a header/footer

# Invisible characters: soft hyphen, zero-width characters, bidi marks, byte order mark.
INVISIBLE = re.compile("[\u00AD\u200B-\u200F\u202A-\u202E\u2060\u2066-\u2069\uFEFF]")
# Page numbers after repeat_key() (digits already replaced by "#"): "#", "page # of #", "iv"...
PAGE_NUMBER = re.compile(r"^(page|p\.)?\s*(#+|[ivxlc]{1,6})(\s*(/|of|sur|de)\s*#+)?$", re.IGNORECASE)
DOT_LEADER = re.compile(r"\s*(?:\.\s*){4,}")  # 4+ dots, with the spaces around them
LONE_ROMAN_NUMERAL = re.compile(r"^[IVX]{1,5}$")
HYPHEN_AT_LINE_END = re.compile(r"(\w)-\n\s*(?=[a-zà-öø-ÿœæç])")
REAL_HYPHEN_AT_LINE_END = re.compile(r"(\w)-\n\s*(?=[A-ZÀ-ÖØ-Þ0-9])")
# A newline is kept before list items: bullets, dashes, "1.", "1)", "a)", "(a)", "iv)".
NEWLINE_NOT_BEFORE_LIST = re.compile(r"\n(?!\s*(?:[•●○◦▪■□➢✓\-–—*]|\(?[0-9]{1,2}[.)]|\(?[a-z][.)]|\(?[ivx]{1,4}\))\s)")
SPACES = re.compile(r"[ \t]+")


def repeat_key(text: str) -> str:
    """Text used to recognise a repeated header/footer: lowercase, digits as '#'."""
    return SPACES.sub(" ", re.sub(r"\d+", "#", text.lower())).strip()


def in_margin(block: Block) -> bool:
    if block.y0 is None or block.page_height is None:
        return False
    return (block.y1 <= block.page_height * HEADER_FOOTER_ZONE
            or block.y0 >= block.page_height * (1 - HEADER_FOOTER_ZONE))


def remove_headers_footers(blocks: list[Block]) -> list[Block]:
    pages = {b.page for b in blocks if b.page is not None}
    if not pages:
        return blocks
    # On how many different pages does each margin text appear?
    pages_per_key: dict[str, set[int]] = {}
    for b in blocks:
        if in_margin(b):
            pages_per_key.setdefault(repeat_key(b.text), set()).add(b.page)
    repeated = {k for k, p in pages_per_key.items() if len(p) >= MIN_REPEAT_PAGES}

    kept = []
    for b in blocks:
        key = repeat_key(b.text)
        if in_margin(b) and (key in repeated or PAGE_NUMBER.match(key)):
            continue
        kept.append(b)
    return kept


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return INVISIBLE.sub("", text)


def fix_hyphenation(text: str) -> str:
    text = HYPHEN_AT_LINE_END.sub(r"\1", text)
    return REAL_HYPHEN_AT_LINE_END.sub(r"\1-", text)


def join_lines(text: str) -> str:
    text = NEWLINE_NOT_BEFORE_LIST.sub(" ", text)
    lines = [SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def collapse_dot_leaders(text: str) -> str:
    return DOT_LEADER.sub(" … ", text)


def clean_blocks(blocks: list[Block]) -> list[Block]:
    cleaned = []
    for b in remove_headers_footers(blocks):
        text = collapse_dot_leaders(join_lines(fix_hyphenation(normalize_text(b.text))))
        if any(ch.isalnum() for ch in text) and not LONE_ROMAN_NUMERAL.match(text):
            b.text = text
            cleaned.append(b)
    return cleaned
