"""Tests for app.cleaning: headers/footers, Unicode, hyphenation, line joining."""

from app.cleaning import (
    clean_blocks,
    collapse_dot_leaders,
    fix_hyphenation,
    join_lines,
    normalize_text,
    remove_headers_footers,
)
from app.parsing import Block

PAGE_HEIGHT = 800.0


def at(text: str, page: int, y0: float, y1: float) -> Block:
    return Block(text=text, page=page, y0=y0, y1=y1, page_height=PAGE_HEIGHT)


def test_repeated_header_and_page_numbers_are_removed():
    blocks = []
    for page in range(1, 5):
        blocks.append(at("Procurement Regulations for IPF Borrowers", page, 10, 30))  # top margin
        blocks.append(at(f"Body text of page {page}.", page, 300, 320))
        blocks.append(at(f"Page {page} of 4", page, 770, 790))  # bottom margin
    kept = remove_headers_footers(blocks)
    assert [b.text for b in kept] == [f"Body text of page {p}." for p in range(1, 5)]


def test_margin_text_on_only_two_pages_is_kept():
    blocks = [at("Annex note", 1, 10, 30), at("Annex note", 2, 10, 30), at("Body", 3, 300, 320)]
    assert len(remove_headers_footers(blocks)) == 3


def test_repeated_text_in_the_middle_of_the_page_is_kept():
    blocks = [at("Same sentence", p, 300, 320) for p in range(1, 6)]
    assert len(remove_headers_footers(blocks)) == 5


def test_lone_page_number_removed_even_once():
    blocks = [at("12", 1, 770, 790), at("iv", 2, 770, 790), at("Body", 1, 300, 320)]
    assert [b.text for b in remove_headers_footers(blocks)] == ["Body"]


def test_normalize_ligatures_invisible_chars_and_arabic_forms():
    assert normalize_text("ﬁnance") == "finance"
    assert normalize_text("procure­ment") == "procurement"  # soft hyphen
    assert normalize_text("ﻻ") == "لا"  # Arabic presentation form -> normal letters
    assert normalize_text("الشـــروط") == "الشروط"  # tatweel removed
    assert normalize_text("‏نص‎") == "نص"  # bidi marks removed


def test_hyphenation():
    assert fix_hyphenation("infor-\nmation") == "information"
    assert fix_hyphenation("Jean-\nPierre") == "Jean-Pierre"
    assert fix_hyphenation("COVID-\n19") == "COVID-19"
    assert fix_hyphenation("peut-être") == "peut-être"  # mid-line hyphen untouched


def test_join_lines_keeps_list_items():
    text = "The Borrower shall\nprepare a plan:\n• first item\n• second item\n(a) third item"
    assert join_lines(text) == "The Borrower shall prepare a plan:\n• first item\n• second item\n(a) third item"


def test_dot_leaders_collapsed():
    assert collapse_dot_leaders("Value Engineering . . . . . . . . 17") == "Value Engineering … 17"
    assert collapse_dot_leaders("Section 1.2.3 applies.") == "Section 1.2.3 applies."


def test_clean_blocks_drops_empty_blocks_and_lone_chapter_numbers():
    blocks = [Block(text="  •  ", page=1), Block(text="IV", page=1), Block(text="Real text", page=1),
              Block(text="Section IV", page=1), Block(text="10", page=1)]
    assert [b.text for b in clean_blocks(blocks)] == ["Real text", "Section IV", "10"]
