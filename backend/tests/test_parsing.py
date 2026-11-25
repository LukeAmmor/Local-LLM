"""Tests for app.parsing, with small documents created on the fly (no real files)."""

import docx
import pymupdf
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from app.parsing import heading_levels, parse_document, repair_arabic_span


def make_pdf(path):
    """Two pages: a big heading, body text, and a second heading on page 2."""
    pdf = pymupdf.open()
    for page_number, title in [(1, "Introduction"), (2, "Procurement Methods")]:
        page = pdf.new_page()
        page.insert_text((72, 100), title, fontsize=18)
        page.insert_textbox(pymupdf.Rect(72, 130, 520, 400),
                            "The Borrower shall prepare a procurement plan. " * 8, fontsize=10)
    pdf.save(path)


def test_pdf_headings_and_pages(tmp_path):
    path = tmp_path / "doc.pdf"
    make_pdf(path)
    parsed = parse_document(path)
    assert parsed.page_count == 2
    headings = [(b.text, b.page, b.level) for b in parsed.blocks if b.is_heading]
    assert headings == [("Introduction", 1, 1), ("Procurement Methods", 2, 1)]
    body = [b for b in parsed.blocks if not b.is_heading]
    assert {b.page for b in body} == {1, 2}


def test_pdf_chapter_number_alone_is_not_a_heading(tmp_path):
    path = tmp_path / "chapters.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 120), "III", fontsize=60)  # big chapter number, decoration only
    page.insert_text((72, 180), "Secure the network", fontsize=20)
    page.insert_textbox(pymupdf.Rect(72, 220, 520, 500), "Segment the network into zones. " * 10, fontsize=10)
    pdf.save(path)
    headings = [b.text for b in parse_document(path).blocks if b.is_heading]
    assert headings == ["Secure the network"]


def test_docx_headings_tables_and_no_pages(tmp_path):
    path = tmp_path / "doc.docx"
    document = docx.Document()
    document.add_heading("Leave policy", level=1)
    document.add_paragraph("Employees have 25 days of leave per year.")
    document.add_heading("Sick leave", level=2)
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Days"
    table.rows[0].cells[1].text = "10"
    document.save(path)

    parsed = parse_document(path)
    assert [(b.text, b.level) for b in parsed.blocks if b.is_heading] == [("Leave policy", 1), ("Sick leave", 2)]
    assert "Days | 10" in [b.text for b in parsed.blocks]
    assert parsed.page_count is None and all(b.page is None for b in parsed.blocks)


def test_docx_page_markers_are_ignored(tmp_path):
    """Even with Word's saved page-break markers, DOCX blocks have no page (cited by section)."""
    path = tmp_path / "pages.docx"
    document = docx.Document()
    document.add_paragraph("First page.")
    paragraph = document.add_paragraph()
    paragraph.add_run()._r.append(OxmlElement("w:lastRenderedPageBreak"))  # what Word saves
    paragraph.add_run("Second page.")
    document.save(path)
    parsed = parse_document(path)
    assert [b.text for b in parsed.blocks] == ["First page.", "Second page."]
    assert parsed.page_count is None and all(b.page is None for b in parsed.blocks)


def test_docx_custom_heading_styles_use_outline_level(tmp_path):
    """Like the World Bank French DOCX: headings in custom styles, recognised by their outline level."""
    path = tmp_path / "custom.docx"
    document = docx.Document()
    section_style = document.styles.add_style("REGs H1", WD_STYLE_TYPE.PARAGRAPH)
    section_style.base_style = document.styles["Heading 1"]  # inherits outline level 0 = level 1
    topic_style = document.styles.add_style("Style2", WD_STYLE_TYPE.PARAGRAPH)
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), "1")  # outline level 1 = level 2
    topic_style.element.get_or_add_pPr().append(outline)
    document.add_paragraph("Section I. Introduction", style="REGs H1")
    document.add_paragraph("Champ d'application", style="Style2")
    document.add_paragraph("Le présent Règlement s'applique aux marchés financés par la Banque.")
    document.add_paragraph("mot " * 50, style="Style2")  # 200 characters: too long to be a heading
    document.save(path)

    blocks = parse_document(path).blocks
    assert [(b.text, b.level) for b in blocks if b.is_heading] == [
        ("Section I. Introduction", 1), ("Champ d'application", 2)]


def test_markdown(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nFirst paragraph\non two lines.\n\n## Part\n\nSecond.", encoding="utf-8")
    blocks = parse_document(path).blocks
    assert [(b.text, b.is_heading, b.level) for b in blocks] == [
        ("Title", True, 1), ("First paragraph\non two lines.", False, 0), ("Part", True, 2), ("Second.", False, 0)]


def test_heading_levels_group_close_sizes():
    levels = heading_levels([24.0, 16.0, 15.5, 13.0, 12.0, 10.0])
    assert levels[24.0] == 1
    assert levels[16.0] == levels[15.5] == 2  # less than 1 point apart
    assert levels[13.0] == 3 and levels[10.0] == 3  # capped at level 3


def char(c: str, x0: float, x1: float) -> dict:
    return {"c": c, "bbox": (x0, 0.0, x1, 10.0)}


def test_arabic_lam_alef_ligature_is_swapped_back():
    # "والأحكام" extracted as و ا أ(zero width) ل ح ك ا م
    chars = [char("و", 26, 32), char("ا", 25, 28), char("أ", 25.78, 25.78), char("ل", 19.9, 25.78),
             char("ح", 13, 19), char("ك", 9, 13), char("ا", 6, 9), char("م", 2, 6)]
    assert repair_arabic_span(chars) == "والأحكام"


def test_normal_alef_lam_is_not_touched():
    # "الشراء": the article ا ل with a normal-width alef must stay as it is.
    chars = [char(c, 50 - 5 * i, 54 - 5 * i) for i, c in enumerate("الشراء")]
    assert repair_arabic_span(chars) == "الشراء"


def test_arabic_leading_punctuation_moves_to_the_end():
    chars = [char(c, 0, 4) for c in ".في هذا الطلب"]
    assert repair_arabic_span(chars) == "في هذا الطلب."
    latin = [char(c, 0, 4) for c in ". Next"]
    assert repair_arabic_span(latin) == ". Next"  # only Arabic segments are changed
