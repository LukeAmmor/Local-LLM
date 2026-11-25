# Document corpus

The Phase 1 corpus: public documents from international organisations,
assigned to three departments plus a public area, in English, French and
Arabic. The list lives in `data/manifest.yaml`. The files are not in the
repository (`data/raw/` is gitignored): `uv run python -m app.download`
fetches them, except the ones marked `manual_download: true`. A document
with `exclude:` stays in the manifest as a documented finding: the download
check still examines it, but it is never indexed.

| id | department | language | format | pages | chunks |
|---|---|---|---|---|---|
| wb-procurement-regs-en | procurement | en | PDF | 154 | 293 |
| wb-procurement-regs-fr | procurement | fr | DOCX | none (cited by section) | 254 |
| wbg-po-terms-ar | procurement | ar + en | PDF | 9 | 39 |
| anssi-hygiene-en | it-security | en | PDF | 72 | 57 |
| anssi-hygiene-fr | it-security | fr | PDF | 72 | 64 |
| anssi-ransomware-fr | it-security | fr | PDF | 21 | 39 |
| itcilo-conduct-en | hr | en | PDF | 11 | 23 |
| ilo-harassment-policy-en | hr | en | PDF | 4 | 8 |
| ilo-c190-brief-ar | hr | ar | PDF | 6 | excluded: damaged text (see below) |
| ilo-job-evaluation-ar | hr | ar | PDF | - | file not placed yet |
| wb-anticorruption-en | public | en | PDF | 7 | 14 |
| wb-anticorruption-guide-fr | public | fr | PDF | 35 | 37 |

Chunks: at most 400 bge-m3 tokens, 15% overlap (settings of Phase 1).

## Realistic cases kept on purpose

**A bilingual document (wbg-po-terms-ar).** The World Bank Group purchase
order terms "in Arabic" are in fact bilingual: each page has the English text
in one column and the Arabic translation in the other, and the PDF stores them
paragraph by paragraph. The document states that the English version governs.
Internal documents often mix languages like this. Language is therefore
detected for each chunk, not only for each document: 25 chunks are mostly
English and 14 mostly Arabic. Because bge-m3 is cross-lingual, an Arabic
question and an English question both retrieve a mix of the English and
Arabic chunks of the same clause (measured on 4 questions): the question's
language does not decide which version comes back.

**The same content in two languages.** The ANSSI hygiene guide exists in
English and French (same edition: version 2.0, September 2017). A question
can retrieve both versions; bge-m3 places them close to each other.

**Two editions of the same regulation.** The procurement regulations are the
7th edition (September 2025) in English and the 6th edition (February 2025) in
French. The two can differ, and the citation tells which edition an answer
comes from.

## Extraction quality

**Arabic PDF (wbg-po-terms-ar).** The Arabic text comes out in the right
order. Two extraction errors are repaired in `app.parsing.repair_arabic_span`:
  - the lam-alef ligature "لا" comes out with its two letters swapped
    ("واألحكام" instead of "والأحكام"): 202 repairs;
  - punctuation at the end of an Arabic segment comes out at its start: 136
    lines, 3 left.

Remaining, minor for search: numbers inside Arabic sentences come out out of
place ("( ثلاثين30 يوما" instead of "ثلاثين (30) يوما"), and a few words are
split ("مجموع ة").

**Arabic PDFs with unusable text: two findings.** Text extraction from
Arabic PDFs depends on how the PDF was made: a PDF shows glyphs (shapes), and
the text comes from a table inside the PDF that maps glyphs back to letters.
When that table is missing or wrong, the page looks perfect but the
extracted text is broken. Two Arabic HR documents failed the download check
this way. Such files are replaced, not repaired.
  1. An ILO HIV/AIDS guide for employers (2008) was made with QuarkXPress
     and old Mac Arabic fonts ("AXt"). These fonts have no Unicode mapping
     at all, so the text comes out as Latin symbols: "πª©dG" instead of
     "العمل", with 0 Arabic letters out of 11,185 (flag "no Arabic
     letters"). It was removed from the corpus.
  2. Its replacement, the ILO brief on Convention 190 (ilo-c190-brief-ar,
     2023), was exported to PDF by Word with the Calibri font. The text is
     Unicode, but letter pairs drawn as one glyph (ligatures) come out
     wrong: "في" is extracted as "ف" 83 times out of 86, "على" as "عىل",
     "إلى" as "إىل", "التي" as "النر", "القوانين" as "القوانير" (flag "damaged
     Arabic letters"). The damaged endings cannot be repaired by rule ("ير"
     stands for "ين", "يز"...). It stays in the manifest with `exclude:` so
     the finding can be reproduced, and is not indexed.

By contrast, wbg-po-terms-ar (also made with Word, but another font) has
none of these problems: "في" appears 80 times and a lone "ف" never does.
The backup Arabic HR document, ilo-job-evaluation-ar, is pending: its file is
not placed yet.

**French DOCX (wb-procurement-regs-fr).**
  - Headings come from Word's outline levels, the setting that builds Word's
    table of contents. Most of this document's headings use custom styles
    ("Style2", "REGs H1"), which a style-name rule would miss.
  - Paragraph numbers such as "6.23" are Word automatic numbering. They are
    not part of the text, so French chunks do not contain them; the English
    PDF does.
  - No pages: a DOCX has no fixed pages (Word computes them when it displays
    or prints the file). Word saves "rendered page break" markers, but this
    file has 145 markers for 151 pages (Word's own count), and a table row
    that starts a new page gets one marker per cell. Measured against the
    printed page numbers of the document's own table of contents, the
    computed pages drifted by up to 8 pages. DOCX chunks therefore have no
    page, and their citations show the section ("Plan de Passation des
    Marchés"). 237 of the 254 chunks have a section; the other 17 are the
    front matter (cover, foreword, definitions, table of contents).

**Design-heavy PDFs (ANSSI guides).** Chapter pages print a large roman
numeral ("IV") next to the chapter title. The numeral alone is not a heading
and is dropped during cleaning; the title is a level-1 heading.
