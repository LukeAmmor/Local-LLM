# Document corpus

The corpus: public documents from international organisations, assigned to
three departments plus a public area, in **English and French**. The list
lives in `data/manifest.yaml`. The files are not in the repository
(`data/raw/` is gitignored): `uv run python -m app.download` fetches them,
except the ones marked `manual_download: true`. A document with `exclude:`
stays in the manifest as a documented finding: it is never indexed.

| id | department | language | format | pages | chunks |
|---|---|---|---|---|---|
| wb-procurement-regs-en | procurement | en | PDF | 154 | 293 |
| wb-procurement-regs-fr | procurement | fr | DOCX | none (cited by section) | 254 |
| anssi-hygiene-en | it-security | en | PDF | 72 | 57 |
| anssi-hygiene-fr | it-security | fr | PDF | 72 | 64 |
| anssi-ransomware-fr | it-security | fr | PDF | 21 | 39 |
| itcilo-conduct-en | hr | en | PDF | 11 | 23 |
| ilo-harassment-policy-en | hr | en | PDF | 4 | 8 |
| wb-anticorruption-en | public | en | PDF | 7 | 14 |
| wb-anticorruption-guide-fr | public | fr | PDF | 35 | 37 |
| **total** | | 393 en + 396 fr chunks | | | **789** |
| wbg-po-terms-ar | procurement | ar + en | PDF | 9 | excluded: Arabic dropped |
| ilo-c190-brief-ar | hr | ar | PDF | 6 | excluded: Arabic dropped |
| ilo-job-evaluation-ar | hr | ar | PDF | 108 | excluded: Arabic dropped |

Chunks: at most 400 bge-m3 tokens, 15% overlap (settings of Phase 1).

## Cross-lingual cases

The embedder, bge-m3, is multilingual: a French question can find an
English passage and the other way round.

**A French question about an English document.** The only harassment
policy is in English (ILO Circular 543). The French question "Quelle est la
procédure en cas de harcèlement sexuel au travail ?" finds it in the top 5
with dense vectors. Sparse (keyword) vectors alone fail on it: French and
English words do not match. This is a test case for Phase 5, and a reason
for hybrid search.

**The same content in two languages.** The ANSSI hygiene guide exists in
English and French (same edition: version 2.0, September 2017). A question
can retrieve both versions; bge-m3 places them close to each other.

**Two editions of the same regulation.** The procurement regulations are the
7th edition (September 2025) in English and the 6th edition (February 2025) in
French. The two can differ, and the citation tells which edition an answer
comes from.

## Why Arabic was dropped from scope

The project started with three languages: English, French and Arabic.
Arabic was dropped after Phase 1, for two reasons measured on real files.

**1. Most Arabic PDFs gave broken text.** A PDF shows glyphs (shapes), and
the text comes from a table inside the PDF that maps glyphs back to letters.
When that table is missing or wrong, the page looks perfect on screen but
the extracted text is broken. Three Arabic HR documents failed this way:
  1. An ILO HIV/AIDS guide for employers (2008) was made with QuarkXPress
     and old Mac Arabic fonts ("AXt"). These fonts have no Unicode mapping
     at all, so the text comes out as Latin symbols: "πª©dG" instead of
     "العمل", with 0 Arabic letters out of 11,185. Removed from the corpus.
  2. The ILO brief on Convention 190 (ilo-c190-brief-ar, 2023) was exported
     to PDF by Word with the Calibri font. The text is Unicode, but letter
     pairs drawn as one glyph (ligatures) come out wrong: "في" is extracted
     as "ف" 83 times out of 86, "على" as "عىل", "إلى" as "إىل", "التي" as
     "النر", "القوانين" as "القوانير". The damaged endings cannot be repaired
     by rule ("ير" stands for "ين", "يز"...).
  3. An ILO guide to gender-neutral job evaluation (ilo-job-evaluation-ar,
     2011, 108 pages) uses the same "AXt" fonts as file 1, this time with a
     partial and wrong Unicode mapping: "في" comes out as "يف" 882 times
     (correct: 10), "المرأة" as "املراأة", "أن" as "اأن", "الرئيسية" as
     "الرئيùشية", and the cover titles as Latin symbols ("õjõ©J").

The damage depends on the tool and fonts that produced the PDF, not on the
publisher: the same organisation (ILO) produced all three.

**2. The one usable Arabic document needed repair code.** The World Bank
Group purchase order terms (wbg-po-terms-ar, 2025) were bilingual: English
in one column, the Arabic translation in the other. Its Arabic text came out
in the right order, but with two systematic extraction errors, which the
parser repaired (code removed when Arabic was dropped):
  - The lam-alef ligature "لا" came out with its two letters swapped
    ("واألحكام" instead of "والأحكام"). PyMuPDF splits the ligature glyph
    into an alef of zero width followed by a lam; the zero width was the
    marker to detect it and swap the two letters back: 202 repairs.
  - Punctuation at the end of an Arabic segment came out at its start
    (".في هذا الطلب"); it was moved back to the end: 136 lines, 3 left.

Still unrepaired: numbers inside Arabic sentences out of place
("( ثلاثين30 يوما" instead of "ثلاثين (30) يوما"), and a few split words
("مجموع ة").

The Phase 1 download check also had Arabic-specific flags ("no Arabic
letters", "damaged Arabic letters": many lone "ف" or "يف" and few "في"); they
found the three broken files above and were removed with Arabic. All this
code is in the git history (commit 8ab35ef).

**The decision.** Supporting Arabic meant writing and explaining
font-specific repair code, checking every new file by hand, and still
accepting damaged text. English and French text extracts cleanly. The
three Arabic files stay in the manifest with `exclude:`, and the language
detector knows English and French only: a document detected as another
language is skipped by ingestion with a message.

## Extraction quality (English and French)

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
