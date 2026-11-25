"""Download the documents listed in data/manifest.yaml, then check each file.

Usage:
    uv run python -m app.download            # all documents
    uv run python -m app.download --only anssi-hygiene-en

What it does, for each manifest entry:
  1. If the file is already in data/raw/<department>/, skip the download.
     If `manual_download: true` and the file is missing, remind you to place it.
     Otherwise download it from `source_url`.
  2. Check that the file opens and contains extractable text.
  3. Print a summary: pages, characters per page, declared vs detected language.

Flags printed for a file:
  - "possible scan": fewer than MIN_CHARS_PER_PAGE characters per page on
    average. The PDF is probably made of images and would need OCR.
  - "N pages almost empty": some pages have almost no text (scanned pages,
    figures, or blank pages).
  - "language": the detected language differs from the declared one.
  - "Arabic order": Arabic words come out reversed (see app.langdetect).
  - "no Arabic letters": an Arabic document whose text comes out as Latin
    symbols ("πª©dG" instead of "العمل"). Its fonts have no Unicode mapping
    (old desktop-publishing fonts), so the text layer is unusable: the file
    needs OCR or must be replaced.
  - "damaged Arabic letters": the text is Arabic, but letter pairs drawn as a
    single glyph (ligatures) come out wrong, e.g. "في" as "ف", "على" as "عىل",
    "القوانين" as "القوانير". Seen in PDFs exported by Word with the Calibri
    font. The test: "ف" alone is never an Arabic word while "في" is one of the
    most frequent ones, so many lone "ف" and few "في" means damage.

Raw documents are never committed: data/raw/ is in .gitignore.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import docx
import httpx
import pymupdf

from app.config import get_settings
from app.langdetect import ARABIC_LETTER, arabic_word_order, detect_language
from app.manifest import DocumentEntry, load_manifest

USER_AGENT = "Mozilla/5.0 (compatible; local-rag-downloader/0.1)"
MIN_CHARS_PER_PAGE = 200  # below this on average, the PDF is probably scanned
EMPTY_PAGE_CHARS = 50  # a page with fewer characters counts as "almost empty"
MIN_ARABIC_SHARE = 0.05  # an Arabic document with a smaller share of Arabic letters is unreadable
MIN_BROKEN_WORDS = 5  # a few lone "ف" can be typos; more than this (and more than "في") is damage


def download(url: str, dest: Path) -> str | None:
    """Download url to dest. Return an error message, or None on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")  # renamed only when complete
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60,
                          headers={"User-Agent": USER_AGENT}) as response:
            if response.status_code != 200:
                return f"HTTP {response.status_code}"
            if "text/html" in response.headers.get("content-type", ""):
                return "the URL returned a web page, not a file"
            with open(partial, "wb") as f:
                for block in response.iter_bytes():
                    f.write(block)
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        return f"{type(exc).__name__}: {exc}"
    partial.replace(dest)
    return None


def page_texts(path: Path) -> tuple[list[str], bool]:
    """Raw text per page, and whether the format really has pages.

    DOCX, Markdown and TXT have no pages: their whole text is returned as one item.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pymupdf.TOOLS.mupdf_display_errors(False)  # harmless warnings such as "No default Layer config"
        with pymupdf.open(path) as pdf:
            return [page.get_text() for page in pdf], True
    if suffix == ".docx":
        document = docx.Document(str(path))
        return ["\n".join(p.text for p in document.paragraphs)], False
    return [path.read_text(encoding="utf-8")], False


@dataclass
class FileCheck:
    pages: int | None = None
    chars: int = 0
    chars_per_page: float | None = None
    min_chars_page: int | None = None
    detected_language: str = "?"
    flags: list[str] = field(default_factory=list)


def check_file(path: Path, entry: DocumentEntry) -> FileCheck:
    """Open the file and measure how much text it contains."""
    check = FileCheck()
    try:
        texts, has_pages = page_texts(path)
    except Exception as exc:  # noqa: BLE001 - any failure means "cannot open"
        check.flags.append(f"cannot open: {type(exc).__name__}: {exc}")
        return check

    lengths = [len(t.strip()) for t in texts]
    check.chars = sum(lengths)
    full_text = "\n".join(texts)
    check.detected_language = detect_language(full_text)

    if has_pages:
        check.pages = len(texts)
        check.chars_per_page = check.chars / max(len(texts), 1)
        check.min_chars_page = min(lengths) if lengths else 0
        empty = sum(n < EMPTY_PAGE_CHARS for n in lengths)
        if check.chars_per_page < MIN_CHARS_PER_PAGE:
            check.flags.append(f"possible scan ({check.chars_per_page:.0f} chars/page): may need OCR")
        elif empty:
            check.flags.append(f"{empty}/{len(texts)} pages almost empty")
    if check.chars == 0:
        check.flags.append("no extractable text")
    if check.detected_language != entry.language:
        check.flags.append(f"language: declared {entry.language}, detected {check.detected_language}")
    if entry.language == "ar":
        letters = sum(ch.isalpha() for ch in full_text)
        arabic = len(ARABIC_LETTER.findall(full_text))
        if letters and arabic < MIN_ARABIC_SHARE * letters:
            check.flags.append(f"no Arabic letters ({arabic} of {letters}): fonts without Unicode mapping, "
                               "text unusable (needs OCR or another file)")
        words = re.findall(r"[ء-ي]+", full_text)
        broken_fi, fi = words.count("ف"), words.count("في")
        if broken_fi >= MIN_BROKEN_WORDS and broken_fi > fi:
            check.flags.append(f"damaged Arabic letters ('في' extracted as 'ف' {broken_fi} times, correct {fi} times): "
                               "font ligatures with a wrong Unicode mapping")
        normal, reversed_ = arabic_word_order(full_text)
        if reversed_ > normal:
            check.flags.append(f"Arabic order looks reversed (common words: {normal} normal, {reversed_} reversed)")
    return check


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # titles contain Arabic; Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Download and check the manifest documents.")
    parser.add_argument("--only", nargs="+", metavar="DOC_ID", help="only these document ids")
    args = parser.parse_args()

    settings = get_settings()
    manifest = load_manifest(settings.manifest_path)
    entries = [manifest.get(i) for i in args.only] if args.only else manifest.documents

    problems = 0
    rows = []
    for entry in entries:
        path = entry.file_path(settings.raw_dir)
        if path.exists():
            status = "present"
        elif entry.manual_download:
            status = "MISSING"
            print(f"[{entry.id}] manual download: save the file as {path}\n"
                  f"    source page: {entry.source_url}")
        else:
            print(f"[{entry.id}] downloading {entry.source_url}")
            error = download(entry.source_url, path)
            status = "downloaded" if error is None else "FAILED"
            if error:
                print(f"    failed: {error}. Download it by hand and save it as {path}")

        check = check_file(path, entry) if path.exists() else FileCheck()
        if status in ("MISSING", "FAILED") or any(f.startswith(("cannot open", "no extractable")) for f in check.flags):
            problems += 1
        rows.append((entry, status, check))

    print(f"\n{'id':<28} {'status':<10} {'lang':<7} {'pages':>5} {'chars':>8} {'chars/pg':>8} {'min/pg':>6}  flags")
    print("-" * 110)
    for entry, status, c in rows:
        pages = "-" if c.pages is None else str(c.pages)
        cpp = "-" if c.chars_per_page is None else f"{c.chars_per_page:.0f}"
        mpp = "-" if c.min_chars_page is None else str(c.min_chars_page)
        lang = f"{entry.language}/{c.detected_language}"
        print(f"{entry.id:<28} {status:<10} {lang:<7} {pages:>5} {c.chars:>8} {cpp:>8} {mpp:>6}  {'; '.join(c.flags)}")
    print("lang = declared/detected. DOCX/MD/TXT have no pages, so no per-page numbers.")
    print(f"\n{len(rows) - problems}/{len(rows)} documents ready." + (f" {problems} need attention." if problems else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
