"""A small language detector for our three languages: English, French, Arabic.

What: `detect_language(text)` returns "en", "fr", "ar" or "unknown".
  1. Count letters by script. If most letters are Arabic script -> "ar".
  2. Otherwise count very common words that exist in only one of the two
     languages ("the", "and", "of"... vs "le", "et", "des"...). The language
     with more hits wins.

Also `arabic_word_order(text)`: a check for Arabic text extracted from PDFs
in the wrong order. Some PDFs store Arabic glyphs in visual (left-to-right)
order, so the extracted words come out reversed: "في" becomes "يف". We count
very common Arabic words in their normal and reversed spelling.

Why: only three languages, so a full language-identification package is not
needed. This fits in a few lines that can be explained one by one.
"""

import re

ARABIC_LETTER = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
LATIN_LETTER = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœ]")
LATIN_WORD = re.compile(r"[a-zà-öø-ÿœæç]+")  # apostrophes split "l'article" into "l", "article"

# Frequent words that belong to only one of the two languages.
ENGLISH_WORDS = {
    "the", "and", "of", "to", "is", "are", "that", "this", "with", "for", "be", "by",
    "which", "shall", "should", "not", "from", "has", "have", "it", "its", "any", "will",
}
FRENCH_WORDS = {
    "le", "la", "les", "des", "du", "et", "est", "une", "un", "pour", "que", "qui",
    "dans", "par", "sur", "au", "aux", "avec", "sont", "doit", "ne", "pas", "l", "d", "qu",
}

# Common Arabic words; written backwards they are (almost) never real words.
ARABIC_COMMON = ["في", "من", "على", "إلى", "التي", "الذي", "أن", "عن", "هذا", "هذه"]


def detect_language(text: str) -> str:
    arabic = len(ARABIC_LETTER.findall(text))
    latin = len(LATIN_LETTER.findall(text))
    if arabic + latin == 0:
        return "unknown"
    if arabic > latin:
        return "ar"

    words = LATIN_WORD.findall(text.lower())
    en = sum(w in ENGLISH_WORDS for w in words)
    fr = sum(w in FRENCH_WORDS for w in words)
    if en == fr:
        return "unknown"
    return "en" if en > fr else "fr"


def arabic_word_order(text: str) -> tuple[int, int]:
    """Return (normal, reversed): how often common Arabic words appear each way.

    Much more `reversed` than `normal` means the text was extracted in the
    wrong (visual) order.
    """
    words = re.findall(r"[؀-ۿ]+", text)
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    normal = sum(counts.get(w, 0) for w in ARABIC_COMMON)
    reversed_ = sum(counts.get(w[::-1], 0) for w in ARABIC_COMMON)
    return normal, reversed_
