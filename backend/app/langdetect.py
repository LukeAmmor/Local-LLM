"""A small language detector for our two languages: English and French.

What: `detect_language(text)` returns "en", "fr", "other" or "unknown".
  1. Count letters. If fewer than half of them are Latin letters, the text is
     written in another script (Arabic, Cyrillic, Chinese...): "other".
  2. Otherwise count very common words that exist in only one of the two
     languages ("the", "and", "of"... vs "le", "et", "des"...). The language
     with more hits wins.
  3. No letters at all, or a tie: "unknown".

Ingestion uses it twice: on the whole document (a document that is not
English or French is skipped, see the setting `supported_languages`) and on
each chunk (stored in Qdrant, so search results can show or filter it).

Why: only two languages, so a full language-identification package is not
needed. This fits in a few lines that can be explained one by one.
"""

import re

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


def detect_language(text: str) -> str:
    letters = sum(ch.isalpha() for ch in text)
    if letters == 0:
        return "unknown"
    if len(LATIN_LETTER.findall(text)) < letters / 2:
        return "other"

    words = LATIN_WORD.findall(text.lower())
    en = sum(w in ENGLISH_WORDS for w in words)
    fr = sum(w in FRENCH_WORDS for w in words)
    if en == fr:
        return "unknown"
    return "en" if en > fr else "fr"
