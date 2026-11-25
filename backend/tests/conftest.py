"""Shared test helpers.

`WordCounter` counts one token per word. It replaces the bge-m3 tokenizer in
chunking tests: the numbers are easy to check by hand and no model is loaded.
"""

import pytest


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())

    def windows(self, text: str, size: int, overlap: int) -> list[str]:
        words = text.split()
        step = max(size - overlap, 1)
        pieces = []
        for start in range(0, len(words), step):
            pieces.append(" ".join(words[start : start + size]))
            if start + size >= len(words):
                break
        return pieces


@pytest.fixture
def words():
    return WordCounter()
