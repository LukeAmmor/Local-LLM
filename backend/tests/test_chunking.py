"""Tests for app.chunking: sizes, overlap, sections, pages, headers."""

import re

import pytest

from app.chunking import HFTokenCounter, chunk_document
from app.parsing import Block


def sentence(i: int, n_words: int = 10) -> str:
    """A sentence of exactly n_words words, recognisable by its number."""
    return " ".join([f"s{i}"] + ["word"] * (n_words - 2) + ["end."])


def paragraph(first: int, n_sentences: int, page: int = 1) -> Block:
    return Block(text=" ".join(sentence(first + k) for k in range(n_sentences)), page=page)


def heading(text: str, level: int = 1, page: int = 1) -> Block:
    return Block(text=text, page=page, is_heading=True, level=level)


def ids(text: str) -> list[str]:
    """Sentence numbers found in a chunk text, in order: ["s0", "s1", ...]."""
    return re.findall(r"\bs\d+\b", text)


def chunk(blocks, words, size=40, overlap=0.25, header=True):
    return chunk_document(blocks, words, "Doc Title", "en", chunk_size=size, overlap_ratio=overlap, add_header=header)


def test_chunks_never_exceed_chunk_size(words):
    chunks = chunk([paragraph(0, 30)], words, size=40)
    assert len(chunks) > 1
    assert all(c.n_tokens <= 40 for c in chunks)


def test_overlap_repeats_last_sentences_within_budget(words):
    # 10-word sentences, chunk 40 words, overlap 25% = 10 words = 1 sentence.
    chunks = chunk([paragraph(0, 12)], words, size=40, overlap=0.25)
    assert len(chunks) >= 3
    for prev, nxt in zip(chunks, chunks[1:]):
        assert ids(nxt.text)[0] == ids(prev.text)[-1]  # the next chunk starts with the last sentence
        assert len(set(ids(prev.text)) & set(ids(nxt.text))) == 1  # and repeats only that one


def test_no_overlap_when_disabled(words):
    chunks = chunk([paragraph(0, 12)], words, size=40, overlap=0.0)
    all_ids = [i for c in chunks for i in ids(c.text)]
    assert all_ids == [f"s{k}" for k in range(12)]  # every sentence exactly once, in order


def test_heading_starts_new_chunk_without_overlap(words):
    blocks = [heading("Intro"), paragraph(0, 3), heading("Rules"), paragraph(10, 3)]
    chunks = chunk(blocks, words, size=60)
    assert [c.section for c in chunks] == ["Intro", "Rules"]
    assert "s0 " not in chunks[1].text  # nothing carried over from the previous section
    assert chunks[1].text.startswith("Rules")


def test_tiny_chunk_continues_across_heading(words):
    # The first section has only 5 words, less than 25% of 40: it does not become its own chunk.
    blocks = [heading("Term A"), Block(text="A short definition here.", page=1),
              heading("Term B"), paragraph(0, 2)]
    chunks = chunk(blocks, words, size=40)
    assert len(chunks) == 1
    assert "Term A" in chunks[0].text and "Term B" in chunks[0].text
    assert chunks[0].section == "Term A"


def test_section_path_follows_heading_levels(words):
    blocks = [heading("Part 1", 1), heading("Chapter A", 2), paragraph(0, 3),
              heading("Chapter B", 2), paragraph(10, 3), heading("Part 2", 1), paragraph(20, 3)]
    sections = [c.section for c in chunk(blocks, words, size=60)]
    assert sections == ["Part 1 > Chapter A", "Part 1 > Chapter B", "Part 2"]


def test_pages_cover_all_sentences_of_the_chunk(words):
    blocks = [paragraph(0, 2, page=3), paragraph(10, 2, page=4), paragraph(20, 5, page=5)]
    chunks = chunk(blocks, words, size=40, overlap=0.0)
    assert (chunks[0].page_start, chunks[0].page_end) == (3, 4)
    assert chunks[-1].page_end == 5


def test_docx_without_pages_gives_none(words):
    chunks = chunk([Block(text=sentence(0), page=None)], words)
    assert chunks[0].page_start is None and chunks[0].page_end is None


def test_sentence_longer_than_chunk_is_split(words):
    long_sentence = Block(text=" ".join(["word"] * 100) + ".", page=1)
    chunks = chunk([long_sentence], words, size=40, overlap=0.25)
    assert len(chunks) >= 3
    assert all(c.n_tokens <= 40 for c in chunks)


def test_header_only_in_embed_text(words):
    c = chunk([heading("Rules"), paragraph(0, 2)], words, header=True)[0]
    assert c.embed_text.startswith("Doc Title > Rules\n\n")
    assert not c.text.startswith("Doc Title")
    c_no_header = chunk([heading("Rules"), paragraph(0, 2)], words, header=False)[0]
    assert c_no_header.embed_text == c_no_header.text


def test_chunk_language_is_detected(words):
    blocks = [Block(text="Le responsable des achats doit vérifier que le contrat est signé.", page=1)]
    assert chunk(blocks, words)[0].language == "fr"


def test_real_bge_m3_tokenizer_respects_chunk_size():
    """Same check with the real tokenizer (skipped if the model is not downloaded)."""
    try:
        from app.embeddings import load_tokenizer
        tokenizer = load_tokenizer("BAAI/bge-m3")
    except Exception:  # noqa: BLE001
        pytest.skip("bge-m3 tokenizer not available")
    counter = HFTokenCounter(tokenizer)
    text = ("Le responsable des achats doit vérifier chaque contrat. "
            "The Borrower shall apply the Procurement Regulations. "
            "يجب على المورد احترام شروط طلب الشراء. ") * 60
    chunks = chunk_document([Block(text=text, page=1)], counter, "T", "en", chunk_size=100, overlap_ratio=0.15)
    assert len(chunks) > 5
    assert all(c.n_tokens <= 100 for c in chunks)
