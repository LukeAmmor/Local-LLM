"""Structure-aware chunking: cut a document into passages of at most N tokens.

What: `chunk_document(blocks, ...)` turns the cleaned blocks of one document
into chunks of at most `chunk_size` tokens (bge-m3 tokens, default 400).

How, step by step:
  1. Sections. Walk through the blocks and keep the current heading path,
     e.g. "Section III > Procurement Methods". Every paragraph belongs to the
     section of the last heading above it.
  2. Sentences. Each paragraph is split into sentences. These are the pieces
     we pack into chunks. A sentence longer than a whole chunk (rare) is cut
     into token windows.
  3. Packing. Add sentences to the current chunk until the next one would
     exceed `chunk_size`, then start a new chunk.
  4. Headings start a new chunk, so a chunk normally stays inside one
     section. Exception: while the current chunk is still very small (less
     than a quarter of `chunk_size`), it continues across the heading. Some
     PDFs have many short bold lines detected as headings (table labels,
     defined terms), and cutting at each of them would create tiny chunks.
  5. Overlap. When a chunk is cut because it is full, the next chunk starts
     by repeating the last sentences of the previous one, up to
     `overlap_ratio * chunk_size` tokens (default 15%, i.e. 60 tokens). A
     sentence cut in two chunks is still found by the search. There is no
     overlap after a heading: a new section starts clean.

Each chunk records its pages (page_start, page_end), its section, and two
texts: `text` (what the user sees in a citation) and `embed_text` (what the
embedder sees). With the chunk header switched on, `embed_text` starts with
"Title > Section", which tells the embedder what the passage is about.

The token counter is passed in (see `TokenCounter`), so tests can use a
simple word counter instead of the real bge-m3 tokenizer.
"""

import re
from dataclasses import dataclass
from typing import Protocol

from app.langdetect import detect_language
from app.parsing import Block

MIN_CHUNK_SHARE = 0.25  # below this share of chunk_size, a chunk continues across a heading
MAX_HEADING_IN_PATH = 80  # characters kept from each heading in the section path

SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def windows(self, text: str, size: int, overlap: int) -> list[str]:
        """Cut text into pieces of at most `size` tokens, overlapping by `overlap` tokens."""
        ...


class HFTokenCounter:
    """Token counter backed by a Hugging Face tokenizer (bge-m3's in practice)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def windows(self, text: str, size: int, overlap: int) -> list[str]:
        # Offsets map each token back to its characters, so we cut the
        # original text exactly at token boundaries.
        offsets = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        step = max(size - overlap, 1)
        pieces = []
        for start in range(0, len(offsets), step):
            window = offsets[start : start + size]
            pieces.append(text[window[0][0] : window[-1][1]].strip())
            if start + size >= len(offsets):
                break
        return pieces


@dataclass
class Piece:
    """A sentence (or a token window of a long sentence), the unit we pack."""

    text: str
    tokens: int
    page: int | None
    paragraph: int  # pieces of the same paragraph are joined by a space, others by a newline
    section: str
    is_heading: bool = False  # the text of a heading line


@dataclass
class Chunk:
    index: int  # position in the document: 0, 1, 2...
    text: str
    embed_text: str
    section: str
    page_start: int | None
    page_end: int | None
    n_tokens: int
    language: str


def with_header(title: str, section: str, text: str) -> str:
    """The text preceded by its "Title > Section" header (embedder input, also reranker input)."""
    header = f"{title} > {section}" if section else title
    return f"{header}\n\n{text}"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s and s.strip()]


def to_pieces(blocks: list[Block], counter: TokenCounter, chunk_size: int, overlap_tokens: int
              ) -> list[Piece | str]:
    """Flatten the blocks into pieces. A heading becomes the marker string "HEADING"."""
    out: list[Piece | str] = []
    path: list[tuple[int, str]] = []  # (level, heading text) from the top level down
    for paragraph_id, block in enumerate(blocks):
        if block.is_heading:
            path = [(lvl, txt) for lvl, txt in path if lvl < block.level]
            path.append((block.level, block.text[:MAX_HEADING_IN_PATH]))
            out.append("HEADING")
        section = " > ".join(txt for _, txt in path)
        for sentence in split_sentences(block.text):
            n = counter.count(sentence)
            if n <= chunk_size:
                out.append(Piece(sentence, n, block.page, paragraph_id, section, block.is_heading))
            else:
                for w in counter.windows(sentence, chunk_size, overlap_tokens):
                    out.append(Piece(w, counter.count(w), block.page, paragraph_id, section, block.is_heading))
    return out


def join_pieces(pieces: list[Piece]) -> str:
    text = ""
    for i, p in enumerate(pieces):
        if i == 0:
            text = p.text
        else:
            text += (" " if p.paragraph == pieces[i - 1].paragraph else "\n") + p.text
    return text


def chunk_document(blocks: list[Block], counter: TokenCounter, title: str, doc_language: str,
                   chunk_size: int = 400, overlap_ratio: float = 0.15, add_header: bool = True
                   ) -> list[Chunk]:
    overlap_tokens = int(chunk_size * overlap_ratio)
    min_tokens = int(chunk_size * MIN_CHUNK_SHARE)
    groups: list[list[Piece]] = []  # pieces of each chunk
    current: list[Piece] = []
    current_tokens = 0

    for item in to_pieces(blocks, counter, chunk_size, overlap_tokens):
        if item == "HEADING":
            if current_tokens >= min_tokens:  # big enough: close it, start the section clean
                groups.append(current)
                current, current_tokens = [], 0
            continue
        if current and current_tokens + item.tokens > chunk_size:
            groups.append(current)
            # Overlap: repeat the last sentences of the full chunk that fit in the budget.
            carried, carried_tokens = [], 0
            for p in reversed(current):
                if carried_tokens + p.tokens > overlap_tokens or carried_tokens + p.tokens + item.tokens > chunk_size:
                    break
                carried.insert(0, p)
                carried_tokens += p.tokens
            current, current_tokens = carried, carried_tokens
        current.append(item)
        current_tokens += item.tokens
    if current:
        groups.append(current)

    chunks = []
    for index, pieces in enumerate(groups):
        text = join_pieces(pieces)
        # The chunk's section is the one of its first real content (not of a
        # heading line): "Part 1" directly followed by "Chapter A" -> "Part 1 > Chapter A".
        content = [p for p in pieces if not p.is_heading] or pieces
        section = content[0].section
        pages = [p.page for p in pieces if p.page is not None]
        language = detect_language(text)
        chunks.append(Chunk(
            index=index,
            text=text,
            embed_text=with_header(title, section, text) if add_header else text,
            section=section,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            n_tokens=counter.count(text),
            language=language if language != "unknown" else doc_language,
        ))
    return chunks
