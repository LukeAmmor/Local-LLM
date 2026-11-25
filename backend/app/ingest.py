"""Index the documents into Qdrant: parse -> clean -> chunk -> embed -> store.

Usage:
    uv run python -m app.ingest                          # every document in the manifest
    uv run python -m app.ingest --only anssi-ransomware-fr
    uv run python -m app.ingest --dry-run                # parse + chunk only (no GPU, no Qdrant)
    uv run python -m app.ingest --collection kb_c256 --chunk-size 256
    uv run python -m app.ingest --no-header              # experiment: no "Title > Section" header
    uv run python -m app.ingest --force                  # re-index even if nothing changed

For each document of data/manifest.yaml:
  0. A document with `exclude:` (kept in the manifest only as a documented
     finding) is not indexed; chunks indexed before are removed.
  1. Compute the SHA-256 fingerprint of the file. If the registry (SQLite)
     says this exact file was already indexed in this collection with the
     same model and chunking settings, skip it. Re-running is idempotent.
  2. Parse (app.parsing) and clean (app.cleaning). If the language detected
     on the whole text is not in the setting `supported_languages` (English
     and French), skip the document with a message; chunks indexed before
     are removed. Then chunk (app.chunking).
  3. Embed every chunk with bge-m3: one dense + one sparse vector (app.embeddings).
  4. Delete the document's old points from Qdrant, insert the new ones
     (app.vectorstore), and update the registry (app.db).

--dry-run stops after step 2, prints statistics, and writes the chunks to
data/processed/chunks/<doc_id>.jsonl so they can be read and checked.

Ingestion runs on the GPU (setting ingest_embed_device), in its own process.
To leave the GPU memory to the embedder, it first asks Ollama to unload any
loaded LLM (Ollama reloads it on the next question).
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from sqlmodel import Session

from app.chunking import Chunk, HFTokenCounter, chunk_document
from app.cleaning import clean_blocks
from app.config import Settings, get_settings
from app.db import Document, find_indexed, forget_document, get_engine, record_indexed, save_document
from app.embeddings import Embedder, load_tokenizer
from app.langdetect import detect_language
from app.manifest import DocumentEntry, load_manifest
from app.parsing import parse_document
from app.vectorstore import count_points, delete_document, ensure_collection, upsert_chunks


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def remove_from_index(qdrant: QdrantClient, engine, collection: str, doc_id: str) -> int:
    """Delete a document's chunks from Qdrant and the registry. Return how many chunks were removed."""
    removed = count_points(qdrant, collection, doc_id)
    if removed:
        delete_document(qdrant, collection, doc_id)
    with Session(engine) as session:
        forget_document(session, doc_id, collection)
    return removed


def unload_ollama_models(ollama_url: str) -> None:
    """Ask Ollama to free the GPU (keep_alive=0 unloads a model). Ignored if Ollama is not running."""
    try:
        with httpx.Client(base_url=ollama_url, timeout=30) as client:
            for model in client.get("/api/ps").json()["models"]:
                client.post("/api/generate", json={"model": model["name"], "keep_alive": 0})
                print(f"Unloaded {model['name']} from Ollama to free the GPU for ingestion.")
    except httpx.HTTPError:
        pass


def chunk_payload(entry: DocumentEntry, chunk: Chunk) -> dict:
    """Everything Qdrant stores next to the vectors of one chunk."""
    return {
        "chunk_id": f"{entry.id}:{chunk.index}",
        "doc_id": entry.id,
        "chunk_index": chunk.index,
        "title": entry.title,
        "department": entry.department,
        "language": chunk.language,  # detected on the chunk itself
        "doc_language": entry.language,  # declared in the manifest
        "allowed_groups": entry.allowed_groups,
        "section": chunk.section,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "text": chunk.text,
        "n_tokens": chunk.n_tokens,
        "source_url": entry.source_url,
    }


def write_chunks_jsonl(path: Path, entry: DocumentEntry, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": f"{entry.id}:{c.index}", **asdict(c)}, ensure_ascii=False) + "\n")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # messages may contain any Unicode; Windows consoles default to cp1252
    s: Settings = get_settings()
    p = argparse.ArgumentParser(description="Index the manifest documents into Qdrant.")
    p.add_argument("--only", nargs="+", metavar="DOC_ID", help="only these document ids")
    p.add_argument("--collection", default=s.qdrant_collection)
    p.add_argument("--chunk-size", type=int, default=s.chunk_size_tokens)
    p.add_argument("--overlap", type=float, default=s.chunk_overlap_ratio, help="overlap ratio, e.g. 0.15")
    p.add_argument("--no-header", action="store_true", help="do not prepend 'Title > Section' to embedded text")
    p.add_argument("--force", action="store_true", help="re-index even if file and settings are unchanged")
    p.add_argument("--dry-run", action="store_true", help="parse and chunk only, write chunks to data/processed/")
    args = p.parse_args()

    add_header = s.chunk_add_header and not args.no_header
    settings_key = f"{s.embed_model}|size={args.chunk_size}|overlap={args.overlap}|header={int(add_header)}"
    manifest = load_manifest(s.manifest_path)
    entries = [manifest.get(i) for i in args.only] if args.only else manifest.documents
    counter = HFTokenCounter(load_tokenizer(s.embed_model))

    if not args.dry_run:
        if s.ingest_embed_device == "cuda":
            unload_ollama_models(s.ollama_url)
        embedder = Embedder(s.embed_model, s.ingest_embed_device, s.embed_fp16, s.embed_batch_size, s.embed_max_length)
        qdrant = QdrantClient(url=s.qdrant_url, timeout=60)
        ensure_collection(qdrant, args.collection)
        engine = get_engine(s.db_path)

    print(f"Collection: {args.collection} | settings: {settings_key}" + (" | DRY RUN" if args.dry_run else ""))
    rows = []
    notes = []  # one explanation line per document that is not indexed

    def removed_note(doc_id: str) -> str:
        """Remove chunks indexed before (not in a dry run) and say so."""
        removed = 0 if args.dry_run else remove_from_index(qdrant, engine, args.collection, doc_id)
        return f" ({removed} chunks indexed before were removed)" if removed else ""

    for entry in entries:
        path = entry.file_path(s.raw_dir)
        if entry.exclude:  # kept in the manifest as a documented finding, never indexed
            rows.append((entry.id, "excluded", None, []))
            notes.append(f"Excluded {entry.id}: {entry.exclude}{removed_note(entry.id)}")
            continue
        if not path.exists():
            rows.append((entry.id, "missing", None, []))
            continue
        start = time.perf_counter()
        sha = sha256_of(path)
        if not args.dry_run and not args.force:
            with Session(engine) as session:
                done = find_indexed(session, entry.id, args.collection)
            if done and done.sha256 == sha and done.settings_key == settings_key:
                rows.append((entry.id, "unchanged", None, []))
                continue

        parsed = parse_document(path)
        blocks = clean_blocks(parsed.blocks)
        language = detect_language("\n".join(b.text for b in blocks))
        if language not in s.supported_languages:
            rows.append((entry.id, "skipped", None, []))
            notes.append(f"Skipped {entry.id}: detected language '{language}' is not supported "
                         f"(supported: {', '.join(s.supported_languages)}){removed_note(entry.id)}")
            continue
        chunks = chunk_document(blocks, counter, entry.title, entry.language,
                                args.chunk_size, args.overlap, add_header)
        if not chunks:
            rows.append((entry.id, "no text", parsed.page_count, []))
            continue

        if args.dry_run:
            write_chunks_jsonl(s.data_dir / "processed" / "chunks" / f"{entry.id}.jsonl", entry, chunks)
            status = "chunked"
        else:
            embeddings = embedder.encode([c.embed_text for c in chunks])
            delete_document(qdrant, args.collection, entry.id)
            upsert_chunks(qdrant, args.collection, [chunk_payload(entry, c) for c in chunks], embeddings)
            with Session(engine) as session:
                record_indexed(session, entry.id, args.collection, sha, settings_key, len(chunks))
                save_document(session, Document(
                    doc_id=entry.id, title=entry.title, department=entry.department,
                    language=entry.language, allowed_groups=entry.allowed_groups,
                    path=entry.path, source_url=entry.source_url, page_count=parsed.page_count,
                ))
            status = "indexed"
        rows.append((entry.id, status, parsed.page_count, chunks, time.perf_counter() - start))

    print(f"\n{'id':<28} {'status':<10} {'pages':>5} {'chunks':>6} {'avg tok':>7} {'min':>5} {'max':>5} {'secs':>6}")
    print("-" * 80)
    for row in rows:
        doc_id, status, pages, chunks = row[:4]
        if chunks:
            tokens = [c.n_tokens for c in chunks]
            print(f"{doc_id:<28} {status:<10} {pages or '-':>5} {len(chunks):>6} {sum(tokens) / len(tokens):>7.0f} "
                  f"{min(tokens):>5} {max(tokens):>5} {row[4]:>6.1f}")
        else:
            print(f"{doc_id:<28} {status:<10}")
    if not args.dry_run:
        print(f"\nQdrant collection '{args.collection}' now holds {count_points(qdrant, args.collection)} chunks.")
    missing = [r[0] for r in rows if r[1] == "missing"]
    if missing:
        print(f"Missing files (run `python -m app.download`): {', '.join(missing)}")
    for note in notes:
        print(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
