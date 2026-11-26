# CLAUDE.md: project conventions

Private local LLM knowledge assistant (RAG) with access control. Solo
end-of-semester Deep Learning project, deadline end of December 2026, presented
with a recorded demo video. The student must be able to explain every line, so
simplicity beats cleverness.

## Working rules
- Work in phases (0 to 7). At the end of each phase: stop, summarise, show how to
  run and test it, wait for approval.
- After the student approves a phase: make ONE git commit with a clear message.
  Never commit `data/raw/`, `.env` or model weights.
- Ask before adding any package that is not in the stack below, or before
  changing an approach. Do not add features that were not asked for.
- No LangChain / LlamaIndex in the core pipeline. Small, readable custom code.
- Every module starts with a docstring: what it does, why, how.
- Everything configurable lives in `backend/app/config.py` (pydantic-settings,
  prefix `RAG_`, overridable from `.env`). No hard-coded model names, URLs,
  devices, chunk sizes or top-k values elsewhere.
- No Windows-only code: the system must also run on a Linux server (maybe CPU only).
- Repo is public and in English.

## Stack (approved)
Python 3.12 + uv · FastAPI · Pydantic v2 · pydantic-settings · SQLModel + SQLite ·
PyMuPDF · python-docx · FlagEmbedding (BAAI/bge-m3 dense + sparse,
BAAI/bge-reranker-v2-m3) · Qdrant (Docker) + qdrant-client · Ollama (qwen3:8b,
thinking disabled) over plain HTTP with httpx · pyyaml · pytest ·
React + Vite + TypeScript + Tailwind + shadcn/ui · RAGAS · sentence-transformers.

## Commands (run from the repository root)
```powershell
uv sync                                       # Windows + RTX 5060: CUDA torch (default group cu130)
docker compose up -d qdrant                   # Qdrant on 127.0.0.1:6333 (dashboard: /dashboard)
uv run python scripts/check_env.py            # Phase 0: GPU, Ollama, Qdrant
uv run python scripts/measure_vram.py --matrix --label apps-open
uv run python scripts/measure_vram.py --summary
uv run python -m app.download                 # download + check the manifest documents
uv run python -m app.ingest                   # parse, clean, chunk, embed, store (skips unchanged docs)
uv run python -m app.ingest --dry-run         # chunks only -> data/processed/chunks/<doc_id>.jsonl
uv run python -m app.ingest --force           # REQUIRED after changing parsing/cleaning/chunking code
uv run python -m app.ask "question" --groups hr                    # Phase 2: answer with citations
uv run python -m app.ask "question" --groups hr --mode sparse --no-rerank --no-generate --verbose
uv run pytest
```
The ingestion registry is keyed on the file hash + settings, not on our code:
after a code change in parsing, cleaning or chunking, re-index with `--force`.
Linux CPU server: `uv sync --no-default-groups --group cpu` (and run the venv's
python directly, or keep passing the same flags to `uv run`). torch is selected
through dependency groups, not extras, because `uv run` drops extras and would
silently swap the CUDA build for the CPU one.

## Layout
- `backend/app/`: the Python package `app` (installed editable, so
  `python -m app.<module>` works from the root)
  - ingestion pipeline: `manifest` -> `download` -> `parsing` -> `cleaning` ->
    `chunking` (+ `langdetect`) -> `embeddings` -> `vectorstore` (Qdrant) + `db` (SQLite registry);
    `ingest` is the CLI that runs it
  - question pipeline: `retrieval` (access filter, dense/sparse search, RRF) ->
    `rerank` -> `generation` (prompt, Ollama, citation check); `pipeline`
    (`RagPipeline`, reused by the API and the evaluation) ties them together;
    `ask` is the CLI
- `backend/tests/`: pytest
- `scripts/`: one-off environment and measurement scripts
- `data/manifest.yaml`: list of documents; `data/raw/`: downloaded files (gitignored)
- `eval/`, `finetune/`, `frontend/`, `docs/`: later phases

## Decisions log
- PyTorch from the cu130 index (torch 2.14.1). cu128 builds stop at torch 2.11;
  the driver supports CUDA 13.5. CPU index for the Linux server. Chosen with
  dependency groups `cu130` (default) / `cpu`.
- Models are resolved with `app.model_files.local_model_path()`: downloads only
  the PyTorch weights (bge-m3's 2.1 GiB ONNX copy is skipped) and passes a local
  folder to FlagEmbedding. Also accepts local folders (fine-tuned models).
- Qdrant pinned to v1.20.2, named Docker volume (no Windows bind mount), ports
  bound to 127.0.0.1.
- Ports 80, 8000 and 8080 are taken on the dev machine by other Docker stacks:
  backend default port will be 8100.
- Scope: English and French only (decided 2026-11-25, after Phase 1). Arabic
  was dropped: most Arabic PDFs gave broken text and the usable one needed
  font-specific repair code (docs/corpus.md, "Why Arabic was dropped"). The
  3 Arabic files stay in the manifest with `exclude: "Arabic dropped from
  scope"`. The Arabic repair code and download checks are in git history (8ab35ef).
- bge-m3 stays the embedder: multilingual, so French questions find English
  passages and back (cross-lingual search between the two languages).
- Language detection: small custom heuristic, no package. Less than half Latin
  letters -> "other"; otherwise English vs French stop-word vote ("unknown" on
  a tie or no letters). Setting `supported_languages` (default ["en", "fr"]):
  `app.ingest` skips a document whose detected language is not in it, with a
  message, and removes its old chunks. Language is also stored per chunk.
- Chunk header ("Title > Section" prepended to the embedded text, not shown):
  config switch, and an experiment (with vs without).
- DOCX: no pages at all, cited by section; in the UI too, DOCX citations show
  the section, not a page. Word's saved page-break markers were measured
  unreliable (145 markers for 151 pages, up to 8 pages of drift on
  wb-procurement-regs-fr). Page images only for PDFs.
- Manifest field `exclude: <reason>`: the document stays in the manifest as a
  documented finding, is still checked by `app.download`, never indexed by
  `app.ingest` (which removes chunks indexed before, from Qdrant and SQLite).
- Manifest `language` is any two-letter code; what gets indexed is decided by
  the setting `supported_languages`, not by the manifest schema.
- Manifest field `manual_download: true` for documents whose URL is a web page;
  the student places those files by hand.
- Phase 6: sentence-transformers fine-tunes only the dense part of bge-m3.
  Compare base dense vs fine-tuned dense; hybrid = fine-tuned dense + base sparse.
  `embeddings.py` keeps dense and sparse outputs separate for that reason.
  WARNING: FlagEmbedding silently creates a RANDOM sparse head when a model
  folder has no `sparse_linear.pt` (it only logs a message). A fine-tuned
  folder saved by sentence-transformers has none, so the sparse vectors must
  always come from the base model.
- If FlagEmbedding breaks with the newest transformers, pin transformers and say so.
- VRAM (Phase 0, docs/vram-budget.md): question time = embedder CPU, reranker
  GPU fp16, qwen3:8b GPU with num_ctx 4096. Ingestion = embedder on GPU in its
  own process with the LLM unloaded. Ollama runs with OLLAMA_FLASH_ATTENTION=1
  and OLLAMA_KV_CACHE_TYPE=q8_0 (user env vars). On Windows an over-full GPU
  pages to RAM instead of failing, and Ollama still reports "100% GPU": judge by
  speed (prefill about 2,700 tokens/s when healthy, about 300 when paging).
  Ollama does not see our PyTorch memory, so the budget is set through settings.
- Long-context experiments (800-token chunks): run retrieval, then generation as
  separate steps so that the LLM has the GPU to itself.
- Reranker on CPU takes about 50 s for 30 passages: a CPU-only server needs fewer
  candidates or no reranker (Phase 7).
- URLs use 127.0.0.1, never "localhost": on Windows localhost tries IPv6 first
  and Qdrant/Ollama listen on IPv4 only (+2 s per new connection, measured).
- `model_files.local_model_path` is offline-first: cached models never trigger
  a Hugging Face request.
- PyMuPDF 2.x hard-depends on pymupdf-layout, pymupdf4llm and onnxruntime. We do
  not use them (no layout model); they are only installed.
- PDF parsing keeps PyMuPDF's stored block order (no `sort=True`): sorting mixes
  two-column layouts line by line. Known cost: the procurement regulations draw
  chapter-title boxes after the first paragraphs of chapter pages.
- Headings: short blocks with font >= body + 1 pt, or bold. Levels by font size
  (sizes < 1 pt apart share a level, max 3). Adjacent same-level headings merge.
- PDF text is read with PyMuPDF's "dict" mode (blocks -> lines -> spans with
  font size and bold flag); no per-character extraction is needed.
- Headers/footers: text in the top/bottom 8% repeated on >= 3 pages (digits
  ignored), plus lone page numbers (digits or roman). Dot leaders collapsed.
- Chunking: sentences packed up to chunk_size; a heading cuts the chunk unless it
  is still < 25% of chunk_size; overlap (trailing sentences) only on size cuts;
  chunk section = section of its first non-heading sentence.
- PDF headings need >= 3 letters besides roman numerals/numbers ("III" alone is
  decoration); blocks that are only a roman numeral are dropped in cleaning.
- DOCX headings: Word outline level (paragraph, then style chain), then style
  name. Custom styles ("Style2", "REGs H1") are headings through outline levels.
- `manual_download: true` also for URLs that refuse scripts (403) or are dead (404).
- Phase 2 retrieval: the access filter (allowed_groups MatchAny user groups) is
  inside EVERY Qdrant query; `check_access` re-checks in Python (tripwire). The
  public group is always added to the user's groups. Hybrid = dense top 30 +
  sparse top 30 fused by our own RRF (k=60), so each candidate keeps its dense
  and sparse ranks (shown by `--verbose`). Then rerank, then top 5 to the LLM.
  Switches: `--mode dense|sparse|hybrid`, `--rerank/--no-rerank`, `--no-generate`.
- The reranker reads "Title > Section" + text when chunk_add_header is on
  (`chunking.with_header`, same text as the embedder).
- Generation: fixed "I don't know" sentence per language (says "the documents
  you have access to"); the question's language is detected and the prompt says
  "Write your answer in French/English"; "unknown" -> "in the language of the
  question". Temperature 0. Ollama `"truncate": false`: by default Ollama cuts an
  over-long prompt SILENTLY (measured: 5,126 tokens cut to 2,050 with num_ctx 4096);
  with false it returns HTTP 400. If no passage is found, the LLM is not called.
- Citations: `[n]`, `[1][3]`, `[1, 3]`; invalid numbers are removed from the
  answer and reported. The pipeline result keeps ALL passages given to the LLM,
  each with `cited` true/false (RAGAS needs them in Phase 5); the CLI (and later
  the UI) shows only the cited ones.
- `RagPipeline` warms the models up at load time (first runs are 3-5x slower),
  so per-question timings are comparable. Ollama reuses a cached identical
  prompt start: the prompt-reading speed is only meaningful for a new question.
- Rerank scores of the passage holding the answer can be very low across
  languages: 0.019 (French question, English ILO passage: "six mois") and 0.058
  (English question, French ANSSI passage), while other pairs score 0.85-1.0. A
  score threshold for "I don't know" would reject right answers. Not used;
  revisit with the Phase 5 test set.
- Corpus notes and extraction quality per document: docs/corpus.md.

## Reminders
- Before recording the demo (end of Phase 7): re-run the default VRAM
  measurement with OBS open AND recording, and check there is no paging:
  `uv run python scripts/measure_vram.py --embed-device cpu --rerank-device cuda --num-ctx 4096 --label obs-recording`
- Test set idea (Phase 5): a French question about harassment at work must
  retrieve the English ILO harassment policy (ilo-harassment-policy-en):
  cross-lingual recall. "Quelle est la procédure en cas de harcèlement sexuel au
  travail ?" (Phase 2): for a user with ALL groups, sparse alone misses it (top 5 =
  French it-security/procurement passages), hybrid + rerank gives 5/5 ILO
  passages. For hr + public only, sparse finds it at rank 1: the access filter
  removes the competing French documents. Run this test with all groups.
- Access test (Phase 2): "Dans quel délai dois-je déposer une plainte écrite pour
  harcèlement sexuel ?" -> hr: "six mois [n]"; procurement: "Je ne sais pas...".
- VRAM during Phase 2: the desktop apps (Steam, Wallpaper Engine, Discord...)
  varied between about 1.0 and 1.75 GB within minutes, and prompt reading
  dropped to 205-447 tokens/s (paging) in 2 of 12 runs, once even without the
  reranker. Close them for the demo.
- `scripts/measure_vram.py` still contains an Arabic sample passage: left as
  it is so the Phase 0 VRAM measurements stay comparable.

## Machine notes (dev)
RTX 5060 8 GB (Blackwell, compute capability 12.0), about 2 GB of VRAM used by
desktop apps when they are open. Ryzen 7 5700X, 32 GB RAM. Ollama native on
Windows at http://localhost:11434 (from containers: http://host.docker.internal:11434).
