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
uv run pytest
```
Linux CPU server: `uv sync --no-default-groups --group cpu` (and run the venv's
python directly, or keep passing the same flags to `uv run`). torch is selected
through dependency groups, not extras, because `uv run` drops extras and would
silently swap the CUDA build for the CPU one.

## Layout
- `backend/app/`: the Python package `app` (installed editable, so
  `python -m app.<module>` works from the root)
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
- Language detection: small custom heuristic (Arabic script ratio, then
  English vs French stop words). No package.
- Chunk header ("Title > Section" prepended to the embedded text, not shown):
  config switch, and an experiment (with vs without).
- DOCX has no real pages: use Word's saved page-break markers when present,
  otherwise cite the section only. Page images only for PDFs.
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

## Reminders
- Before recording the demo (end of Phase 7): re-run the default VRAM
  measurement with OBS open AND recording, and check there is no paging:
  `uv run python scripts/measure_vram.py --embed-device cpu --rerank-device cuda --num-ctx 4096 --label obs-recording`

## Machine notes (dev)
RTX 5060 8 GB (Blackwell, compute capability 12.0), about 2 GB of VRAM used by
desktop apps when they are open. Ryzen 7 5700X, 32 GB RAM. Ollama native on
Windows at http://localhost:11434 (from containers: http://host.docker.internal:11434).
