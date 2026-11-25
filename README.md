# Private Local LLM Knowledge Assistant (RAG) with Access Control

An assistant that answers questions from internal documents with a local LLM,
cites its sources down to the page, and only uses documents the user is
allowed to see. Documents and questions in English and French (a French
question can find an English passage, and the other way round).

> Work in progress. The full README (architecture, setup, results) comes in Phase 7.

## Quick start (development, Windows + NVIDIA GPU)

```powershell
uv sync                        # Linux CPU server: uv sync --no-default-groups --group cpu
docker compose up -d qdrant
uv run python scripts/check_env.py
uv run python -m app.download  # documents listed in data/manifest.yaml -> data/raw/
uv run python -m app.ingest    # index them into Qdrant
uv run pytest
```
