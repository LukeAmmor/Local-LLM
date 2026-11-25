"""Central configuration for the whole project.

What: one `Settings` class holds every value that may change between
machines or experiments: model names, devices, URLs, chunking parameters.

Why: the same code must run on the Windows dev machine (GPU) and later on a
Linux server (maybe CPU only), and the experiments change chunk sizes and
retrieval stages. Instead of editing code, we change settings.

How: pydantic-settings reads, in this order of priority:
  1. environment variables prefixed with RAG_ (e.g. RAG_EMBED_DEVICE=cpu)
  2. the .env file at the repository root
  3. the defaults written below
Pydantic also validates types, so a typo like RAG_EMBED_DEVICE=gpu fails
immediately with a clear error instead of crashing later.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> parents[2] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]

Device = Literal["cuda", "cpu"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    data_dir: Path = REPO_ROOT / "data"

    # Default devices come from the Phase 0 measurements (docs/vram-budget.md):
    # on an 8 GB GPU only "embedder on CPU + reranker on GPU + 4k context" fits
    # without Windows paging GPU memory to RAM. One question embeds in ~0.3 s
    # on the CPU, but the reranker needs the GPU (~50 s for 30 passages on CPU).

    # --- Embedding model (dense + sparse vectors) ---
    embed_model: str = "BAAI/bge-m3"
    embed_device: Device = "cpu"
    embed_fp16: bool = True  # half precision halves memory; only used on cuda
    embed_batch_size: int = 16
    embed_max_length: int = 1024  # max tokens per text given to the embedder

    # --- Reranker (cross-encoder) ---
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: Device = "cuda"
    rerank_fp16: bool = True
    rerank_max_length: int = 1024  # max tokens for one (question, passage) pair

    # --- LLM served by Ollama ---
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"
    llm_num_ctx: int = 4096  # context window in tokens; drives the KV cache size in VRAM
    llm_keep_alive: str = "10m"  # how long Ollama keeps the model loaded after a request

    # --- Vector database ---
    qdrant_url: str = "http://localhost:6333"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, created once and then reused (cached)."""
    return Settings()
