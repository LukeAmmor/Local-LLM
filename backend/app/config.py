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

from pydantic import SecretStr
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

    @property
    def raw_dir(self) -> Path:
        """Downloaded documents, one folder per department (gitignored)."""
        return self.data_dir / "raw"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.yaml"

    @property
    def db_path(self) -> Path:
        """SQLite database: document registry, users and audit log."""
        return self.data_dir / "app.db"

    # Default devices come from the Phase 0 measurements (docs/vram-budget.md):
    # on an 8 GB GPU only "embedder on CPU + reranker on GPU + 4k context" fits
    # without Windows paging GPU memory to RAM. One question embeds in ~0.3 s
    # on the CPU, but the reranker needs the GPU (~50 s for 30 passages on CPU).

    # --- Languages ---
    # Documents detected as another language are skipped by ingestion
    # (app.langdetect knows English and French only).
    supported_languages: list[str] = ["en", "fr"]

    # --- Embedding model (dense + sparse vectors) ---
    # bge-m3 is multilingual: an English question finds French passages and back.
    embed_model: str = "BAAI/bge-m3"
    embed_device: Device = "cpu"
    embed_fp16: bool = True  # half precision halves memory; only used on cuda
    embed_batch_size: int = 16
    embed_max_length: int = 1024  # max tokens per text given to the embedder
    # Ingestion is a separate process that runs while the LLM is unloaded, so
    # it can use the GPU (about 35x faster than the CPU for many chunks).
    ingest_embed_device: Device = "cuda"

    # --- Chunking ---
    chunk_size_tokens: int = 400  # maximum chunk length, in bge-m3 tokens
    chunk_overlap_ratio: float = 0.15  # share of a chunk repeated at the start of the next one
    chunk_add_header: bool = True  # prepend "Title > Section" to the text that is embedded

    # --- Retrieval (question time) ---
    # dense = meaning, sparse = keywords, hybrid = both merged with RRF.
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    retrieval_top_k: int = 30  # candidates kept by the search (each search of hybrid returns this many)
    rrf_k: int = 60  # Reciprocal Rank Fusion constant: score = sum of 1 / (rrf_k + rank)
    context_passages: int = 5  # passages given to the LLM (best after reranking, or after the search)
    public_group: str = "public"  # every user may read the documents of this group

    # --- Reranker (cross-encoder) ---
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: Device = "cuda"
    rerank_fp16: bool = True
    rerank_batch_size: int = 32  # (question, passage) pairs scored at once
    rerank_max_length: int = 1024  # max tokens for one (question, passage) pair

    # URLs use 127.0.0.1, not "localhost": on Windows, "localhost" tries IPv6
    # (::1) first, and Ollama and Qdrant only listen on IPv4, which costs about
    # 2 s per new connection (measured in Phase 1).

    # --- LLM served by Ollama ---
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:8b"
    llm_num_ctx: int = 4096  # context window in tokens; drives the KV cache size in VRAM
    # How long Ollama keeps the model loaded after a request: a duration ("10m")
    # or a number of seconds; -1 = forever (the API server, so no question waits
    # for the model to load), 0 = unload at once.
    llm_keep_alive: str = "10m"
    llm_num_predict: int = 768  # maximum answer length in tokens
    llm_temperature: float = 0.0  # 0 = always the most likely token: same answer every run (experiments)
    llm_timeout_s: float = 300.0  # the first request also loads the model

    # --- Vector database ---
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "kb"  # experiments use other names, e.g. kb_c256

    # --- API server (Phase 3) ---
    # Ports 80, 8000 and 8080 are taken on the dev machine (other Docker stacks).
    api_host: str = "127.0.0.1"  # only this machine can connect
    api_port: int = 8100
    question_max_chars: int = 1000  # longer questions are refused (they would also fill the LLM context)
    page_image_dpi: int = 150  # source page images: an A4 page at 150 dpi is 1240 x 1754 pixels

    # --- Authentication (Phase 3) ---
    # The secret signs the login tokens: whoever knows it can forge a token for
    # any user. It lives in .env only, never in the repository, and the API
    # refuses to start without it. SecretStr prints as "**********".
    jwt_secret: SecretStr = SecretStr("")
    jwt_expire_minutes: int = 480  # a login is valid for one working day
    # Password of the demo users created by `python -m app.seed` (.env only).
    seed_password: SecretStr = SecretStr("")


@lru_cache
def get_settings() -> Settings:
    """Return the settings, created once and then reused (cached)."""
    return Settings()
