"""Phase 0 environment check: GPU + PyTorch, Ollama, Qdrant.

What: runs three independent checks and prints PASS/FAIL for each line.
  1. PyTorch: is it a CUDA build, does it see the GPU, does the GPU really
     compute (an fp16 matrix product compared with the CPU result)?
  2. Ollama: does the server answer, is the configured model pulled, and does
     it answer WITHOUT thinking when we send "think": false?
  3. Qdrant: does the server answer, and can it store a point with a dense
     AND a sparse vector and run a hybrid query (prefetch + RRF fusion)?

Why: every later phase depends on these three. Checking them first, in one
place, means a failure later is a bug in our code, not in the setup.

Usage:
    uv run python scripts/check_env.py
Exit code 0 when everything passes, 1 otherwise.
"""

import sys
import time

import httpx
import torch
from qdrant_client import QdrantClient, models

from app.config import get_settings

results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    """Remember and print one check result."""
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def check_torch() -> None:
    print("\n1. PyTorch / GPU")
    report("torch version", True, f"{torch.__version__} (CUDA build: {torch.version.cuda})")
    if not torch.cuda.is_available():
        report("CUDA available", False, "torch does not see a GPU (CPU-only build or driver issue)")
        return
    report("CUDA available", True, torch.cuda.get_device_name(0))

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    # Blackwell consumer GPUs are compute capability 12.0 (sm_120). If the
    # PyTorch build has no kernels for it, every GPU operation fails.
    report("GPU architecture supported", arch in supported, f"{arch}; build supports {', '.join(supported)}")

    free, total = torch.cuda.mem_get_info(0)
    report("VRAM", True, f"{(total - free) / 2**20:.0f} MiB used / {total / 2**20:.0f} MiB total (whole GPU)")

    # A real computation: same matrix product on GPU (fp16) and CPU (fp32).
    a = torch.randn(1024, 1024)
    b = torch.randn(1024, 1024)
    gpu = (a.cuda().half() @ b.cuda().half()).float().cpu()
    cpu = a @ b
    max_err = (gpu - cpu).abs().max().item()
    report("fp16 matmul on GPU", max_err < 1.0, f"max abs difference vs CPU = {max_err:.3f}")


def check_ollama(url: str, model: str) -> None:
    print("\n2. Ollama")
    try:
        with httpx.Client(base_url=url, timeout=300) as client:
            version = client.get("/api/version").json()["version"]
            report("server reachable", True, f"{url} (version {version})")

            names = [m["name"] for m in client.get("/api/tags").json()["models"]]
            report(f"model '{model}' pulled", model in names, f"available: {', '.join(names)}")
            if model not in names:
                return

            start = time.perf_counter()
            r = client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
                    "think": False,  # qwen3 is a "thinking" model; we disable that mode
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 20},
                },
            )
            r.raise_for_status()
            data = r.json()
            elapsed = time.perf_counter() - start
            answer = data["message"]["content"].strip()
            thinking = data["message"].get("thinking", "")
            report("generation works", "paris" in answer.lower(), f"answer={answer!r} ({elapsed:.1f} s incl. model load)")
            report("thinking disabled", "<think>" not in answer and not thinking, "no <think> block, no thinking field")
            if data.get("eval_duration"):
                tps = data["eval_count"] / (data["eval_duration"] / 1e9)
                report("generation speed", True, f"{tps:.1f} tokens/s (short answer, indicative only)")
    except httpx.HTTPError as exc:
        report("server reachable", False, f"{url}: {exc}")


def check_qdrant(url: str) -> None:
    print("\n3. Qdrant")
    name = "phase0_check"
    try:
        client = QdrantClient(url=url, timeout=10)
        client.get_collections()
        report("server reachable", True, url)

        if client.collection_exists(name):
            client.delete_collection(name)
        # Same shape as the real collection: a named dense vector + a named sparse vector.
        client.create_collection(
            name,
            vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        client.upsert(
            name,
            points=[
                models.PointStruct(
                    id=1,
                    vector={
                        "dense": [0.1, 0.2, 0.3, 0.4],
                        "sparse": models.SparseVector(indices=[7, 42], values=[0.8, 0.3]),
                    },
                    payload={"allowed_groups": ["public"]},
                ),
                models.PointStruct(
                    id=2,
                    vector={
                        "dense": [0.9, -0.1, 0.0, 0.1],
                        "sparse": models.SparseVector(indices=[3], values=[0.5]),
                    },
                    payload={"allowed_groups": ["hr"]},
                ),
            ],
        )
        # Hybrid query: two prefetches (dense, sparse) fused with RRF, with an
        # access filter applied inside each prefetch.
        only_public = models.Filter(
            must=[models.FieldCondition(key="allowed_groups", match=models.MatchAny(any=["public"]))]
        )
        res = client.query_points(
            name,
            prefetch=[
                models.Prefetch(query=[0.9, -0.1, 0.0, 0.1], using="dense", filter=only_public, limit=5),
                models.Prefetch(query=models.SparseVector(indices=[3], values=[1.0]), using="sparse", filter=only_public, limit=5),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=5,
        )
        ids = [p.id for p in res.points]
        # Point 2 is the best match but belongs to "hr": the filter must remove it.
        report("hybrid query + access filter", ids == [1], f"returned ids {ids} (expected [1])")
        client.delete_collection(name)
    except Exception as exc:  # noqa: BLE001 - any failure here means "Qdrant not usable"
        report("server reachable", False, f"{url}: {exc}")


def main() -> int:
    settings = get_settings()
    check_torch()
    check_ollama(settings.ollama_url, settings.llm_model)
    check_qdrant(settings.qdrant_url)
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
