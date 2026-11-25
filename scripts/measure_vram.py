"""Measure GPU memory (VRAM) and latency of the LLM, embedder and reranker together.

What: loads the three models in the same order as the real application and,
after each step, records:
  - the memory used on the WHOLE GPU, read with nvidia-smi (so it includes
    Ollama, which is a separate program, and the desktop apps),
  - the peak during the step (nvidia-smi is polled every 200 ms in a thread),
  - how long the step took.

Steps of one run:
  baseline       every Ollama model unloaded, nothing of ours on the GPU
  embedder       bge-m3 loaded, one question encoded
  ingest_batch   one batch of passages encoded (what `app.ingest` does), then the
                 PyTorch cache is emptied, because ingestion is a separate process
  reranker       bge-reranker-v2-m3 loaded, 30 (question, passage) pairs scored
  llm            Ollama loads the LLM and answers a prompt with 5 passages
  full_question  embed question -> rerank 30 -> generate, everything already loaded

Why: 8 GB must hold the three models. Two traps:
  - Ollama checks free VRAM when it loads a model. If VRAM is short it quietly
    keeps some layers on the CPU, which is slower. /api/ps tells us how much of
    the model stayed on the GPU.
  - On Windows, when the GPU is over-full the driver moves memory into normal
    RAM instead of failing. It still works, but much slower, and Ollama still
    reports "100% GPU". The real symptom is the speed, so the summary flags
    runs whose generation or prompt reading is below 60% of the best run.

One run = one configuration (devices + LLM context size). `--matrix` runs
several configurations, each in a fresh Python process, so memory from one run
cannot leak into the next. Results are appended to eval/results/phase0_vram.json.

Usage (models must be downloaded once before, see CLAUDE.md):
    uv run python scripts/measure_vram.py --matrix --label apps-open
    uv run python scripts/measure_vram.py --embed-device cpu --num-ctx 4096 --label try
    uv run python scripts/measure_vram.py --summary
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime

import httpx
import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from transformers import AutoTokenizer

from app.config import REPO_ROOT, Settings, get_settings
from app.model_files import local_model_path

RESULTS_FILE = REPO_ROOT / "eval" / "results" / "phase0_vram.json"
FULL_GPU_MARGIN_MIB = 300  # peak closer than this to the total = "GPU full" warning
PASSAGE_TOKENS = 450  # a 400-token chunk plus its "Title > Section" header
N_CANDIDATES = 30  # passages sent to the reranker
N_CONTEXT = 5  # passages given to the LLM

MATRIX = [
    {"embed_device": "cuda", "rerank_device": "cuda", "num_ctx": 8192},
    {"embed_device": "cuda", "rerank_device": "cuda", "num_ctx": 4096},
    {"embed_device": "cpu", "rerank_device": "cuda", "num_ctx": 8192},
    {"embed_device": "cpu", "rerank_device": "cuda", "num_ctx": 4096},
    # Reference: the LLM alone on the GPU, retrieval models on the CPU.
    {"embed_device": "cpu", "rerank_device": "cpu", "num_ctx": 8192},
]

# Filler text in the three languages. The content does not matter for memory,
# only the number of tokens does.
SENTENCES = [
    "The procurement officer must check that every contract above the threshold "
    "is approved by the committee before signature. ",
    "Le responsable des achats doit vérifier que chaque contrat au-dessus du seuil "
    "est approuvé par le comité avant la signature. ",
    "يجب على مسؤول المشتريات التحقق من أن كل عقد يتجاوز الحد المسموح به قد وافقت عليه اللجنة قبل التوقيع. ",
]
QUESTION = "Who must approve a contract above the threshold before it is signed?"


# --------------------------------------------------------------------------
# GPU memory monitor
# --------------------------------------------------------------------------
class GpuMonitor:
    """Reads whole-GPU memory use from nvidia-smi every 200 ms in a background thread."""

    def __init__(self, interval_ms: int = 200):
        self.total = self.current = self.peak = 0
        self._proc = subprocess.Popen(
            ["nvidia-smi", "-i", "0", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits", "-lms", str(interval_ms)],
            stdout=subprocess.PIPE,
            text=True,
        )
        threading.Thread(target=self._read_forever, daemon=True).start()
        deadline = time.time() + 10
        while self.total == 0:  # wait for the first reading
            if time.time() > deadline:
                raise RuntimeError("nvidia-smi gave no reading")
            time.sleep(0.05)

    def _read_forever(self) -> None:
        for line in self._proc.stdout:
            used, total = (int(x) for x in line.strip().split(","))
            self.current, self.total = used, total
            self.peak = max(self.peak, used)

    def start_step(self) -> None:
        """Begin a new step: the peak restarts from the current value."""
        time.sleep(0.4)  # let a fresh reading arrive
        self.peak = self.current

    def stop(self) -> None:
        self._proc.terminate()


# --------------------------------------------------------------------------
# Ollama helpers (plain HTTP)
# --------------------------------------------------------------------------
def ollama_loaded(client: httpx.Client) -> list[dict]:
    """Models currently loaded by Ollama, with their size and size on the GPU."""
    return client.get("/api/ps").json()["models"]


def ollama_unload_all(client: httpx.Client) -> None:
    """Unload every Ollama model (keep_alive=0) and wait until memory is released."""
    for m in ollama_loaded(client):
        client.post("/api/generate", json={"model": m["name"], "keep_alive": 0})
    for _ in range(100):
        if not ollama_loaded(client):
            break
        time.sleep(0.2)
    time.sleep(1.0)  # the driver needs a moment to report the freed memory


def ollama_chat(client: httpx.Client, model: str, prompt: str, num_ctx: int, num_predict: int) -> dict:
    r = client.post(
        "/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "think": False,
            "stream": False,
            "keep_alive": "5m",
            "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0},
        },
    )
    r.raise_for_status()
    return r.json()


def build_prompt(passages: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return (
        "Answer the question using only the passages below and cite them like [1].\n\n"
        f"{context}\n\nQuestion: {QUESTION}\nAnswer in about 150 words."
    )


# --------------------------------------------------------------------------
# One measurement run
# --------------------------------------------------------------------------
def make_passages(tokenizer, n: int, n_tokens: int) -> list[str]:
    """n passages of about n_tokens tokens each, cycling through the 3 languages."""
    passages = []
    for i in range(n):
        sentence = SENTENCES[i % len(SENTENCES)]
        text = sentence
        while len(tokenizer(text)["input_ids"]) < n_tokens:
            text += sentence
        passages.append(text)
    return passages


def run_once(s: Settings, embed_device: str, rerank_device: str, num_ctx: int,
             label: str, num_predict: int) -> dict:
    uses_cuda = "cuda" in (embed_device, rerank_device)
    gpu = GpuMonitor()
    ollama = httpx.Client(base_url=s.ollama_url, timeout=600)
    steps: list[dict] = []

    def step(name: str, fn):
        """Run fn(), record its duration and the whole-GPU memory."""
        gpu.start_step()
        if uses_cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn()
        seconds = time.perf_counter() - t0
        time.sleep(0.5)  # one more nvidia-smi reading after the step
        record = {"step": name, "seconds": round(seconds, 2),
                  "gpu_used_mib": gpu.current, "gpu_peak_mib": gpu.peak}
        if uses_cuda:
            record["torch_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        steps.append(record)
        print(f"  {name:<14} {seconds:7.2f} s   GPU used {gpu.current:5d} MiB  (peak {gpu.peak:5d})")
        return out

    print(f"\n=== {label}: embed={embed_device} rerank={rerank_device} "
          f"num_ctx={num_ctx} llm={s.llm_model} ===")
    embed_path = local_model_path(s.embed_model)
    rerank_path = local_model_path(s.rerank_model)
    tokenizer = AutoTokenizer.from_pretrained(embed_path)
    passages = make_passages(tokenizer, N_CANDIDATES, PASSAGE_TOKENS)
    prompt = build_prompt(passages[:N_CONTEXT])
    # The full question uses OTHER passages: Ollama caches the last prompt, and
    # a real question always comes with a new context.
    prompt_question = build_prompt(passages[N_CONTEXT : 2 * N_CONTEXT])

    step("baseline", lambda: ollama_unload_all(ollama))

    def load_embedder():
        model = BGEM3FlagModel(embed_path, use_fp16=s.embed_fp16 and embed_device == "cuda",
                               devices=embed_device)
        # FlagEmbedding moves the weights to the device on the first encode.
        model.encode([QUESTION], return_dense=True, return_sparse=True)
        return model

    embedder = step("embedder", load_embedder)

    def ingest_batch():
        embedder.encode(passages[: s.embed_batch_size], batch_size=s.embed_batch_size,
                        max_length=s.embed_max_length, return_dense=True, return_sparse=True)
        if embed_device == "cuda":
            torch.cuda.empty_cache()  # give the batch memory back, like the end of an ingest process

    step("ingest_batch", ingest_batch)

    pairs = [[QUESTION, p] for p in passages]

    def load_reranker():
        model = FlagReranker(rerank_path, use_fp16=s.rerank_fp16 and rerank_device == "cuda",
                             devices=rerank_device)
        model.compute_score(pairs, batch_size=N_CANDIDATES, max_length=s.rerank_max_length)
        return model

    reranker = step("reranker", load_reranker)
    first = step("llm", lambda: ollama_chat(ollama, s.llm_model, prompt, num_ctx, num_predict))
    ps = next((m for m in ollama_loaded(ollama) if m["name"] == s.llm_model), {})

    def full_question():
        t = {}
        t0 = time.perf_counter()
        embedder.encode([QUESTION], return_dense=True, return_sparse=True)
        t["embed_query_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        reranker.compute_score(pairs, batch_size=N_CANDIDATES, max_length=s.rerank_max_length)
        t["rerank_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = ollama_chat(ollama, s.llm_model, prompt_question, num_ctx, num_predict)
        t["llm_s"] = time.perf_counter() - t0
        t["llm"] = out
        return t

    q = step("full_question", full_question)
    gpu.stop()
    ollama_unload_all(ollama)  # leave the GPU free

    by_name = {r["step"]: r for r in steps}
    peak = max(r["gpu_peak_mib"] for r in steps)
    llm = q["llm"]
    size, size_vram = ps.get("size", 0), ps.get("size_vram", 0)
    return {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "llm_model": s.llm_model,
        "embed_device": embed_device,
        "rerank_device": rerank_device,
        "num_ctx": num_ctx,
        "gpu_total_mib": gpu.total,
        "baseline_mib": by_name["baseline"]["gpu_used_mib"],
        "peak_mib": peak,
        "gpu_full_warning": peak >= gpu.total - FULL_GPU_MARGIN_MIB,
        # Memory added by each component (whole-GPU difference between steps).
        # The first component on CUDA also pays the CUDA context (~300-500 MiB).
        "embedder_mib": by_name["embedder"]["gpu_used_mib"] - by_name["baseline"]["gpu_used_mib"],
        "ingest_peak_mib": by_name["ingest_batch"]["gpu_peak_mib"] - by_name["baseline"]["gpu_used_mib"],
        "reranker_mib": by_name["reranker"]["gpu_used_mib"] - by_name["ingest_batch"]["gpu_used_mib"],
        "llm_mib": by_name["llm"]["gpu_used_mib"] - by_name["reranker"]["gpu_used_mib"],
        "llm_size_mib": round(size / 2**20),
        "llm_on_gpu_pct": round(100 * size_vram / size, 1) if size else None,
        "llm_load_s": round(first.get("load_duration", 0) / 1e9, 2),
        "prompt_tokens": llm.get("prompt_eval_count"),
        "prefill_tokens_per_s": round(llm["prompt_eval_count"] / (llm["prompt_eval_duration"] / 1e9), 1),
        "output_tokens": llm.get("eval_count"),
        "llm_tokens_per_s": round(llm["eval_count"] / (llm["eval_duration"] / 1e9), 1),
        "embed_query_s": round(q["embed_query_s"], 3),
        "rerank_30_s": round(q["rerank_s"], 3),
        "llm_answer_s": round(q["llm_s"], 2),
        "question_total_s": round(q["embed_query_s"] + q["rerank_s"] + q["llm_s"], 2),
        "steps": steps,
    }


# --------------------------------------------------------------------------
# Results file and summary table
# --------------------------------------------------------------------------
def load_results() -> list[dict]:
    return json.loads(RESULTS_FILE.read_text(encoding="utf-8")) if RESULTS_FILE.exists() else []


def save_result(result: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = load_results() + [result]
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(results: list[dict]) -> None:
    """Print one line per run.

    Warning flags:
      GPU full  the peak came within FULL_GPU_MARGIN_MIB of the total.
      paging    generation or prompt reading (prefill) was below 60% of the
                fastest run with the same label and model. On Windows this is
                the real sign that VRAM was over-full: the driver moved memory
                to normal RAM, even though Ollama still reports "100% GPU".
                Prefill is checked too because it slows down first.
    """
    best_gen, best_prefill = {}, {}
    for r in results:
        key = (r["label"], r["llm_model"])
        best_gen[key] = max(best_gen.get(key, 0), r["llm_tokens_per_s"])
        best_prefill[key] = max(best_prefill.get(key, 0), r["prefill_tokens_per_s"])

    header = (f"{'label':<12} {'embed':<5} {'rerank':<6} {'ctx':>5} | {'base':>5} {'peak':>5} {'free':>5} | "
              f"{'emb':>5} {'rrk':>5} {'LLMsz':>5} {'LLM@GPU':>7} | {'tok/s':>6} {'prefill':>7} {'embQ s':>6} "
              f"{'rrk30 s':>7} {'answer s':>8} {'total s':>7}")
    print("\n" + header + "\n" + "-" * len(header))
    for r in results:
        warn = ""
        if r["gpu_full_warning"]:
            warn += "  <- GPU full"
        key = (r["label"], r["llm_model"])
        if (r["llm_tokens_per_s"] < 0.6 * best_gen[key]
                or r["prefill_tokens_per_s"] < 0.6 * best_prefill[key]):
            warn += "  <- paging"
        print(f"{r['label']:<12} {r['embed_device']:<5} {r['rerank_device']:<6} {r['num_ctx']:>5} | "
              f"{r['baseline_mib']:>5} {r['peak_mib']:>5} {r['gpu_total_mib'] - r['peak_mib']:>5} | "
              f"{r['embedder_mib']:>5} {r['reranker_mib']:>5} {r['llm_size_mib']:>5} {r['llm_on_gpu_pct']:>6}% | "
              f"{r['llm_tokens_per_s']:>6} {r['prefill_tokens_per_s']:>7.0f} {r['embed_query_s']:>6} "
              f"{r['rerank_30_s']:>7} {r['llm_answer_s']:>8} {r['question_total_s']:>7}{warn}")
    print("Memory in MiB, whole GPU. emb/rrk = memory added by the embedder/reranker; "
          "LLMsz = memory Ollama allocated for the LLM; LLM@GPU = share Ollama says is on the GPU. "
          "tok/s = generation speed, prefill = prompt reading speed (tokens/s).")


def main() -> int:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embed-device", choices=["cuda", "cpu"], default=s.embed_device)
    p.add_argument("--rerank-device", choices=["cuda", "cpu"], default=s.rerank_device)
    p.add_argument("--num-ctx", type=int, default=s.llm_num_ctx)
    p.add_argument("--num-predict", type=int, default=256, help="max output tokens of the LLM")
    p.add_argument("--label", default="run", help="name of this measurement, e.g. apps-open")
    p.add_argument("--matrix", action="store_true", help="run every configuration in MATRIX")
    p.add_argument("--summary", action="store_true", help="only print the saved results")
    args = p.parse_args()

    if args.summary:
        print_summary(load_results())
        return 0

    if args.matrix:
        # Each configuration in its own process: a clean GPU every time.
        for cfg in MATRIX:
            subprocess.run(
                [sys.executable, __file__, "--label", args.label, "--num-predict", str(args.num_predict),
                 "--embed-device", cfg["embed_device"], "--rerank-device", cfg["rerank_device"],
                 "--num-ctx", str(cfg["num_ctx"])],
                check=True,
            )
        print_summary([r for r in load_results() if r["label"] == args.label])
        return 0

    result = run_once(s, args.embed_device, args.rerank_device, args.num_ctx, args.label, args.num_predict)
    save_result(result)
    print_summary([result])
    return 0


if __name__ == "__main__":
    sys.exit(main())
